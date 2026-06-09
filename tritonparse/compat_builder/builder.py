# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
Core CompatBuilder class for the compat_builder workflow.

Orchestrates the process of building a commits.csv for a single LLVM bump:
  1. initialize() — create/reuse a dedicated git worktree + compat branch
  2. Loop:
       find_next_incompatible() — binary-search LLVM history for first bad commit
       record_pair()            — record (triton_commit, llvm_last_compatible)
       fix_incompatibility()    — AI-first, manual fallback
  3. generate_csv()             — write the final CSV with metadata header
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable, TYPE_CHECKING
from uuid import uuid4

from tritonparse.bisect.executor import CommandResult, ShellExecutor
from tritonparse.bisect.logger import BisectLogger
from tritonparse.compat_builder.state import CompatBuildPhase, CompatBuildState

if TYPE_CHECKING:
    from tritonparse.compat_builder.ai_fixer import AICompatFixer

# LLVM hash file location inside Triton repo
_LLVM_HASH_FILE = "cmake/llvm-hash.txt"

# Canonical LLVM source repo path within a Triton worktree.
# Note: .llvm-project/ is the *build output* directory; the actual git clone
# created by scripts/build-llvm-project.sh lives at llvm-project/ (without dot).
_LLVM_SUBDIR = "llvm-project"

# Compat branch prefix
_COMPAT_BRANCH_PREFIX = "compat"

# CSV schema version
_CSV_SCHEMA_VERSION = "1"

_CXX_FOR_CC: dict[str, str] = {"gcc": "g++", "clang": "clang++"}


class WaitingForFixError(Exception):
    """
    Raised when fix_incompatibility() cannot proceed automatically.

    Indicates the workflow is paused. The caller should:
    1. Inspect state_path to understand what failed.
    2. Apply a fix commit on the compat branch.
    3. Resume with CompatBuilder.apply_fix(fix_commit).
    """

    def __init__(
        self,
        state_path: Path,
        incompatible_llvm: str,
        build_error_log: str | None = None,
    ) -> None:
        self.state_path = state_path
        self.incompatible_llvm = incompatible_llvm
        self.build_error_log = build_error_log
        short = incompatible_llvm[:12]
        super().__init__(
            f"Manual fix required for LLVM {short}. State saved to {state_path}"
        )


class CompatBuilder:
    """
    Builds the compatibility map between Triton commits and LLVM commits
    for a single LLVM bump.

    The builder runs inside a dedicated git worktree to avoid polluting the
    user's main working tree (decision 8.9). All fix commits are created on
    a compat branch within that worktree.

    Usage::

        logger = BisectLogger("./logs")
        builder = CompatBuilder(
            triton_dir="/path/to/triton",
            llvm_bump_commit="abc123",
            output_csv="./commits.csv",
            logger=logger,
        )
        builder.initialize()
        while not builder.is_complete():
            incompatible = builder.find_next_incompatible()
            if incompatible is None:
                break
            builder.record_pair()
            try:
                builder.fix_incompatibility(incompatible)
            except WaitingForFixError:
                raise  # CLI handles --resume flow
        csv_path = builder.generate_csv()
    """

    def __init__(
        self,
        triton_dir: str,
        llvm_bump_commit: str,
        output_csv: str,
        logger: BisectLogger,
        conda_env: str = "triton_bisect",
        worktree_root: str | None = None,
        worktree_path: str | None = None,
        ai_fixer_factory: Callable[[str], AICompatFixer] | None = None,
        compiler: str | None = None,
    ) -> None:
        self.triton_dir = triton_dir
        self.llvm_bump_commit = llvm_bump_commit
        self.output_csv = output_csv
        self.logger = logger
        self.conda_env = conda_env
        self.conda_prefix: str | None = None
        self.worktree_root = worktree_root
        self._worktree_path = worktree_path
        self.ai_fixer_factory = ai_fixer_factory
        cc = compiler or os.environ.get("TRITON_BISECT_COMPILER") or "clang"
        self._cc = cc
        self._cxx = _CXX_FOR_CC.get(cc, cc)
        self.executor = ShellExecutor(logger)
        self.state: CompatBuildState | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize the compat-build workflow.

        Steps:
        1. Resolve conda environment prefix (fail-fast if not found).
        2. Resolve or create a dedicated git worktree.
        3. Checkout llvm_bump~1 and create compat/<bump> branch.
        4. Ensure the LLVM repo exists at <worktree>/llvm-project.
        5. Read old_llvm and new_llvm from llvm-hash.txt.
        6. Create initial CompatBuildState.
        """
        self.logger.info(f"Initializing compat-build for bump: {self.llvm_bump_commit}")

        self.conda_prefix = self._resolve_conda_prefix()
        self._validate_conda_env()

        worktree_path = self._resolve_worktree_path()
        self._setup_worktree(worktree_path)

        old_llvm = self._get_llvm_hash_at_commit(
            f"{self.llvm_bump_commit}~1", worktree_path
        )
        new_llvm = self._get_llvm_hash_at_commit(self.llvm_bump_commit, worktree_path)
        self.logger.info(f"LLVM range: {old_llvm[:12]} → {new_llvm[:12]}")

        self._ensure_llvm_repo(old_llvm, worktree_path)

        parent_triton = self._resolve_commit(
            f"{self.llvm_bump_commit}~1", worktree_path
        )

        self.state = CompatBuildState(
            triton_dir=self.triton_dir,
            llvm_bump_commit=self.llvm_bump_commit,
            output_csv=self.output_csv,
            conda_env=self.conda_env,
            worktree_root=self.worktree_root,
            worktree_path=str(worktree_path),
            old_llvm=old_llvm,
            new_llvm=new_llvm,
            current_triton=parent_triton,
            current_llvm_good=old_llvm,
            phase=CompatBuildPhase.FINDING_INCOMPATIBLE,
        )
        self.logger.info("Initialization complete.")

    def find_next_incompatible(
        self, output_callback: Callable[[str], None] | None = None
    ) -> str | None:
        """
        Binary-search LLVM history for the next incompatible commit.

        Uses git bisect on the llvm-project repo between
        current_llvm_good (good) and new_llvm (bad). The compat probe
        runs three steps for each tested LLVM commit:
          1. build:  scripts/build-llvm-project.sh + pip install -e .
          2. import: python -c "import triton"
          3. compile smoke: fixed kernel compilation

        Returns:
            The first incompatible LLVM commit hash, or None if all commits
            in [current_llvm_good, new_llvm] are compatible.
        """
        state = self._require_state()
        state.phase = CompatBuildPhase.FINDING_INCOMPATIBLE

        worktree = state.worktree_path
        if worktree is None:
            raise RuntimeError("worktree_path not set; call initialize() first")
        llvm_dir = str(Path(worktree) / _LLVM_SUBDIR)

        good = state.current_llvm_good
        bad = state.new_llvm
        if good is None or bad is None:
            raise RuntimeError("old_llvm/new_llvm not set; call initialize() first")

        if good == bad:
            self.logger.info("No LLVM range to search; current_llvm_good == new_llvm")
            return None

        self.logger.info(
            f"Searching for incompatible LLVM in [{good[:12]}, {bad[:12]}]"
        )

        conda_prefix = self.conda_prefix
        if conda_prefix is None:
            raise RuntimeError("conda_prefix not resolved; call initialize() first")

        # Clean untracked files that may conflict with LLVM checkouts
        clean = self.executor.run_command(
            ["git", "clean", "-fd"],
            cwd=llvm_dir,
        )
        if not clean.success:
            self.logger.warning(f"git clean failed in {llvm_dir}: {clean.stderr}")

        probe_script = self._write_compat_probe_script(worktree)
        result = self.executor.run_git_bisect_sequence(
            repo_path=llvm_dir,
            good_commit=good,
            bad_commit=bad,
            run_script=probe_script,
            env={"TRITON_WORKTREE": worktree, "CONDA_PREFIX": conda_prefix},
            output_callback=output_callback,
        )

        if not result.success:
            incompatible = self._parse_first_bad(result.stdout)
            if incompatible is None:
                raise RuntimeError(
                    f"git bisect failed (exit {result.exit_code}): "
                    f"{result.stderr or result.stdout[-500:]}"
                )
        else:
            incompatible = self._parse_first_bad(result.stdout)

        if incompatible is None:
            self.logger.info("All LLVM commits in range are compatible.")
            state.current_llvm_good = bad
            self.record_pair()
            return None

        # Verify the bisect result and capture a clean build error log.
        # The bisect output (result.output) contains build errors from ALL
        # tested LLVM commits — including commits after the first-bad that
        # have unrelated API changes. Feeding that to the AI causes it to
        # fix errors from future breakpoints. This single-commit probe both
        # verifies the bisect result (catching false positives from AI fixes
        # that made the bad boundary compatible) and captures only the errors
        # relevant to this specific incompatibility.
        probe_result = self._probe_llvm(worktree, incompatible, conda_prefix)
        if probe_result.success:
            self.logger.info(
                f"LLVM {incompatible[:12]} is actually compatible "
                f"(bisect false positive); advancing good boundary."
            )
            state.current_llvm_good = incompatible
            self.record_pair()
            return None

        last_good = self._get_prev_commit(llvm_dir, incompatible)
        state.current_llvm_good = last_good
        state.last_incompatible_llvm = incompatible
        log_path = self._save_build_error_log(probe_result.output)
        state.last_build_error_log = str(log_path) if log_path else None
        self.logger.info(f"Found incompatible LLVM: {incompatible[:12]}")
        return incompatible

    def record_pair(self) -> None:
        """
        Record the current (triton_commit, llvm_last_compatible) pair.

        Called after find_next_incompatible() returns (both None and non-None
        paths), and before fix_incompatibility() when applicable.
        """
        state = self._require_state()
        triton = state.current_triton
        llvm_good = state.current_llvm_good
        if triton is None or llvm_good is None:
            raise RuntimeError(
                "current_triton/current_llvm_good not set; call initialize() first"
            )
        state.add_pair(triton, llvm_good)
        self.logger.info(
            f"Recorded pair: triton={triton[:12]} llvm_last_good={llvm_good[:12]}"
        )

    _MAX_FIX_ATTEMPTS: int = 5

    def fix_incompatibility(self, incompatible_llvm: str) -> str:
        """Attempt to fix the incompatibility with a verify-and-retry loop.

        For each attempt:
          1. AI reads the build error log + LLVM diff and produces a fix commit
          2. Probe tests the fix against the incompatible LLVM commit
          3. If probe passes, apply the fix and return the commit
          4. If probe fails, save the new build output to a log file and pass
             that path to the next AI attempt

        Falls back to WaitingForFixError after max attempts or if AI fails.

        Args:
            incompatible_llvm: The first incompatible LLVM commit hash.

        Returns:
            The fix commit hash (only if AI fix + verification succeeds).

        Raises:
            WaitingForFixError: When manual intervention is required.
        """
        state = self._require_state()

        if self.ai_fixer_factory is None:
            return self._raise_waiting_for_fix(state, incompatible_llvm)

        state.phase = CompatBuildPhase.AI_FIXING
        error_log: Path | None = (
            Path(state.last_build_error_log) if state.last_build_error_log else None
        )
        pre_fix_head = state.current_triton

        for attempt in range(self._MAX_FIX_ATTEMPTS):
            self.logger.info(
                f"AI fix attempt {attempt + 1}/{self._MAX_FIX_ATTEMPTS} "
                f"for LLVM {incompatible_llvm[:12]}"
            )

            fix_commit = self._try_ai_fix(incompatible_llvm, state, error_log)
            if fix_commit is None:
                self.logger.info("AI did not produce a fix; falling back.")
                break

            self.logger.info(
                f"Verifying fix {fix_commit[:12]} against LLVM {incompatible_llvm[:12]}"
            )
            success, new_error_log = self._verify_fix(fix_commit, incompatible_llvm)

            if success:
                self.logger.info(f"Fix verified successfully on attempt {attempt + 1}")
                self._reattach_compat_branch(fix_commit)
                self.apply_fix(fix_commit)
                return fix_commit

            self.logger.info(f"Fix incomplete — see {new_error_log} for new errors")
            self._reattach_compat_branch(fix_commit)
            error_log = new_error_log

        self._reset_to_pre_fix(pre_fix_head, state)
        return self._raise_waiting_for_fix(state, incompatible_llvm)

    def apply_fix(self, fix_commit: str) -> None:
        """
        Apply a (user-provided or AI-generated) fix commit.

        Advances current_triton to fix_commit and resets ai_fix_attempted
        so that the next incompatibility gets a fresh AI attempt.

        Args:
            fix_commit: The hash of the fix commit on the compat branch.
        """
        state = self._require_state()
        state.current_triton = fix_commit
        state.ai_fix_attempted = False
        state.last_incompatible_llvm = None
        state.last_build_error_log = None
        state.phase = CompatBuildPhase.FINDING_INCOMPATIBLE
        self.logger.info(f"Applied fix commit: {fix_commit[:12]}")

    def is_complete(self) -> bool:
        """Return True if the entire LLVM range has been covered."""
        state = self._require_state()
        return (
            state.current_llvm_good is not None
            and state.new_llvm is not None
            and state.current_llvm_good == state.new_llvm
        )

    def generate_csv(self) -> Path:
        """
        Write the final commits.csv for this LLVM bump.

        CSV format (decision 8.6/8.7):
          - Metadata header with # key=value lines
          - terminal bad boundary in metadata (not a data row)
          - Data rows: triton_commit,llvm_commit_last_compatible

        Returns:
            Path to the written CSV file.
        """
        state = self._require_state()
        state.phase = CompatBuildPhase.COMPLETED

        output_path = Path(state.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            f"# schema_version={_CSV_SCHEMA_VERSION}",
            f"# llvm_bump_commit={state.llvm_bump_commit}",
            f"# old_llvm={state.old_llvm}",
            f"# new_llvm={state.new_llvm}",
            f"# final_bad_triton_commit={state.llvm_bump_commit}",
            f"# final_bad_llvm={state.new_llvm}",
            "triton_commit,llvm_commit_last_compatible",
        ]
        for triton_commit, llvm_last_compat in state.pairs:
            lines.append(f"{triton_commit},{llvm_last_compat}")

        output_path.write_text("\n".join(lines) + "\n")
        self.logger.info(f"Generated CSV with {len(state.pairs)} pairs: {output_path}")
        return output_path

    def print_fix_instructions(self) -> None:
        """Print human-readable fix guide for manual mode."""
        state = self._require_state()
        worktree = state.worktree_path or "<worktree>"
        llvm_short = (state.last_incompatible_llvm or "")[:12]
        print(
            f"\n{'=' * 60}\n"
            f"Manual fix required\n"
            f"{'=' * 60}\n"
            f"Worktree:        {worktree}\n"
            f"Triton commit:   {state.current_triton}\n"
            f"Incompatible:    {llvm_short} (first bad LLVM)\n"
            f"\nSteps:\n"
            f"  1. cd {worktree}\n"
            f"  2. Fix Triton source to compile with LLVM {llvm_short}\n"
            f"  3. git add -A && git commit -m 'compat fix: <description>'\n"
            f"  4. Resume: tritonparseoss compat-build --resume \\\n"
            f"       --state <state_file> --fix-commit <commit>\n"
            f"{'=' * 60}\n"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_compiler_path(self, name: str) -> str:
        """Resolve a compiler name to its full path via conda/which/bare fallback."""
        if os.path.isabs(name):
            return name
        conda_prefix = self.conda_prefix or ""
        if conda_prefix:
            conda_path = os.path.join(conda_prefix, "bin", name)
            if os.path.isfile(conda_path):
                return conda_path
        result = self.executor.run_command(["which", name])
        if result.success:
            return result.stdout.strip()
        return name

    @property
    def _using_clang(self) -> bool:
        return "clang" in self._cc

    def _require_state(self) -> CompatBuildState:
        if self.state is None:
            raise RuntimeError("State not initialized; call initialize() first")
        return self.state

    def _resolve_worktree_path(self) -> str:
        if self._worktree_path:
            return self._worktree_path
        root = self.worktree_root or os.path.join(self.triton_dir, ".compat_worktrees")
        short = self.llvm_bump_commit[:8]
        return os.path.join(root, short)

    def _resolve_conda_prefix(self) -> str:
        """Resolve conda env name to its full prefix path.

        Tries two strategies:
        1. If CONDA_PREFIX is already set in the environment and its basename
           matches ``self.conda_env``, use it directly.
        2. Otherwise, parse ``conda env list`` output to find the path.

        Returns:
            Absolute path to the conda environment (e.g.
            ``/home/user/miniconda3/envs/triton_bisect``).

        Raises:
            RuntimeError: If the environment cannot be found.
        """
        env_prefix = os.environ.get("CONDA_PREFIX", "")
        if env_prefix and os.path.basename(env_prefix) == self.conda_env:
            self.logger.info(f"Using active CONDA_PREFIX: {env_prefix}")
            return env_prefix

        result = self.executor.run_command(["conda", "env", "list"])
        if not result.success:
            raise RuntimeError(f"Failed to list conda environments: {result.stderr}")

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name = parts[0]
            path = parts[-1]
            if name == self.conda_env:
                self.logger.info(f"Resolved conda env '{self.conda_env}' → {path}")
                return path

        raise RuntimeError(
            f"Conda environment '{self.conda_env}' not found. "
            f"Available environments:\n{result.stdout}"
        )

    def _validate_conda_env(self) -> None:
        """Validate that the resolved conda environment has a working python.

        Must be called after ``_resolve_conda_prefix()``.

        Raises:
            RuntimeError: If the python binary is missing or broken.
        """
        prefix = self.conda_prefix
        if prefix is None:
            raise RuntimeError(
                "conda_prefix not resolved; call _resolve_conda_prefix() first"
            )

        python_bin = os.path.join(prefix, "bin", "python")
        if not os.path.isfile(python_bin):
            raise RuntimeError(
                f"Python binary not found at {python_bin}. "
                f"Is the conda environment '{self.conda_env}' correctly set up?"
            )

        result = self.executor.run_command([python_bin, "--version"])
        if not result.success:
            raise RuntimeError(
                f"Python binary at {python_bin} is not functional: {result.stderr}"
            )
        self.logger.info(f"Conda env validated: {python_bin} ({result.stdout.strip()})")

    def _setup_worktree(self, worktree_path: str) -> None:
        """Create or reuse a git worktree, always reset to llvm_bump~1."""
        wt = Path(worktree_path)
        parent_commit = f"{self.llvm_bump_commit}~1"
        branch = f"{_COMPAT_BRANCH_PREFIX}/{self.llvm_bump_commit[:8]}"

        if wt.exists():
            self.logger.info(f"Reusing existing worktree: {worktree_path}")
            self.executor.run_command(
                ["git", "reset", "--hard", parent_commit],
                cwd=worktree_path,
            )
            return

        wt.parent.mkdir(parents=True, exist_ok=True)

        result = self.executor.run_command(
            ["git", "worktree", "add", "-b", branch, worktree_path, parent_commit],
            cwd=self.triton_dir,
        )
        if result.success:
            self.logger.info(f"Created worktree at {worktree_path} on branch {branch}")
            return

        # Branch already exists (leftover) — reuse it, then reset
        result = self.executor.run_command(
            ["git", "worktree", "add", worktree_path, branch],
            cwd=self.triton_dir,
        )
        if not result.success:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")
        self.executor.run_command(
            ["git", "reset", "--hard", parent_commit],
            cwd=worktree_path,
        )
        self.logger.info(
            f"Reattached worktree to existing branch {branch}, reset to {parent_commit}"
        )

    def _get_llvm_hash_at_commit(self, commit: str, worktree_path: str) -> str:
        """Read llvm-hash.txt at a specific Triton commit."""
        result = self.executor.run_command(
            ["git", "show", f"{commit}:{_LLVM_HASH_FILE}"],
            cwd=worktree_path,
        )
        if not result.success:
            raise RuntimeError(
                f"Cannot read {_LLVM_HASH_FILE} at {commit}: {result.stderr}"
            )
        return result.stdout.strip()

    def _resolve_commit(self, rev: str, repo_path: str) -> str:
        """Resolve a git revision (e.g. ``abc123~1``) to a full SHA hash."""
        result = self.executor.run_command(
            ["git", "rev-parse", rev],
            cwd=repo_path,
        )
        if not result.success:
            raise RuntimeError(f"Cannot resolve {rev} in {repo_path}: {result.stderr}")
        return result.stdout.strip()

    def _probe_llvm(
        self, worktree: str, llvm_commit: str, conda_prefix: str
    ) -> CommandResult:
        """Run the compat probe against a single LLVM commit.

        Checks out the given LLVM commit, runs the probe, then restores
        the previous HEAD.

        Returns:
            The CommandResult from the probe (check .success for pass/fail,
            .output for the build log specific to this single LLVM commit).
        """
        llvm_dir = str(Path(worktree) / _LLVM_SUBDIR)

        prev_head = self.executor.run_command(
            ["git", "rev-parse", "HEAD"], cwd=llvm_dir
        )
        prev = prev_head.stdout.strip() if prev_head.success else None

        checkout = self.executor.run_command(
            ["git", "checkout", llvm_commit], cwd=llvm_dir
        )
        if not checkout.success:
            self.logger.warning(f"Cannot checkout LLVM {llvm_commit[:12]} for probe")
            return CommandResult(
                command=f"git checkout {llvm_commit}",
                exit_code=1,
                stdout="",
                stderr=checkout.stderr,
                duration_seconds=0.0,
            )

        self.logger.info(f"Probing LLVM {llvm_commit[:12]} for compatibility")
        probe = self._write_compat_probe_script(worktree)
        result = self.executor.run_command_streaming(
            ["bash", str(probe)],
            cwd=worktree,
            env={"TRITON_WORKTREE": worktree, "CONDA_PREFIX": conda_prefix},
        )

        if prev:
            self.executor.run_command(["git", "checkout", prev], cwd=llvm_dir)

        return result

    def _probe_llvm_compatible(
        self, worktree: str, llvm_commit: str, conda_prefix: str
    ) -> bool:
        """Test if a single LLVM commit is compatible with current Triton."""
        return self._probe_llvm(worktree, llvm_commit, conda_prefix).success

    def _patch_build_script_compilers(self, worktree_path: str) -> None:
        """Patch build-llvm-project.sh for non-clang compilers. No-op for clang."""
        if self._using_clang:
            return

        build_script = Path(worktree_path) / "scripts" / "build-llvm-project.sh"
        if not build_script.exists():
            return

        cc = self._resolve_compiler_path(self._cc)
        cxx = self._resolve_compiler_path(self._cxx)

        content = build_script.read_text()
        content = content.replace(
            "-DCMAKE_C_COMPILER=clang",
            f"-DCMAKE_C_COMPILER={cc}",
        )
        content = content.replace(
            "-DCMAKE_CXX_COMPILER=clang++",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        )
        content = content.replace("-DLLVM_ENABLE_LLD=ON", "")
        build_script.write_text(content)
        self.logger.info(f"Patched build-llvm-project.sh to use {cc}/{cxx}")

    def _ensure_llvm_repo(self, llvm_commit: str, worktree_path: str) -> None:
        """Ensure llvm-project exists in the worktree with full git history.

        ``scripts/build-llvm-project.sh`` often creates a shallow clone.
        The bisect step needs every commit between *old_llvm* and *new_llvm*,
        so we unconditionally unshallow the repo after the initial build.
        """
        llvm_dir = Path(worktree_path) / _LLVM_SUBDIR
        if not llvm_dir.exists():
            self.logger.info(
                f"Cloning/fetching LLVM repo into {llvm_dir} at {llvm_commit[:12]}"
            )
            self._patch_build_script_compilers(worktree_path)
            llvm_build_path = str(Path(worktree_path) / ".llvm-project" / "build")
            result = self.executor.run_command_streaming(
                [str(Path(worktree_path) / "scripts" / "build-llvm-project.sh")],
                cwd=worktree_path,
                env={
                    "LLVM_COMMIT_HASH": llvm_commit,
                    "LLVM_BUILD_PATH": llvm_build_path,
                },
            )
            if not result.success:
                raise RuntimeError(
                    f"Failed to set up LLVM repo: {result.stdout[-500:]}"
                )
        else:
            self.logger.info(f"LLVM repo already present at {llvm_dir}")

        self._unshallow_llvm_repo(llvm_dir)

    def _unshallow_llvm_repo(self, llvm_dir: Path) -> None:
        """Fetch full git history so that ``git bisect`` can walk all commits."""
        result = self.executor.run_command(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=str(llvm_dir),
        )
        if result.success and result.stdout.strip() == "true":
            self.logger.info("LLVM repo is shallow — fetching full history")
            fetch = self.executor.run_command(
                ["git", "fetch", "--unshallow"],
                cwd=str(llvm_dir),
            )
            if not fetch.success:
                raise RuntimeError(f"Failed to unshallow LLVM repo: {fetch.stderr}")
            self.logger.info("LLVM repo unshallowed successfully")
        else:
            self.logger.info("LLVM repo already has full history")

    def _write_compat_probe_script(self, worktree_path: str) -> Path:
        """
        Write a temporary compat probe shell script.

        The script is passed to `git bisect run`. For each LLVM commit that
        git bisect checks out in llvm-project, it runs the four-step probe:
          1. build LLVM (scripts/build-llvm-project.sh)
          2. build Triton (pip install -e .)
          3. import (python -c "import triton")
          4. compile smoke

        Exit code 0 = compatible (good), non-zero = incompatible (bad).
        """
        cc = self._resolve_compiler_path(self._cc)
        cxx = self._resolve_compiler_path(self._cxx)

        # When using a non-clang compiler, patch the build script and
        # override Triton's build env.  When using clang (default), the
        # build script's own config is correct — no patching needed.
        if self._using_clang:
            patch_lines = ""
            triton_build_env = ""
        else:
            patch_lines = (
                "# Patch build script compilers (mirrors _patch_build_script_compilers)\n"
                f'sed -i "s|-DCMAKE_C_COMPILER=clang|-DCMAKE_C_COMPILER={cc}|g" '
                '"$TRITON_WORKTREE/scripts/build-llvm-project.sh"\n'
                f'sed -i "s|-DCMAKE_CXX_COMPILER=clang++|-DCMAKE_CXX_COMPILER={cxx}|g" '
                '"$TRITON_WORKTREE/scripts/build-llvm-project.sh"\n'
                'sed -i "s|-DLLVM_ENABLE_LLD=ON||g" '
                '"$TRITON_WORKTREE/scripts/build-llvm-project.sh"\n'
            )
            triton_build_env = (
                "export TRITON_BUILD_WITH_CLANG_LLD=0\n"
                f'export CC="{cc}"\n'
                f'export CXX="{cxx}"\n'
            )

        script_content = (
            "#!/usr/bin/env bash\n"
            "set -e\n"
            'LLVM_COMMIT=$(git -C "$TRITON_WORKTREE/llvm-project" rev-parse HEAD)\n'
            "export LLVM_COMMIT_HASH=$LLVM_COMMIT\n"
            'LLVM_BUILD_PATH="$TRITON_WORKTREE/.llvm-project/build"\n'
            "# Clean LLVM build directory to avoid stale CMake cache\n"
            'rm -rf "$LLVM_BUILD_PATH"\n'
            "# Clean Triton build directory to avoid generator mismatch\n"
            'rm -rf "$TRITON_WORKTREE/build"\n'
            "unset CMAKE_ARGS\n"
            f"{patch_lines}"
            "# Step 1: build LLVM\n"
            'LLVM_BUILD_PATH="$LLVM_BUILD_PATH" '
            '"$TRITON_WORKTREE/scripts/build-llvm-project.sh" || exit 1\n'
            "# Step 2: build Triton\n"
            f"{triton_build_env}"
            "export TRITON_BUILD_WITH_CCACHE=0\n"
            'export LLVM_INCLUDE_DIRS="$LLVM_BUILD_PATH/include"\n'
            'export LLVM_LIBRARY_DIR="$LLVM_BUILD_PATH/lib"\n'
            'export LLVM_SYSPATH="$LLVM_BUILD_PATH"\n'
            'cd "$TRITON_WORKTREE"\n'
            '"$CONDA_PREFIX/bin/python" -m pip install -e . '
            "--no-build-isolation -v || exit 1\n"
            "# Step 3: import\n"
            '"$CONDA_PREFIX/bin/python" -c "import triton" || exit 1\n'
            "# Step 4: compile smoke kernel (must be a .py file for @triton.jit)\n"
            "_SMOKE=/tmp/_compat_smoke_kernel.py\n"
            "cat > $_SMOKE << 'KERNEL_EOF'\n"
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            "def _copy_kernel(x, y, n, BLOCK: tl.constexpr):\n"
            "    pid = tl.program_id(0)\n"
            "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
            "    tl.store(y + offs, tl.load(x + offs))\n"
            "\n"
            "import torch\n"
            "x = torch.zeros(128, device='cuda')\n"
            "_copy_kernel[(1,)](x, x, 128, BLOCK=128)\n"
            "KERNEL_EOF\n"
            '"$CONDA_PREFIX/bin/python" $_SMOKE || exit 1\n'
            "exit 0\n"
        )
        fd, path = tempfile.mkstemp(suffix=".sh", prefix="compat_probe_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(script_content)
        except OSError:
            os.close(fd)
            raise
        os.chmod(path, 0o755)
        return Path(path)

    def _parse_first_bad(self, bisect_output: str) -> str | None:
        """
        Extract the first-bad commit hash from git bisect run output.

        Git bisect prints a line like:
          <hash> is the first bad commit
        """
        for line in bisect_output.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[1:5] == ["is", "the", "first", "bad"]:
                return parts[0]
        return None

    def _get_prev_commit(self, repo_dir: str, commit: str) -> str:
        """Return the parent commit (commit~1) in the given repo."""
        result = self.executor.run_command(
            ["git", "rev-parse", f"{commit}~1"],
            cwd=repo_dir,
        )
        if not result.success:
            raise RuntimeError(
                f"Cannot resolve {commit}~1 in {repo_dir}: {result.stderr}"
            )
        return result.stdout.strip()

    def _save_build_error_log(self, output: str | None) -> Path | None:
        """Save raw build output to a log file for the AI agent to read.

        Logs are written to the state log_dir (outside the worktree) so
        that ``git add -A`` in the worktree won't commit them.

        Returns the path to the log file, or None if output is empty.
        """
        if not output:
            return None
        state = self._require_state()

        log_dir = Path(state.log_dir) / "build_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"build_error_{uuid4().hex[:12]}.log"
        log_path.write_text(output)
        self.logger.info(f"Build error log saved to {log_path}")
        return log_path

    def _try_ai_fix(
        self,
        incompatible_llvm: str,
        state: CompatBuildState,
        error_log: Path | None = None,
    ) -> str | None:
        """Create an AI fixer and attempt a fix with the given build error log.

        Args:
            incompatible_llvm: The incompatible LLVM commit hash.
            state: Current compat build state.
            error_log: Path to the build error log file for the AI to read.
                If None, uses state.last_build_error_log.

        Returns the fix commit hash, or None if the AI did not produce one.
        """
        if self.ai_fixer_factory is None:
            return None
        worktree = state.worktree_path
        if worktree is None:
            return None

        log_path = error_log
        if log_path is None and state.last_build_error_log:
            log_path = Path(state.last_build_error_log)

        try:
            ai_fixer = self.ai_fixer_factory(worktree)
            fix_commit = ai_fixer.attempt_fix(
                build_error_log=log_path,
                incompatible_llvm=incompatible_llvm,
                llvm_bump_commit=state.llvm_bump_commit,
            )
        except Exception as e:
            self.logger.warning(f"AI fixer raised an exception: {e}")
            return None

        if fix_commit and isinstance(fix_commit, str):
            self.logger.info(f"AI produced fix commit: {fix_commit[:12]}")
            return fix_commit
        return None

    def _raise_waiting_for_fix(
        self, state: CompatBuildState, incompatible_llvm: str
    ) -> str:
        """Save state and raise WaitingForFixError."""
        state.phase = CompatBuildPhase.WAITING_FOR_FIX
        state_path = state.save()
        self.print_fix_instructions()
        raise WaitingForFixError(
            state_path=state_path,
            incompatible_llvm=incompatible_llvm,
            build_error_log=state.last_build_error_log,
        )

    def _reset_to_pre_fix(
        self, pre_fix_head: str | None, state: CompatBuildState
    ) -> None:
        """Discard stacked broken commits from failed AI attempts."""
        worktree = state.worktree_path
        if not pre_fix_head or not worktree:
            return
        self.logger.info(f"Resetting worktree to pre-fix state {pre_fix_head[:12]}")
        self.executor.run_command(
            ["git", "reset", "--hard", pre_fix_head],
            cwd=worktree,
        )
        self._reattach_compat_branch(pre_fix_head)

    def _reattach_compat_branch(self, commit: str) -> None:
        """Re-attach HEAD to the compat branch after a detaching checkout."""
        state = self._require_state()
        worktree = state.worktree_path
        if worktree is None:
            return
        branch = f"{_COMPAT_BRANCH_PREFIX}/{state.llvm_bump_commit[:8]}"
        self.executor.run_command(
            ["git", "checkout", "-B", branch, commit],
            cwd=worktree,
        )

    def _verify_fix(
        self, fix_commit: str, llvm_commit: str
    ) -> tuple[bool, Path | None]:
        """Verify that the fix commit is compatible with the given LLVM commit.

        Checks out fix_commit in the worktree and runs the compat probe.

        Returns:
            Tuple of (success, error_log_path). error_log_path is the path to
            the raw build output if the probe failed, None if it succeeded.
        """
        state = self._require_state()
        worktree = state.worktree_path
        if worktree is None:
            return False, None

        checkout = self.executor.run_command(
            ["git", "checkout", fix_commit],
            cwd=worktree,
        )
        if not checkout.success:
            self.logger.error(f"Cannot checkout fix commit {fix_commit[:12]}")
            return False, None

        llvm_dir = str(Path(worktree) / _LLVM_SUBDIR)
        llvm_checkout = self.executor.run_command(
            ["git", "checkout", llvm_commit],
            cwd=llvm_dir,
        )
        if not llvm_checkout.success:
            self.logger.error(f"Cannot checkout LLVM {llvm_commit[:12]} for verify")
            return False, None

        probe = self._write_compat_probe_script(worktree)
        result: CommandResult = self.executor.run_command(
            ["bash", str(probe)],
            cwd=worktree,
            env={"TRITON_WORKTREE": worktree, "CONDA_PREFIX": self.conda_prefix or ""},
        )

        if result.success:
            return True, None

        error_log = self._save_build_error_log(result.output)
        return False, error_log
