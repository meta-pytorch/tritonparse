# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Core builder for compat_builder.

This module provides the CompatBuilder class which orchestrates the
compat-build workflow: finding LLVM compatibility boundaries and
generating commits.csv for the bisect workflow.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, TYPE_CHECKING

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger
from tritonparse.bisect.scripts import get_script_path
from tritonparse.compat_builder.state import (
    BumpBlock,
    CompatBuildPhase,
    CompatBuildState,
)

if TYPE_CHECKING:
    from tritonparse.compat_builder.fixer import FixStrategy


class CompatBuilderError(Exception):
    """compat-build related errors."""

    pass


class CompatBuilder:
    """
    Compatibility list builder.

    Uses COMPAT_MODE=1 bisect_llvm.sh to progressively find LLVM compatibility
    boundaries, then waits for/executes fixes, and finally generates commits.csv.

    Attributes:
        triton_dir: Path to Triton repository
        llvm_dir: Path to LLVM repository (inside triton_dir)
        llvm_bump_commit: Target LLVM bump commit hash
        output_csv: Path to output CSV file
        logger: BisectLogger instance for logging
        conda_env: Conda environment name
        executor: ShellExecutor for running commands
        state: Current CompatBuildState
        fix_strategy: Strategy for fixing incompatibilities
    """

    def __init__(
        self,
        triton_dir: str,
        llvm_bump_commit: str,
        output_csv: str,
        logger: BisectLogger,
        conda_env: str = "triton_bisect",
        fix_strategy: Optional[FixStrategy] = None,
    ):
        """
        Initialize CompatBuilder.

        Args:
            triton_dir: Path to Triton repository
            llvm_bump_commit: Target LLVM bump commit hash
            output_csv: Path to output CSV file
            logger: BisectLogger instance
            conda_env: Conda environment name
            fix_strategy: Optional fix strategy (defaults to ManualFixer)
        """
        self.triton_dir = Path(triton_dir).resolve()
        self.llvm_dir = self.triton_dir / "llvm-project"
        self.llvm_bump_commit = llvm_bump_commit
        self.output_csv = Path(output_csv).resolve()
        self.logger = logger
        self.conda_env = conda_env
        self.executor = ShellExecutor(logger)

        self.state: Optional[CompatBuildState] = None
        self.fix_strategy = fix_strategy

    def initialize(self) -> None:
        """
        Initialize the compat-build workflow.

        Reads the LLVM range from the bump commit and prepares initial state.

        Raises:
            CompatBuilderError: If initialization fails.
        """
        self.logger.info("=" * 60)
        self.logger.info("compat-build: Initializing")
        self.logger.info("=" * 60)

        # Read LLVM hashes before and after the bump
        old_llvm = self._get_llvm_hash_at_commit(f"{self.llvm_bump_commit}~1")
        new_llvm = self._get_llvm_hash_at_commit(self.llvm_bump_commit)

        self.logger.info(f"LLVM bump commit: {self.llvm_bump_commit}")
        self.logger.info(f"LLVM range: {old_llvm[:12]} -> {new_llvm[:12]}")

        # Validate LLVM order
        if not self._is_ancestor(old_llvm, new_llvm):
            raise CompatBuilderError(
                f"Invalid LLVM range: {old_llvm[:12]} is not before {new_llvm[:12]}"
            )

        # Checkout to llvm_bump~1
        self._checkout_triton(f"{self.llvm_bump_commit}~1")

        # Initialize state
        self.state = CompatBuildState(
            triton_dir=str(self.triton_dir),
            llvm_bump_commit=self.llvm_bump_commit,
            output_csv=str(self.output_csv),
            conda_env=self.conda_env,
            log_dir=str(self.logger.log_dir),
            session_name=self.logger.session_name,
            old_llvm=old_llvm,
            new_llvm=new_llvm,
            current_triton=f"{self.llvm_bump_commit}~1",
            current_llvm_good=old_llvm,
            phase=CompatBuildPhase.FINDING_INCOMPATIBLE,
        )
        self.state.save()

        self.logger.info("Initialization complete")

    def find_next_incompatible(
        self,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """
        Find the next incompatible LLVM commit using COMPAT_MODE=1 bisect.

        Args:
            output_callback: Optional callback for real-time output.

        Returns:
            first_incompatible_llvm: Hash of first incompatible LLVM commit.
            None: If the entire range is compatible.

        Raises:
            CompatBuilderError: If bisect fails unexpectedly.
        """
        self.logger.info("=" * 60)
        self.logger.info("Finding next incompatible LLVM commit")
        self.logger.info("=" * 60)

        # Ensure LLVM repo exists
        self._ensure_llvm_repo()

        # Check if already at target
        if self.state.current_llvm_good == self.state.new_llvm:
            self.logger.info("Already at target LLVM - all compatible!")
            return None

        self.logger.info(
            f"Bisecting LLVM: {self.state.current_llvm_good[:12]} -> "
            f"{self.state.new_llvm[:12]}"
        )

        script_path = get_script_path("bisect_llvm.sh")

        env = {
            "TRITON_DIR": str(self.triton_dir),
            "TEST_SCRIPT": "/dev/null",  # COMPAT_MODE doesn't need real test
            "CONDA_ENV": self.conda_env,
            "LOG_DIR": str(self.logger.log_dir),
            "COMPAT_MODE": "1",  # Key: compatibility check mode
        }

        # Run git bisect
        result = self.executor.run_git_bisect_sequence(
            repo_path=str(self.llvm_dir),
            good_commit=self.state.current_llvm_good,
            bad_commit=self.state.new_llvm,
            run_script=str(script_path),
            env=env,
            output_callback=output_callback,
        )

        if not result.success:
            # Check if it's "no commits to bisect"
            output_lower = result.output.lower() if result.output else ""
            if "no commits left to test" in output_lower:
                self.logger.info("No more commits to bisect - range fully covered")
                return None
            raise CompatBuilderError(f"LLVM compat bisect failed: {result.stderr}")

        # Parse result
        pattern = r"([a-f0-9]{40}) is the first bad commit"
        stdout = result.stdout or ""
        match = re.search(pattern, stdout)

        if match:
            first_incompatible = match.group(1)
            self.logger.info(f"Found first incompatible LLVM: {first_incompatible[:12]}")

            # Update state
            self.state.last_incompatible_llvm = first_incompatible
            self.state.phase = CompatBuildPhase.WAITING_FOR_FIX

            # Try to extract build error info
            output = result.output or ""
            self.state.last_build_error = self._extract_build_error(output)

            self.state.save()
            return first_incompatible

        # No bad commit found means all compatible
        self.logger.info("No incompatible LLVM found in range - all compatible!")
        return None

    def record_pair(self) -> None:
        """
        Record the current Triton commit with its last compatible LLVM commit.
        """
        if self.state.last_incompatible_llvm is None:
            # All compatible, record to the end
            self.state.add_pair(self.state.current_triton, self.state.new_llvm)
        else:
            # Record to the commit before the incompatible one
            last_compatible = self._get_prev_commit(
                self.llvm_dir, self.state.last_incompatible_llvm
            )
            self.state.add_pair(self.state.current_triton, last_compatible)

        self.logger.info(
            f"Recorded pair: ({self.state.pairs[-1][0][:12]}, "
            f"{self.state.pairs[-1][1][:12]})"
        )
        self.state.save()

    def apply_fix(self, fix_commit: str) -> None:
        """
        Apply a fix commit and update state.

        Args:
            fix_commit: Hash of the fix commit to apply.

        Raises:
            CompatBuilderError: If fix commit is invalid.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Applying fix: {fix_commit[:12]}")
        self.logger.info("=" * 60)

        # Validate fix commit exists
        result = self.executor.run_command(
            ["git", "rev-parse", "--verify", fix_commit],
            cwd=str(self.triton_dir),
        )
        if not result.success:
            raise CompatBuilderError(f"Fix commit not found: {fix_commit}")

        # Update state
        self.state.current_triton = fix_commit
        self.state.current_llvm_good = self.state.last_incompatible_llvm
        self.state.last_incompatible_llvm = None
        self.state.last_build_error = None
        self.state.phase = CompatBuildPhase.FINDING_INCOMPATIBLE
        self.state.save()

        # Checkout to fix commit
        self._checkout_triton(fix_commit)

        self.logger.info(
            f"State updated. Continuing from LLVM {self.state.current_llvm_good[:12]}"
        )

    def is_complete(self) -> bool:
        """Check if the workflow is complete."""
        if self.state.current_llvm_good == self.state.new_llvm:
            return True
        if self.state.phase == CompatBuildPhase.COMPLETED:
            return True
        return False

    def generate_csv(self, description: Optional[str] = None) -> Path:
        """
        Generate the final commits.csv file.

        This method supports appending to existing CSV files and auto-sorting
        bump blocks by LLVM timestamp.

        Args:
            description: Optional description for the bump block.

        Returns:
            Path to the generated CSV file.
        """
        self.logger.info("=" * 60)
        self.logger.info("Generating commits.csv")
        self.logger.info("=" * 60)

        # Create new bump block from current state
        new_block = self.state.create_bump_block(description)

        # Load existing blocks if CSV exists
        existing_blocks = self._parse_csv_to_blocks()

        # Check for duplicates
        for block in existing_blocks:
            if block.bump_commit == new_block.bump_commit:
                raise CompatBuilderError(
                    f"Duplicate bump_commit detected: {new_block.bump_commit}. "
                    "This LLVM bump has already been processed."
                )

        # Add new block
        all_blocks = existing_blocks + [new_block]

        # Sort by LLVM timestamp
        sorted_blocks = self._sort_bump_blocks(all_blocks)

        # Check for gaps
        self._check_gaps(sorted_blocks)

        # Write to CSV
        self._write_blocks_to_csv(sorted_blocks)

        self.logger.info(f"Generated: {self.output_csv}")
        self.logger.info(f"Total bump blocks: {len(sorted_blocks)}")
        self.logger.info(f"Total pairs in this block: {len(new_block.pairs)}")

        self.state.phase = CompatBuildPhase.COMPLETED
        self.state.save()

        return self.output_csv

    def print_fix_instructions(self) -> None:
        """Print instructions for manual fix."""
        print()
        print("=" * 60)
        print("FIX REQUIRED")
        print("=" * 60)
        print()
        print(f"Current Triton commit: {self.state.current_triton}")
        print(f"First incompatible LLVM: {self.state.last_incompatible_llvm[:12]}")
        print()
        print("Options to proceed:")
        print()
        print("1. Manual fix:")
        print(f"   - Checkout: git checkout {self.state.current_triton}")
        print("   - Apply fixes for the LLVM API change")
        print("   - Commit your changes")
        print("   - Resume: tritonparseoss compat-build resume \\")
        print("       --state <state_file> --fix-commit <your_fix_commit>")
        print()
        print("2. Extract from LLVM bump commit:")
        print(f"   - The LLVM bump commit ({self.llvm_bump_commit}) likely")
        print("     contains the necessary fixes. Try cherry-picking:")
        print(f"   - git cherry-pick --no-commit {self.llvm_bump_commit}")
        print("   - (review and keep only relevant changes)")
        print()
        if self.state.last_build_error:
            print("Build error hint:")
            print("-" * 40)
            error_preview = self.state.last_build_error[:500]
            print(error_preview)
            if len(self.state.last_build_error) > 500:
                print("... (truncated)")
        print()
        print("=" * 60)

    # ========== CSV Management Methods ==========

    def _parse_csv_to_blocks(self) -> List[BumpBlock]:
        """
        Parse existing CSV file into BumpBlock list.

        Returns:
            List of BumpBlock instances. Empty list if CSV doesn't exist.
        """
        if not self.output_csv.exists():
            return []

        with open(self.output_csv) as f:
            content = f.read()

        blocks = []
        current_block_lines = []
        current_metadata = {}
        in_block = False

        for line in content.split("\n"):
            line = line.strip()

            if line == "# === LLVM_BUMP_START ===":
                in_block = True
                current_block_lines = []
                current_metadata = {}
                continue

            if line == "# === LLVM_BUMP_END ===":
                if in_block and current_metadata.get("bump_commit"):
                    # Parse pairs from collected lines
                    pairs = []
                    for pair_line in current_block_lines:
                        if pair_line and not pair_line.startswith("#"):
                            parts = pair_line.split(",")
                            if len(parts) >= 2:
                                pairs.append((parts[0].strip(), parts[1].strip()))

                    block = BumpBlock(
                        bump_commit=current_metadata.get("bump_commit", ""),
                        range_start=current_metadata.get("range_start", ""),
                        range_end=current_metadata.get("range_end", ""),
                        generated_at=current_metadata.get("generated_at", ""),
                        description=current_metadata.get("description"),
                        pairs=pairs,
                    )
                    blocks.append(block)

                in_block = False
                continue

            if in_block:
                # Parse metadata comments
                if line.startswith("# bump_commit:"):
                    current_metadata["bump_commit"] = line.split(":", 1)[1].strip()
                elif line.startswith("# range_start:"):
                    current_metadata["range_start"] = line.split(":", 1)[1].strip()
                elif line.startswith("# range_end:"):
                    current_metadata["range_end"] = line.split(":", 1)[1].strip()
                elif line.startswith("# generated_at:"):
                    current_metadata["generated_at"] = line.split(":", 1)[1].strip()
                elif line.startswith("# description:"):
                    current_metadata["description"] = line.split(":", 1)[1].strip()
                elif not line.startswith("#"):
                    current_block_lines.append(line)

        return blocks

    def _get_commit_timestamp(self, commit: str) -> int:
        """
        Get the Unix timestamp of an LLVM commit.

        Args:
            commit: LLVM commit hash.

        Returns:
            Unix timestamp as integer.

        Raises:
            CompatBuilderError: If timestamp cannot be retrieved.
        """
        result = self.executor.run_command(
            ["git", "log", "-1", "--format=%ct", commit],
            cwd=str(self.llvm_dir),
        )
        if not result.success:
            raise CompatBuilderError(
                f"Failed to get timestamp for {commit}: {result.stderr}"
            )
        return int(result.stdout.strip())

    def _sort_bump_blocks(self, blocks: List[BumpBlock]) -> List[BumpBlock]:
        """
        Sort bump blocks by their range_start LLVM timestamp.

        Args:
            blocks: List of BumpBlock instances.

        Returns:
            Sorted list of BumpBlock instances.
        """
        if not blocks:
            return blocks

        # Get timestamps for all blocks
        for block in blocks:
            if block.range_start_timestamp is None:
                block.range_start_timestamp = self._get_commit_timestamp(
                    block.range_start
                )

        return sorted(blocks, key=lambda b: b.range_start_timestamp or 0)

    def _check_gaps(self, sorted_blocks: List[BumpBlock]) -> None:
        """
        Check for gaps between sorted blocks and log warnings.

        Args:
            sorted_blocks: List of BumpBlock instances sorted by timestamp.
        """
        for i in range(len(sorted_blocks) - 1):
            current = sorted_blocks[i]
            next_block = sorted_blocks[i + 1]

            if current.range_end != next_block.range_start:
                self.logger.warning(
                    f"Gap detected between bumps:\n"
                    f"  Bump {current.bump_commit[:12]}: "
                    f"ends at {current.range_end[:12]}\n"
                    f"  Bump {next_block.bump_commit[:12]}: "
                    f"starts at {next_block.range_start[:12]}\n"
                    f"  Missing LLVM range coverage!"
                )

    def _write_blocks_to_csv(self, blocks: List[BumpBlock]) -> None:
        """
        Write sorted bump blocks to CSV file.

        Args:
            blocks: List of BumpBlock instances to write.
        """
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        lines = ["triton_commit,llvm_commit_last_compatible", ""]

        for block in blocks:
            lines.append(block.format_csv_block())
            lines.append("")

        with open(self.output_csv, "w") as f:
            f.write("\n".join(lines))

    # ========== Private Helper Methods ==========

    def _get_llvm_hash_at_commit(self, commit: str) -> str:
        """Get LLVM hash from cmake/llvm-hash.txt at specified Triton commit."""
        result = self.executor.run_command(
            ["git", "show", f"{commit}:cmake/llvm-hash.txt"],
            cwd=str(self.triton_dir),
        )
        if not result.success:
            raise CompatBuilderError(
                f"Failed to get LLVM hash at {commit}: {result.stderr}"
            )
        return result.stdout.strip()

    def _checkout_triton(self, commit: str) -> None:
        """Checkout Triton to specified commit."""
        result = self.executor.run_command(
            ["git", "checkout", commit],
            cwd=str(self.triton_dir),
        )
        if not result.success:
            raise CompatBuilderError(f"Failed to checkout {commit}: {result.stderr}")

    def _ensure_llvm_repo(self) -> None:
        """Ensure LLVM repo exists, initialize if needed."""
        if not self.llvm_dir.exists():
            self.logger.info("LLVM repo not found, initializing...")
            result = self.executor.run_command(
                ["make", "dev-install-llvm"],
                cwd=str(self.triton_dir),
            )
            if not result.success:
                raise CompatBuilderError(f"Failed to init LLVM repo: {result.stderr}")

    def _get_prev_commit(self, repo_dir: Path, commit: str) -> str:
        """Get the parent commit of specified commit."""
        result = self.executor.run_command(
            ["git", "rev-parse", f"{commit}~1"],
            cwd=str(repo_dir),
        )
        if not result.success:
            raise CompatBuilderError(
                f"Failed to get parent of {commit}: {result.stderr}"
            )
        return result.stdout.strip()

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """
        Check if ancestor commit is an ancestor of descendant commit.

        Args:
            ancestor: Potential ancestor commit hash.
            descendant: Potential descendant commit hash.

        Returns:
            True if ancestor is an ancestor of descendant.
        """
        result = self.executor.run_command(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=str(self.llvm_dir),
        )
        return result.success

    def _extract_build_error(self, output: str) -> Optional[str]:
        """Extract build error information from output."""
        lines = output.split("\n")
        error_lines = []
        in_error = False

        for line in lines:
            if "error:" in line.lower() or "fatal:" in line.lower():
                in_error = True
            if in_error:
                error_lines.append(line)
                if len(error_lines) > 20:
                    break

        return "\n".join(error_lines) if error_lines else None

    @classmethod
    def from_state(cls, state: CompatBuildState, logger: BisectLogger) -> CompatBuilder:
        """
        Create CompatBuilder from existing state.

        Args:
            state: Existing CompatBuildState to resume from.
            logger: BisectLogger instance.

        Returns:
            CompatBuilder instance with restored state.
        """
        builder = cls(
            triton_dir=state.triton_dir,
            llvm_bump_commit=state.llvm_bump_commit,
            output_csv=state.output_csv,
            logger=logger,
            conda_env=state.conda_env,
        )
        builder.state = state
        return builder
