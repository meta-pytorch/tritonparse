# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Base bisector class for git bisect operations.

This module provides the abstract base class that defines the common structure
and behavior for all bisector implementations (Triton, LLVM, etc.).
"""

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from tritonparse._json_compat import dumps
from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger
from tritonparse.bisect.result import BisectResult


class BisectError(Exception):
    """Base exception for bisect related errors."""

    def __init__(
        self, message: str = "", *, result: Optional[BisectResult] = None
    ) -> None:
        super().__init__(message)
        self.result = result


class BaseBisector(ABC):
    """
    Abstract base class for bisect executors.

    This class implements the Template Method pattern, defining the common
    bisect workflow structure while allowing subclasses to customize specific
    steps.

    The bisect workflow consists of:
    1. Log header information
    2. Prepare before bisect (subclass hook)
    3. Pre-bisect validation checks
    4. Get the bisect script
    5. Set up environment variables
    6. Execute git bisect sequence
    7. Parse and return the culprit commit

    Subclasses must implement:
    - bisect_name: Name of the bisect (for logging)
    - default_build_command: Default build command
    - target_repo_dir: Directory where git bisect runs
    - _get_bisect_script(): Return the bisect script path
    - _get_extra_env_vars(): Return additional environment variables

    Subclasses may override:
    - _prepare_before_bisect(): Hook for pre-bisect preparation
    - _log_header(): Custom header logging
    """

    def __init__(
        self,
        triton_dir: str,
        test_script: str,
        conda_env: str,
        logger: BisectLogger,
        build_command: Optional[str] = None,
        per_commit_log: bool = False,
        build_fail_action: str = "skip",
    ) -> None:
        """
        Initialize the bisector.

        Args:
            triton_dir: Path to the Triton repository.
            test_script: Path to the test script that determines pass/fail.
            conda_env: Name of the conda environment to use for builds.
            logger: BisectLogger instance for logging.
            build_command: Custom build command. Defaults to subclass default.
            per_commit_log: If True, the bisect script writes a separate log
                file per tested commit.
            build_fail_action: What to tell git bisect when the build fails on
                an intermediate commit. One of:
                - "skip" (default, recommended): exit 125 so git bisect skips
                  this commit and continues. This handles transient compile
                  breaks in intermediate Triton/Torch commits without aborting
                  the whole bisect.
                - "abort": exit 128 so git bisect aborts immediately. Use this
                  only when build infra itself is broken and you want to stop
                  and investigate.
        """
        self.triton_dir = Path(triton_dir).resolve()
        self.test_script = Path(test_script).resolve()
        self.conda_env = conda_env
        self.logger = logger
        self.build_command = build_command or self.default_build_command
        self.executor = ShellExecutor(logger)
        self.per_commit_log = per_commit_log
        self.build_fail_action = self._validate_build_fail_action(build_fail_action)
        self.result: Optional[BisectResult] = None

    @staticmethod
    def _validate_build_fail_action(value: str) -> str:
        """
        Validate a build_fail_action string.

        Args:
            value: User-provided action name.

        Returns:
            Lowercased, validated action name.

        Raises:
            ValueError: If value is not "skip" or "abort".
        """
        if not isinstance(value, str):
            raise ValueError(
                f"build_fail_action must be a string, got {type(value).__name__}"
            )
        normalized = value.strip().lower()
        if normalized not in ("skip", "abort"):
            raise ValueError(
                f"build_fail_action must be 'skip' or 'abort', got: {value!r}"
            )
        return normalized

    @staticmethod
    def _build_fail_exit_code(action: str) -> str:
        """
        Map a build_fail_action name to the git-bisect exit code (as a string,
        ready for use as an environment variable value).

        Args:
            action: Validated action name ("skip" or "abort").

        Returns:
            "125" for "skip", "128" for "abort".
        """
        return "125" if action == "skip" else "128"

    @property
    @abstractmethod
    def bisect_name(self) -> str:
        """Name of the bisect operation (e.g., 'Triton Bisect', 'LLVM Bisect')."""
        pass

    @property
    @abstractmethod
    def default_build_command(self) -> str:
        """Default build command for this bisector."""
        pass

    @property
    @abstractmethod
    def target_repo_dir(self) -> Path:
        """Directory where git bisect will be executed."""
        pass

    @abstractmethod
    def _get_bisect_script(self) -> Union[str, Path]:
        """
        Get the path to the bisect script.

        Returns:
            Path to the bisect script.
        """
        pass

    @abstractmethod
    def _get_extra_env_vars(self) -> Dict[str, str]:
        """
        Get additional environment variables specific to this bisector.

        Returns:
            Dictionary of additional environment variables.
        """
        pass

    def _prepare_before_bisect(self) -> None:  # noqa: B027
        """
        Hook for subclasses to perform preparation before bisect.

        This method is called after logging the header but before
        pre-bisect checks. Override in subclasses if needed.

        For example, LLVMBisector uses this to checkout Triton commit
        and ensure LLVM repo exists.
        """
        pass

    def _log_header(
        self,
        good_commit: str,
        bad_commit: str,
    ) -> None:
        """
        Log the bisect header information.

        Args:
            good_commit: Known good commit hash.
            bad_commit: Known bad commit hash.
        """
        self.logger.info("=" * 60)
        self.logger.info(self.bisect_name)
        self.logger.info("=" * 60)
        self.logger.info(f"Target directory: {self.target_repo_dir}")
        self.logger.info(f"Test script: {self.test_script}")
        self.logger.info(f"Good commit: {good_commit}")
        self.logger.info(f"Bad commit: {bad_commit}")
        self.logger.info(f"Conda environment: {self.conda_env}")
        self.logger.info(f"Build command: {self.build_command}")
        self.logger.info(
            f"On build failure: {self.build_fail_action} "
            f"(exit {self._build_fail_exit_code(self.build_fail_action)})"
        )

    def _pre_bisect_check(self) -> None:
        """
        Perform pre-bisect validation checks.

        Raises:
            BisectError: If any validation check fails.
        """
        target_dir = self.target_repo_dir

        # Check target directory exists
        if not target_dir.exists():
            raise BisectError(f"Target directory not found: {target_dir}")

        # Check it's a git repository
        git_dir = target_dir / ".git"
        if not git_dir.exists():
            raise BisectError(f"Not a git repository: {target_dir}")

        # The executor checks Git's BISECT_START path before starting; unlike
        # ".git/BISECT_START", that also works for linked worktrees.

        # Check test script exists
        if not self.test_script.exists():
            raise BisectError(f"Test script not found: {self.test_script}")

        # Check working directory status (warning only)
        result = self.executor.run_command(
            ["git", "status", "--porcelain"],
            cwd=str(target_dir),
        )
        if result.stdout.strip():
            self.logger.warning(
                f"Working directory {target_dir} has uncommitted changes. "
                "This may cause issues during bisect."
            )

        self.logger.info("Pre-bisect checks passed")

    def _get_base_env_vars(self) -> Dict[str, str]:
        """
        Get the base environment variables common to all bisectors.

        Note: BUILD_COMMAND is only included if set. For LLVMBisector,
        build_command is None because bisect_llvm.sh uses a fixed
        two-phase build process.

        Returns:
            Dictionary of base environment variables.
        """
        env = {
            "TRITON_DIR": str(self.triton_dir),
            "TEST_SCRIPT": str(self.test_script),
            "CONDA_ENV": self.conda_env,
            "LOG_DIR": str(self.logger.log_dir),
            "PER_COMMIT_LOG": "1" if self.per_commit_log else "0",
            "BUILD_FAIL_EXIT_CODE": self._build_fail_exit_code(self.build_fail_action),
        }
        # Only include BUILD_COMMAND if it's set (not used by LLVMBisector)
        if self.build_command:
            env["BUILD_COMMAND"] = self.build_command
        return env

    @staticmethod
    def _require_culprit(result: BisectResult) -> str:
        if result.status == "found" and result.culprit:
            return result.culprit
        raise BisectError(result.message, result=result)

    def _parse_bisect_result(self, output: str, exit_code: int = 0) -> str:
        """Return only a unique result; incomplete results raise with evidence."""
        return self._require_culprit(BisectResult.from_output(output, exit_code))

    @staticmethod
    def _parse_skip_candidates(output: str) -> list[str]:
        return BisectResult.parse_skip_candidates(output)

    def _log_completion(self, culprit: str) -> None:
        """
        Log bisect completion message.

        Args:
            culprit: The culprit commit hash.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"{self.bisect_name} completed!")
        self.logger.info(f"Culprit commit: {culprit}")
        self.logger.info("=" * 60)

    def _save_result(self) -> None:
        assert self.result is not None
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix=f"{self.logger.session_name}_",
                suffix="_bisect_result.json.tmp",
                dir=self.logger.log_dir,
                delete=False,
            ) as output:
                temporary_path = Path(output.name).resolve()
                result_path = temporary_path.with_suffix("")
                payload = self.result.to_dict()
                payload["result_file"] = str(result_path)
                output.write(dumps(payload, indent=True))
            temporary_path.replace(result_path)
            self.result.result_file = str(result_path)
        except OSError as error:
            self.result.result_file = None
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    self.result.message += (
                        f"\nCannot remove incomplete result {temporary_path}: "
                        f"{cleanup_error}"
                    )
            if self.result.status == "found":
                self.result.candidates = (
                    [self.result.culprit] if self.result.culprit else []
                )
                self.result.culprit = None
                self.result.status = "error"
            self.result.message += f"\nCannot save bisect result: {error}"
            raise BisectError(self.result.message, result=self.result) from error
        self.logger.info(f"Bisect result saved to: {self.result.result_file}")

    def _run_bisect(
        self,
        good_commit: str,
        bad_commit: str,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Run bisect and persist its outcome, returning only a unique culprit."""
        self.result = None
        sequence_started = False
        cause = None
        try:
            self._log_header(good_commit, bad_commit)
            self._prepare_before_bisect()
            self._pre_bisect_check()
            script_path = str(self._get_bisect_script())
            self.logger.info(f"Using bisect script: {script_path}")
            env = self._get_base_env_vars()
            env.update(self._get_extra_env_vars())

            sequence_started = True
            command = self.executor.run_git_bisect_sequence(
                repo_path=str(self.target_repo_dir),
                good_commit=good_commit,
                bad_commit=bad_commit,
                run_script=script_path,
                env=env,
                output_callback=output_callback,
            )
            # Git can return nonzero for a skipped candidate set. Classify the
            # complete output before deciding whether this attempt succeeded.
            self.result = BisectResult.from_output(command.output, command.exit_code)
        except KeyboardInterrupt as error:
            cause = error
            self.result = BisectResult(
                status="aborted", exit_code=130, message="Git bisect interrupted."
            )
        except Exception as error:
            cause = error
            self.result = (
                error.result
                if isinstance(error, BisectError) and error.result is not None
                else BisectResult(status="error", message=str(error))
            )

        self.result.repository = str(self.target_repo_dir)
        self.result.good_commit = good_commit
        self.result.bad_commit = bad_commit
        self.result.command_log = str(self.logger.command_log_path.resolve())
        if sequence_started:
            if self.executor.bisect_log_path is not None:
                self.result.git_bisect_log = str(self.executor.bisect_log_path)
            if self.executor.bisect_cleanup_errors:
                if self.result.status == "found":
                    self.result.candidates = (
                        [self.result.culprit] if self.result.culprit else []
                    )
                    self.result.culprit = None
                    self.result.status = "error"
                self.result.message += "\n" + "\n".join(
                    self.executor.bisect_cleanup_errors
                )

        self._save_result()
        if self.result.status != "found":
            raise BisectError(self.result.message, result=self.result) from cause
        culprit = self._require_culprit(self.result)
        self._log_completion(culprit)
        return culprit
