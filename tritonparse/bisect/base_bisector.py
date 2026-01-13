# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Base bisector class for git bisect operations.

This module provides the abstract base class that defines the common structure
and behavior for all bisector implementations (Triton, LLVM, etc.).
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger


class BisectError(Exception):
    """Base exception for bisect related errors."""

    pass


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
    ) -> None:
        """
        Initialize the bisector.

        Args:
            triton_dir: Path to the Triton repository.
            test_script: Path to the test script that determines pass/fail.
            conda_env: Name of the conda environment to use for builds.
            logger: BisectLogger instance for logging.
            build_command: Custom build command. Defaults to subclass default.
        """
        self.triton_dir = Path(triton_dir).resolve()
        self.test_script = Path(test_script).resolve()
        self.conda_env = conda_env
        self.logger = logger
        self.build_command = build_command or self.default_build_command
        self.executor = ShellExecutor(logger)

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

        # Check for in-progress bisect
        bisect_start = target_dir / ".git" / "BISECT_START"
        if bisect_start.exists():
            raise BisectError(
                f"A bisect is already in progress in {target_dir}. "
                f"Run 'cd {target_dir} && git bisect reset' first."
            )

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
            "PER_COMMIT_LOG": "0",  # Disable per-commit logs in Python mode
        }
        # Only include BUILD_COMMAND if it's set (not used by LLVMBisector)
        if self.build_command:
            env["BUILD_COMMAND"] = self.build_command
        return env

    def _parse_bisect_result(self, output: str) -> str:
        """
        Parse the culprit commit from git bisect output.

        The output contains a line like:
        "<40-char-hash> is the first bad commit"

        Args:
            output: The stdout from git bisect run.

        Returns:
            The culprit commit hash.

        Raises:
            BisectError: If cannot parse the result.
        """
        # Try full 40-character hash first
        pattern_full = r"([a-f0-9]{40}) is the first bad commit"
        match = re.search(pattern_full, output)
        if match:
            return match.group(1)

        # Try shorter hash (7-12 characters)
        pattern_short = r"([a-f0-9]{7,12}) is the first bad commit"
        match = re.search(pattern_short, output)
        if match:
            return match.group(1)

        # If we can't find the pattern, raise an error with context
        raise BisectError(
            f"Cannot parse bisect result. Expected '<hash> is the first bad commit' "
            f"in output:\n{output[-500:]}"  # Last 500 chars for context
        )

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

    def _run_bisect(
        self,
        good_commit: str,
        bad_commit: str,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute the git bisect sequence.

        This is the core template method that defines the bisect workflow.

        Args:
            good_commit: Known good commit hash.
            bad_commit: Known bad commit hash.
            output_callback: Optional callback for real-time output.

        Returns:
            The culprit commit hash.

        Raises:
            BisectError: If bisect fails.
        """
        # Step 1: Log header
        self._log_header(good_commit, bad_commit)

        # Step 2: Prepare before bisect (hook for subclasses)
        self._prepare_before_bisect()

        # Step 3: Pre-bisect validation
        self._pre_bisect_check()

        # Step 4: Get the bisect script
        script_path = str(self._get_bisect_script())
        self.logger.info(f"Using bisect script: {script_path}")

        # Step 5: Set up environment variables
        env = self._get_base_env_vars()
        env.update(self._get_extra_env_vars())

        # Log bisect range for debugging
        self.logger.info("")
        self.logger.info("=" * 40)
        self.logger.info("BISECT RANGE")
        self.logger.info(f"  Good: {good_commit}")
        self.logger.info(f"  Bad:  {bad_commit}")
        self.logger.info("=" * 40)
        self.logger.info("")

        # Step 6: Execute git bisect sequence
        result = self.executor.run_git_bisect_sequence(
            repo_path=str(self.target_repo_dir),
            good_commit=good_commit,
            bad_commit=bad_commit,
            run_script=script_path,
            env=env,
            output_callback=output_callback,
        )

        if not result.success:
            raise BisectError(f"{self.bisect_name} failed: {result.stderr}")

        # Step 7: Parse the culprit commit
        culprit = self._parse_bisect_result(result.stdout)

        # Step 8: Log completion
        self._log_completion(culprit)

        return culprit
