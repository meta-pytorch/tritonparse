# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Triton bisect executor for finding regression-causing commits.

This module implements Phase 1 of the bisect workflow: bisecting Triton
commits to find the first bad commit that causes a test to fail.
"""

import re
from pathlib import Path
from typing import Callable, Optional

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger
from tritonparse.bisect.scripts import get_bisect_triton_script


class TritonBisectError(Exception):
    """Exception raised for Triton bisect related errors."""

    pass


class TritonBisector:
    """
    Triton bisect executor.

    This class handles the complete Triton bisect workflow:
    1. Pre-bisect validation checks
    2. Setting up environment variables
    3. Running git bisect with the embedded script
    4. Parsing results to extract the culprit commit

    Example:
        >>> logger = BisectLogger("./logs")
        >>> bisector = TritonBisector(
        ...     triton_dir="/path/to/triton",
        ...     test_script="/path/to/test.py",
        ...     conda_env="my_env",
        ...     logger=logger,
        ... )
        >>> culprit = bisector.run(good_commit="v2.0.0", bad_commit="HEAD")
        >>> print(f"Culprit commit: {culprit}")
    """

    DEFAULT_BUILD_COMMAND = "pip install -e ."

    def __init__(
        self,
        triton_dir: str,
        test_script: str,
        conda_env: str,
        logger: BisectLogger,
        build_command: Optional[str] = None,
    ) -> None:
        """
        Initialize the Triton bisector.

        Args:
            triton_dir: Path to the Triton repository.
            test_script: Path to the test script that determines pass/fail.
            conda_env: Name of the conda environment to use for builds.
            logger: BisectLogger instance for logging.
            build_command: Custom build command. Defaults to "pip install -e .".
        """
        self.triton_dir = Path(triton_dir).resolve()
        self.test_script = Path(test_script).resolve()
        self.conda_env = conda_env
        self.logger = logger
        self.build_command = build_command or self.DEFAULT_BUILD_COMMAND
        self.executor = ShellExecutor(logger)

    def run(
        self,
        good_commit: str,
        bad_commit: str,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute Triton bisect to find the culprit commit.

        Args:
            good_commit: Known good commit hash or tag (test passes).
            bad_commit: Known bad commit hash or tag (test fails).
            output_callback: Optional callback called for each output line.
                            Used by TUI to display real-time output.

        Returns:
            The culprit commit hash (first bad commit).

        Raises:
            TritonBisectError: If bisect fails or cannot parse the result.
        """
        self.logger.info("=" * 60)
        self.logger.info("Phase 1: Triton Bisect")
        self.logger.info("=" * 60)
        self.logger.info(f"Triton directory: {self.triton_dir}")
        self.logger.info(f"Test script: {self.test_script}")
        self.logger.info(f"Good commit: {good_commit}")
        self.logger.info(f"Bad commit: {bad_commit}")
        self.logger.info(f"Conda environment: {self.conda_env}")
        self.logger.info(f"Build command: {self.build_command}")

        # Pre-bisect validation
        self._pre_bisect_check()

        # Get the embedded bisect script
        script_path = self._prepare_script()
        self.logger.info(f"Using bisect script: {script_path}")

        # Set up environment variables for the bisect script
        env = {
            "TRITON_DIR": str(self.triton_dir),
            "TEST_SCRIPT": str(self.test_script),
            "CONDA_ENV": self.conda_env,
            "BUILD_COMMAND": self.build_command,
            "LOG_DIR": str(self.logger.log_dir),
            "PER_COMMIT_LOG": "0",  # Disable per-commit logs in Python mode
        }

        # Execute git bisect sequence
        result = self.executor.run_git_bisect_sequence(
            repo_path=str(self.triton_dir),
            good_commit=good_commit,
            bad_commit=bad_commit,
            run_script=script_path,
            env=env,
            output_callback=output_callback,
        )

        if not result.success:
            raise TritonBisectError(f"Triton bisect failed: {result.stderr}")

        # Parse the culprit commit from output
        culprit = self._parse_bisect_result(result.stdout)

        self.logger.info("=" * 60)
        self.logger.info("Triton bisect completed!")
        self.logger.info(f"Culprit commit: {culprit}")
        self.logger.info("=" * 60)

        return culprit

    def _pre_bisect_check(self) -> None:
        """
        Perform pre-bisect validation checks.

        Raises:
            TritonBisectError: If any validation check fails.
        """
        # Check Triton directory exists
        if not self.triton_dir.exists():
            raise TritonBisectError(f"Triton directory not found: {self.triton_dir}")

        # Check it's a git repository
        git_dir = self.triton_dir / ".git"
        if not git_dir.exists():
            raise TritonBisectError(f"Not a git repository: {self.triton_dir}")

        # Check for in-progress bisect
        bisect_start = self.triton_dir / ".git" / "BISECT_START"
        if bisect_start.exists():
            raise TritonBisectError(
                "A bisect is already in progress in the Triton repository. "
                f"Run 'cd {self.triton_dir} && git bisect reset' first."
            )

        # Check test script exists
        if not self.test_script.exists():
            raise TritonBisectError(f"Test script not found: {self.test_script}")

        # Check working directory status (warning only)
        result = self.executor.run_command(
            ["git", "status", "--porcelain"],
            cwd=str(self.triton_dir),
        )
        if result.stdout.strip():
            self.logger.warning(
                "Triton working directory has uncommitted changes. "
                "This may cause issues during bisect."
            )

        self.logger.info("Pre-bisect checks passed")

    def _prepare_script(self) -> str:
        """
        Get the path to the embedded bisect script.

        Returns:
            Absolute path to bisect_triton.sh script.
        """
        return get_bisect_triton_script()

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
            TritonBisectError: If cannot parse the result.
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
        raise TritonBisectError(
            f"Cannot parse bisect result. Expected '<hash> is the first bad commit' "
            f"in output:\n{output[-500:]}"  # Last 500 chars for context
        )
