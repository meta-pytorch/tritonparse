# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
LLVM bisect executor for finding regression-causing commits in LLVM.

This module implements Phase 4 of the bisect workflow: bisecting LLVM
commits within a Triton-compatible range to find the first bad LLVM commit.
"""

import re
from pathlib import Path
from typing import Optional

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger
from tritonparse.bisect.scripts import get_bisect_llvm_script


class LLVMBisectError(Exception):
    """Exception raised for LLVM bisect related errors."""

    pass


class LLVMBisector:
    """
    LLVM bisect executor.

    This class handles the complete LLVM bisect workflow:
    1. Get/verify Triton commit (use specified or current HEAD)
    2. Checkout Triton to the specified commit
    3. Ensure LLVM repository exists (initialize if needed)
    4. Pre-bisect validation checks
    5. Run git bisect on the LLVM repository
    6. Parse results to extract the culprit LLVM commit

    The LLVM repository is expected to be at {triton_dir}/llvm-project.

    Example:
        >>> logger = BisectLogger("./logs")
        >>> bisector = LLVMBisector(
        ...     triton_dir="/path/to/triton",
        ...     test_script="/path/to/test.py",
        ...     conda_env="my_env",
        ...     logger=logger,
        ... )
        >>> culprit = bisector.run(
        ...     triton_commit=None,  # Use current HEAD
        ...     good_llvm="abc123",
        ...     bad_llvm="def456",
        ... )
        >>> print(f"Culprit LLVM commit: {culprit}")
    """

    DEFAULT_BUILD_COMMAND = "make dev-install-llvm"

    def __init__(
        self,
        triton_dir: str,
        test_script: str,
        conda_env: str,
        logger: BisectLogger,
        build_command: Optional[str] = None,
    ) -> None:
        """
        Initialize the LLVM bisector.

        Args:
            triton_dir: Path to the Triton repository.
            test_script: Path to the test script that determines pass/fail.
            conda_env: Name of the conda environment to use for builds.
            logger: BisectLogger instance for logging.
            build_command: Custom build command. Defaults to "make dev-install-llvm".
                          Note: The script will set LLVM_COMMIT_HASH automatically.
        """
        self.triton_dir = Path(triton_dir).resolve()
        self.llvm_dir = self.triton_dir / "llvm-project"
        self.test_script = Path(test_script).resolve()
        self.conda_env = conda_env
        self.logger = logger
        self.build_command = build_command or self.DEFAULT_BUILD_COMMAND
        self.executor = ShellExecutor(logger)

    def run(
        self,
        triton_commit: Optional[str],
        good_llvm: str,
        bad_llvm: str,
    ) -> str:
        """
        Execute LLVM bisect to find the culprit commit.

        Args:
            triton_commit: Fixed Triton commit to use. If None, uses current HEAD.
            good_llvm: Known good LLVM commit hash (test passes).
            bad_llvm: Known bad LLVM commit hash (test fails).

        Returns:
            The culprit LLVM commit hash (first bad commit).

        Raises:
            LLVMBisectError: If bisect fails or cannot parse the result.
        """
        self.logger.info("=" * 60)
        self.logger.info("Phase 4: LLVM Bisect")
        self.logger.info("=" * 60)

        # Step 1: Get Triton commit
        actual_triton_commit = self._get_triton_commit(triton_commit)
        self.logger.info(f"Triton directory: {self.triton_dir}")
        self.logger.info(f"Triton commit: {actual_triton_commit}")
        self.logger.info(f"Test script: {self.test_script}")
        self.logger.info(f"Good LLVM commit: {good_llvm}")
        self.logger.info(f"Bad LLVM commit: {bad_llvm}")
        self.logger.info(f"Conda environment: {self.conda_env}")
        self.logger.info(f"Build command: {self.build_command}")

        # Step 2: Checkout Triton commit
        self._checkout_triton(actual_triton_commit)

        # Step 3: Ensure LLVM repo exists
        self._ensure_llvm_repo()

        # Step 4: Pre-bisect validation
        self._pre_bisect_check()

        # Step 5: Get the embedded bisect script
        script_path = self._prepare_script()
        self.logger.info(f"Using bisect script: {script_path}")

        # Step 6: Set up environment variables for the bisect script
        env = {
            "TRITON_DIR": str(self.triton_dir),
            "TEST_SCRIPT": str(self.test_script),
            "CONDA_ENV": self.conda_env,
            "BUILD_COMMAND": self.build_command,
            "LOG_DIR": str(self.logger.log_dir),
            "COMPAT_MODE": "0",  # Normal mode: find regression
        }

        # Step 7: Execute git bisect sequence on LLVM repo
        result = self.executor.run_git_bisect_sequence(
            repo_path=str(self.llvm_dir),
            good_commit=good_llvm,
            bad_commit=bad_llvm,
            run_script=script_path,
            env=env,
        )

        if not result.success:
            raise LLVMBisectError(f"LLVM bisect failed: {result.stderr}")

        # Step 8: Parse the culprit commit from output
        culprit = self._parse_bisect_result(result.stdout)

        self.logger.info("=" * 60)
        self.logger.info("LLVM bisect completed!")
        self.logger.info(f"Culprit LLVM commit: {culprit}")
        self.logger.info("=" * 60)

        return culprit

    def _get_triton_commit(self, specified_commit: Optional[str]) -> str:
        """
        Get the Triton commit to use.

        If a commit is specified, return it. Otherwise, get the current HEAD.

        Args:
            specified_commit: User-specified commit, or None.

        Returns:
            The Triton commit hash to use.

        Raises:
            LLVMBisectError: If cannot get the current HEAD.
        """
        if specified_commit:
            self.logger.info(f"Using specified Triton commit: {specified_commit}")
            return specified_commit

        # Get current HEAD from triton_dir
        self.logger.info("No Triton commit specified, using current HEAD")
        result = self.executor.run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=str(self.triton_dir),
        )

        if not result.success:
            raise LLVMBisectError(
                f"Failed to get Triton HEAD: {result.stderr}"
            )

        commit = result.stdout.strip()
        self.logger.info(f"Current Triton HEAD: {commit}")
        return commit

    def _checkout_triton(self, commit: str) -> None:
        """
        Checkout the specified Triton commit.

        Args:
            commit: The commit hash to checkout.

        Raises:
            LLVMBisectError: If checkout fails.
        """
        self.logger.info(f"Checking out Triton commit: {commit}")
        result = self.executor.run_command(
            ["git", "checkout", commit],
            cwd=str(self.triton_dir),
        )

        if not result.success:
            raise LLVMBisectError(
                f"Failed to checkout Triton commit {commit}: {result.stderr}"
            )

    def _ensure_llvm_repo(self) -> None:
        """
        Ensure the LLVM repository exists and is valid.

        If the LLVM repo doesn't exist, attempt to initialize it by running
        'make dev-install-llvm' in the Triton directory.

        Raises:
            LLVMBisectError: If LLVM repo cannot be initialized or is invalid.
        """
        if not self.llvm_dir.exists():
            self.logger.info(f"LLVM repo not found at {self.llvm_dir}")
            self.logger.info("Initializing LLVM repo with 'make dev-install-llvm'...")

            result = self.executor.run_command_streaming(
                ["make", "dev-install-llvm"],
                cwd=str(self.triton_dir),
            )

            if not result.success:
                raise LLVMBisectError(
                    f"Failed to initialize LLVM repo: {result.stderr}"
                )

            self.logger.info("LLVM repo initialized successfully")

        # Verify it's a git repository
        git_dir = self.llvm_dir / ".git"
        if not git_dir.exists():
            raise LLVMBisectError(
                f"LLVM directory exists but is not a git repository: {self.llvm_dir}"
            )

        self.logger.info(f"LLVM repo verified at: {self.llvm_dir}")

    def _pre_bisect_check(self) -> None:
        """
        Perform pre-bisect validation checks.

        Raises:
            LLVMBisectError: If any validation check fails.
        """
        # Check Triton directory exists
        if not self.triton_dir.exists():
            raise LLVMBisectError(f"Triton directory not found: {self.triton_dir}")

        # Check Triton is a git repository
        triton_git_dir = self.triton_dir / ".git"
        if not triton_git_dir.exists():
            raise LLVMBisectError(f"Triton is not a git repository: {self.triton_dir}")

        # Check for in-progress bisect in LLVM repo
        bisect_start = self.llvm_dir / ".git" / "BISECT_START"
        if bisect_start.exists():
            raise LLVMBisectError(
                "A bisect is already in progress in the LLVM repository. "
                f"Run 'cd {self.llvm_dir} && git bisect reset' first."
            )

        # Check test script exists
        if not self.test_script.exists():
            raise LLVMBisectError(f"Test script not found: {self.test_script}")

        # Check LLVM working directory status (warning only)
        result = self.executor.run_command(
            ["git", "status", "--porcelain"],
            cwd=str(self.llvm_dir),
        )
        if result.stdout.strip():
            self.logger.warning(
                "LLVM working directory has uncommitted changes. "
                "This may cause issues during bisect."
            )

        self.logger.info("Pre-bisect checks passed")

    def _prepare_script(self) -> str:
        """
        Get the path to the embedded bisect script.

        Returns:
            Absolute path to bisect_llvm.sh script.
        """
        return get_bisect_llvm_script()

    def _parse_bisect_result(self, output: str) -> str:
        """
        Parse the culprit commit from git bisect output.

        The output contains a line like:
        "<40-char-hash> is the first bad commit"

        Args:
            output: The stdout from git bisect run.

        Returns:
            The culprit LLVM commit hash.

        Raises:
            LLVMBisectError: If cannot parse the result.
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
        raise LLVMBisectError(
            f"Cannot parse bisect result. Expected '<hash> is the first bad commit' "
            f"in output:\n{output[-500:]}"  # Last 500 chars for context
        )
