# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
LLVM bisect executor for finding regression-causing commits in LLVM.

This module implements Phase 4 of the bisect workflow: bisecting LLVM
commits within a Triton-compatible range to find the first bad LLVM commit.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Union

from tritonparse.bisect.base_bisector import BaseBisector, BisectError
from tritonparse.bisect.logger import BisectLogger
from tritonparse.bisect.scripts import get_bisect_llvm_script


class LLVMBisectError(BisectError):
    """Exception raised for LLVM bisect related errors."""

    pass


class LLVMBisector(BaseBisector):
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

    Note: Unlike TritonBisector, LLVMBisector does not support custom build commands.
    The build process is handled by bisect_llvm.sh which splits into two phases:
    - Phase 1: Build LLVM using scripts/build-llvm-project.sh
    - Phase 2: Build Triton using make dev-install with LLVM env vars

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

    def __init__(
        self,
        triton_dir: str,
        test_script: str,
        conda_env: str,
        logger: BisectLogger,
    ) -> None:
        """
        Initialize the LLVM bisector.

        Args:
            triton_dir: Path to the Triton repository.
            test_script: Path to the test script that determines pass/fail.
            conda_env: Name of the conda environment to use for builds.
            logger: BisectLogger instance for logging.
        """
        # LLVM bisect doesn't use build_command - the build is handled
        # by bisect_llvm.sh with fixed two-phase build process
        super().__init__(triton_dir, test_script, conda_env, logger, build_command=None)
        self.llvm_dir = self.triton_dir / "llvm-project"
        # Store triton_commit for use in _prepare_before_bisect
        self._triton_commit: Optional[str] = None

    @property
    def bisect_name(self) -> str:
        """Name of the bisect operation."""
        return "Phase 4: LLVM Bisect"

    @property
    def default_build_command(self) -> str:
        """Default build command for LLVM."""
        return "make dev-install-llvm"

    @property
    def target_repo_dir(self) -> Path:
        """Directory where git bisect runs (LLVM repo)."""
        return self.llvm_dir

    def _get_bisect_script(self) -> Union[str, Path]:
        """Get the path to the LLVM bisect script."""
        return get_bisect_llvm_script()

    def _get_extra_env_vars(self) -> Dict[str, str]:
        """LLVM-specific environment variables."""
        return {
            "COMPAT_MODE": "0",  # Normal mode: find regression
        }

    def _log_header(self, good_commit: str, bad_commit: str) -> None:
        """Log LLVM-specific header information."""
        self.logger.info("=" * 60)
        self.logger.info(self.bisect_name)
        self.logger.info("=" * 60)
        self.logger.info(f"Triton directory: {self.triton_dir}")
        self.logger.info(f"Triton commit: {self._triton_commit}")
        self.logger.info(f"Test script: {self.test_script}")
        self.logger.info(f"Good LLVM commit: {good_commit}")
        self.logger.info(f"Bad LLVM commit: {bad_commit}")
        self.logger.info(f"Conda environment: {self.conda_env}")
        self.logger.info(f"Build command: {self.build_command}")

    def _prepare_before_bisect(self) -> None:
        """Checkout Triton commit and ensure LLVM repo exists."""
        # Checkout Triton to the specified commit
        self._checkout_triton(self._triton_commit)

        # Ensure LLVM repo exists
        self._ensure_llvm_repo()

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
            raise LLVMBisectError(f"Failed to get Triton HEAD: {result.stderr}")

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

        If the LLVM repo doesn't exist, clone it from GitHub.
        If it exists, verify it's a valid git repository.

        Raises:
            LLVMBisectError: If LLVM repo cannot be cloned or is invalid.
        """
        LLVM_REPO_URL = "https://github.com/llvm/llvm-project"

        if not self.llvm_dir.exists():
            self.logger.info(f"LLVM repo not found at {self.llvm_dir}")
            self.logger.info(f"Cloning LLVM repo from {LLVM_REPO_URL}...")

            result = self.executor.run_command_streaming(
                ["git", "clone", LLVM_REPO_URL, str(self.llvm_dir)],
                cwd=str(self.triton_dir),
            )

            if not result.success:
                raise LLVMBisectError(f"Failed to clone LLVM repo: {result.stderr}")

            self.logger.info("LLVM repo cloned successfully")

        # Verify it's a valid git repository
        result = self.executor.run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=str(self.llvm_dir),
        )

        if not result.success:
            raise LLVMBisectError(
                f"LLVM directory exists but is not a valid git repository: {self.llvm_dir}\n"
                f"Please remove it and retry: rm -rf {self.llvm_dir}"
            )

        current_commit = result.stdout.strip()
        self.logger.info(f"LLVM repo verified at: {self.llvm_dir}")
        self.logger.info(f"Current LLVM commit: {current_commit[:12]}")

    def run(
        self,
        triton_commit: Optional[str],
        good_llvm: str,
        bad_llvm: str,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute LLVM bisect to find the culprit commit.

        Args:
            triton_commit: Fixed Triton commit to use. If None, uses current HEAD.
            good_llvm: Known good LLVM commit hash (test passes).
            bad_llvm: Known bad LLVM commit hash (test fails).
            output_callback: Optional callback called for each output line.
                            Used by TUI to display real-time output.

        Returns:
            The culprit LLVM commit hash (first bad commit).

        Raises:
            LLVMBisectError: If bisect fails or cannot parse the result.
        """
        # Store triton commit for _prepare_before_bisect
        self._triton_commit = self._get_triton_commit(triton_commit)

        try:
            return self._run_bisect(good_llvm, bad_llvm, output_callback)
        except BisectError as e:
            raise LLVMBisectError(str(e)) from e
