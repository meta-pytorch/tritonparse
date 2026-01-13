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
        """Checkout Triton commit and ensure LLVM repo exists.

        Note: _ensure_llvm_repo() is called earlier in run() to support
        _get_next_commit(), so we skip it here to avoid redundant calls.
        """
        # Checkout Triton to the specified commit
        self._checkout_triton(self._triton_commit)

        # Note: LLVM repo is already ensured in run() before _get_next_commit()
        # No need to call _ensure_llvm_repo() again

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

    def _get_next_commit(self, good_llvm: str, bad_llvm: str) -> str:
        """
        Get the next commit after good_llvm (in the direction of bad_llvm).

        The commits.csv uses 'llvm_commit_last_compatible' which represents
        the LAST LLVM commit compatible with a Triton version. When bisecting
        between pairs, we need to use the NEXT commit after 'last compatible'
        as the actual good starting point.

        For example:
            - Pair 7: Triton 28d6b1e6, LLVM 067f87c5 (last compatible for 28d6b1e6)
            - Pair 8: Triton a4d79034, LLVM bcb48aa5 (last compatible for a4d79034)

        When bisecting with Triton a4d79034:
            - good_llvm = 067f87c5 (from Pair 7)
            - We need the NEXT commit after 067f87c5 as actual good
            - Because 067f87c5 itself might not be compatible with a4d79034

        Args:
            good_llvm: The "last compatible" LLVM commit from commits.csv.
            bad_llvm: The bad LLVM commit.

        Returns:
            The next commit after good_llvm, or good_llvm if no next commit exists.

        Raises:
            LLVMBisectError: If git command fails.
        """
        self.logger.info(f"Finding next commit after {good_llvm[:12]}...")

        # Get all commits between good_llvm (exclusive) and bad_llvm (inclusive)
        # --reverse gives us oldest first, so the first one is the next after good_llvm
        result = self.executor.run_command(
            ["git", "rev-list", f"{good_llvm}..{bad_llvm}", "--reverse"],
            cwd=str(self.llvm_dir),
        )

        if not result.success:
            raise LLVMBisectError(
                f"Failed to get commits between {good_llvm} and {bad_llvm}: {result.stderr}"
            )

        commits = result.stdout.strip().split("\n")
        if commits and commits[0]:
            next_commit = commits[0]
            self.logger.info(f"Next commit after {good_llvm[:12]}: {next_commit[:12]}")
            self.logger.info(f"Total commits in range: {len(commits)}")
            return next_commit

        # If no commits between them, they might be the same or adjacent
        self.logger.warning(
            f"No commits found between {good_llvm[:12]} and {bad_llvm[:12]}. "
            f"Using original good_llvm."
        )
        return good_llvm

    def _checkout_llvm(self, commit: str) -> None:
        """
        Checkout the specified LLVM commit.

        Args:
            commit: The LLVM commit hash to checkout.

        Raises:
            LLVMBisectError: If checkout fails.
        """
        self.logger.info(f"Checking out LLVM commit: {commit[:12]}")

        # First fetch the commit (in case it's not in local history)
        # Fetch might fail if commit is already local, that's okay
        self.executor.run_command(
            ["git", "fetch", "origin", commit],
            cwd=str(self.llvm_dir),
        )

        # Checkout the commit
        result = self.executor.run_command(
            ["git", "checkout", commit],
            cwd=str(self.llvm_dir),
        )

        if not result.success:
            raise LLVMBisectError(
                f"Failed to checkout LLVM commit {commit}: {result.stderr}"
            )

    def _build_and_test(
        self,
        llvm_commit: str,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> tuple:
        """
        Build Triton with the specified LLVM commit and run the test.

        This method:
        1. Checks out the LLVM commit
        2. Runs the bisect script (builds LLVM + Triton, runs test)
        3. Returns the build and test results

        Args:
            llvm_commit: The LLVM commit to build and test.
            output_callback: Optional callback for real-time output.

        Returns:
            Tuple of (build_success: bool, test_passed: bool, exit_code: int)
            - build_success: True if both LLVM and Triton built successfully
            - test_passed: True if test exited with code 0
            - exit_code: The actual exit code from the bisect script
        """
        # Checkout LLVM commit
        self._checkout_llvm(llvm_commit)

        # Set up environment
        env = self._get_base_env_vars()
        env.update(self._get_extra_env_vars())

        # Get bisect script
        script_path = str(self._get_bisect_script())

        # Run the bisect script
        result = self.executor.run_command_streaming(
            ["bash", script_path],
            cwd=str(self.llvm_dir),
            env=env,
            output_callback=output_callback,
        )

        exit_code = result.exit_code

        # Interpret exit codes:
        # 0 = good (build success, test passed)
        # 1 = bad (build success, test failed)
        # 125 = skip (LLVM build failed)
        # 128 = abort (Triton build failed or other error)

        if exit_code == 125:
            # LLVM build failed
            return (False, False, exit_code)
        elif exit_code == 128:
            # Triton build failed or other error
            return (False, False, exit_code)
        elif exit_code == 0:
            # Build success, test passed
            return (True, True, exit_code)
        elif exit_code == 1:
            # Build success, test failed
            return (True, False, exit_code)
        else:
            # Unknown exit code, treat as build failure
            return (False, False, exit_code)

    def _validate_commits(
        self,
        good_llvm: str,
        bad_llvm: str,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Validate good and bad commits before starting bisect.

        This ensures:
        1. Bad commit: builds successfully AND test FAILS
        2. Good commit: builds successfully AND test PASSES

        Args:
            good_llvm: The good LLVM commit (adjusted, not from CSV directly).
            bad_llvm: The bad LLVM commit.
            output_callback: Optional callback for real-time output.

        Raises:
            LLVMBisectError: If validation fails for either commit.
        """
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("VALIDATING COMMITS BEFORE BISECT")
        self.logger.info("=" * 60)

        # Validate bad commit first (should build + fail test)
        self.logger.info("")
        self.logger.info(f"Validating BAD commit: {bad_llvm[:12]}")
        self.logger.info("  Expected: build SUCCESS + test FAIL")

        build_success, test_passed, exit_code = self._build_and_test(
            bad_llvm, output_callback
        )

        if not build_success:
            raise LLVMBisectError(
                f"Bad commit validation failed: {bad_llvm}\n"
                f"Build FAILED (exit code: {exit_code})\n"
                f"Expected: build success + test fail\n"
                f"Please verify this commit in commits.csv is correct."
            )

        if test_passed:
            raise LLVMBisectError(
                f"Bad commit validation failed: {bad_llvm}\n"
                f"Test PASSED (expected FAIL)\n"
                f"This commit should reproduce the regression but the test passed.\n"
                f"Please verify:\n"
                f"  1. The test script correctly detects the regression\n"
                f"  2. The bad LLVM commit in commits.csv is correct"
            )

        self.logger.info("  ✓ Bad commit validated: build SUCCESS, test FAIL")

        # Validate good commit (should build + pass test)
        self.logger.info("")
        self.logger.info(f"Validating GOOD commit: {good_llvm[:12]}")
        self.logger.info("  Expected: build SUCCESS + test PASS")

        build_success, test_passed, exit_code = self._build_and_test(
            good_llvm, output_callback
        )

        if not build_success:
            raise LLVMBisectError(
                f"Good commit validation failed: {good_llvm}\n"
                f"Build FAILED (exit code: {exit_code})\n"
                f"Expected: build success + test pass\n"
                f"Please verify the commit range in commits.csv is correct.\n"
                f"Note: This is the next commit after 'last compatible', "
                f"so it should be compatible with the current Triton version."
            )

        if not test_passed:
            raise LLVMBisectError(
                f"Good commit validation failed: {good_llvm}\n"
                f"Test FAILED (expected PASS)\n"
                f"This commit should pass the test but it failed.\n"
                f"Please verify:\n"
                f"  1. The test script is correct\n"
                f"  2. The good LLVM commit range in commits.csv is correct\n"
                f"  3. The LLVM range may need to be split into smaller ranges"
            )

        self.logger.info("  ✓ Good commit validated: build SUCCESS, test PASS")

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("VALIDATION COMPLETE - Starting bisect")
        self.logger.info("=" * 60)
        self.logger.info("")

    def run(
        self,
        triton_commit: Optional[str],
        good_llvm: str,
        bad_llvm: str,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute LLVM bisect to find the culprit commit.

        The good_llvm is expected to be from commits.csv's 'llvm_commit_last_compatible'
        column, which represents the LAST compatible LLVM commit for a Triton version.
        This method automatically adjusts to use the NEXT commit as the actual good
        starting point for bisect.

        Args:
            triton_commit: Fixed Triton commit to use. If None, uses current HEAD.
            good_llvm: Known good LLVM commit hash from commits.csv (last compatible).
                       Will be adjusted to use the next commit as actual good.
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

        # Ensure LLVM repo exists before getting next commit
        # (need repo to run git commands)
        self._ensure_llvm_repo()

        # Adjust good commit to use the next commit after 'last compatible'
        # This is necessary because commits.csv stores 'llvm_commit_last_compatible'
        # which is the last LLVM compatible with the PREVIOUS Triton version,
        # not necessarily compatible with the current Triton version.
        self.logger.info("")
        self.logger.info("=" * 40)
        self.logger.info("ADJUSTING GOOD COMMIT")
        self.logger.info(f"  Original (last compatible): {good_llvm[:12]}")
        actual_good = self._get_next_commit(good_llvm, bad_llvm)
        self.logger.info(f"  Adjusted (next commit):     {actual_good[:12]}")
        self.logger.info("=" * 40)
        self.logger.info("")

        # Checkout Triton before validation (needed for building)
        self._checkout_triton(self._triton_commit)

        # Validate both commits before starting bisect
        # This ensures we don't waste time on a bisect that will fail
        self._validate_commits(actual_good, bad_llvm, output_callback)

        try:
            return self._run_bisect(actual_good, bad_llvm, output_callback)
        except BisectError as e:
            raise LLVMBisectError(str(e)) from e
