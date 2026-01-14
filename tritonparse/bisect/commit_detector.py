# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Commit type detector for Triton bisect workflow.

This module provides the CommitDetector class which implements Phase 2 of the
Triton/LLVM bisect workflow. It detects whether a given Triton commit is an
LLVM bump by checking if the cmake/llvm-hash.txt file was modified.
"""

from pathlib import Path

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger


class CommitDetectorError(Exception):
    """Exception raised for commit detection errors."""

    pass


class CommitDetector:
    """
    Detects the type of a Triton commit.

    This class implements Phase 2 of the bisect workflow - determining whether
    a Triton commit is an LLVM bump by checking if cmake/llvm-hash.txt was
    modified in the commit.

    Example:
        >>> logger = BisectLogger("./logs")
        >>> executor = ShellExecutor(logger)
        >>> detector = CommitDetector(
        ...     triton_dir=Path("/path/to/triton"),
        ...     executor=executor,
        ...     logger=logger,
        ... )
        >>> is_bump = detector.is_llvm_bump(commit="abc123")
        >>> if is_bump:
        ...     print("This commit is an LLVM bump!")
    """

    LLVM_HASH_FILE = "cmake/llvm-hash.txt"

    def __init__(
        self,
        triton_dir: Path,
        executor: ShellExecutor,
        logger: BisectLogger,
    ) -> None:
        """
        Initialize the commit detector.

        Args:
            triton_dir: Path to the Triton repository.
            executor: ShellExecutor instance for running git commands.
            logger: BisectLogger instance for logging.
        """
        self.triton_dir = triton_dir
        self.executor = executor
        self.logger = logger

    def is_llvm_bump(self, commit: str) -> bool:
        """
        Check if a commit is an LLVM bump.

        A commit is considered an LLVM bump if it modifies the
        cmake/llvm-hash.txt file.

        Args:
            commit: The Triton commit hash to check.

        Returns:
            True if the commit modifies LLVM hash, False otherwise.
        """
        self.logger.info(f"Checking if commit is LLVM bump: {commit}")

        result = self.executor.run_command(
            ["git", "diff", "--name-only", f"{commit}~1", commit],
            cwd=str(self.triton_dir),
        )

        if not result.success:
            self.logger.warning(
                f"Failed to get changed files for {commit}: {result.stderr}"
            )
            return False

        changed_files = result.stdout.strip().split("\n")
        is_bump = self.LLVM_HASH_FILE in changed_files

        if is_bump:
            self.logger.info(f"Commit {commit[:7]} IS an LLVM bump")
        else:
            self.logger.info(f"Commit {commit[:7]} is NOT an LLVM bump")

        return is_bump
