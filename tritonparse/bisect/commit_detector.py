# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Commit type detector for Triton bisect workflow.

This module provides the CommitDetector class which implements Phase 2 of the
Triton/LLVM bisect workflow. It detects whether a given Triton commit is an
LLVM bump by checking if the cmake/llvm-hash.txt file was modified.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger


class CommitDetectorError(Exception):
    """Exception raised for commit detection errors."""

    pass


@dataclass
class LLVMBumpInfo:
    """
    Information about an LLVM bump commit.

    Attributes:
        is_llvm_bump: Whether the commit modifies LLVM hash.
        old_hash: Previous LLVM commit hash (if bump detected).
        new_hash: New LLVM commit hash (if bump detected).
        triton_commit: The Triton commit that was analyzed.
    """

    is_llvm_bump: bool
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    triton_commit: Optional[str] = None


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
        >>> info = detector.detect(commit="abc123")
        >>> if info.is_llvm_bump:
        ...     print(f"LLVM bump: {info.old_hash} -> {info.new_hash}")
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

    def detect(self, commit: str) -> LLVMBumpInfo:
        """
        Detect whether a Triton commit is an LLVM bump.

        A commit is considered an LLVM bump if it modifies the
        cmake/llvm-hash.txt file.

        Args:
            commit: The Triton commit hash to analyze.

        Returns:
            LLVMBumpInfo with detection results and hash changes if applicable.

        Raises:
            CommitDetectorError: If detection fails due to git errors.
        """
        self.logger.info(f"Detecting commit type for: {commit}")

        # Check if this commit modifies the LLVM hash file
        if not self._is_llvm_bump_commit(commit):
            self.logger.info(f"Commit {commit[:7]} is NOT an LLVM bump")
            return LLVMBumpInfo(
                is_llvm_bump=False,
                triton_commit=commit,
            )

        # Get the old and new LLVM hashes
        old_hash, new_hash = self._get_llvm_hash_change(commit)

        self.logger.info(f"Commit {commit[:7]} IS an LLVM bump")
        self.logger.info(f"  LLVM hash change: {old_hash[:7]} -> {new_hash[:7]}")

        return LLVMBumpInfo(
            is_llvm_bump=True,
            old_hash=old_hash,
            new_hash=new_hash,
            triton_commit=commit,
        )

    def _is_llvm_bump_commit(self, commit: str) -> bool:
        """
        Check if a commit modifies the LLVM hash file.

        Args:
            commit: The commit hash to check.

        Returns:
            True if the commit modifies cmake/llvm-hash.txt, False otherwise.
        """
        result = self.executor.run_command(
            ["git", "diff", "--name-only", f"{commit}~1", commit],
            cwd=str(self.triton_dir),
        )

        if not result.success:
            self.logger.warning(
                f"Failed to get changed files for {commit}: {result.stderr}"
            )
            # Try alternative approach: check if file exists at both commits
            return self._is_llvm_bump_commit_fallback(commit)

        changed_files = result.stdout.strip().split("\n")
        return self.LLVM_HASH_FILE in changed_files

    def _is_llvm_bump_commit_fallback(self, commit: str) -> bool:
        """
        Fallback method to check for LLVM bump when diff fails.

        This can happen for merge commits or the first commit.

        Args:
            commit: The commit hash to check.

        Returns:
            True if LLVM hash changed, False otherwise.
        """
        try:
            # Get hash at commit
            result_at = self.executor.run_command(
                ["git", "show", f"{commit}:{self.LLVM_HASH_FILE}"],
                cwd=str(self.triton_dir),
            )
            if not result_at.success:
                return False

            # Get hash at parent
            result_parent = self.executor.run_command(
                ["git", "show", f"{commit}~1:{self.LLVM_HASH_FILE}"],
                cwd=str(self.triton_dir),
            )
            if not result_parent.success:
                return False

            # Compare hashes
            hash_at = self._extract_hash_from_content(result_at.stdout)
            hash_parent = self._extract_hash_from_content(result_parent.stdout)

            return hash_at != hash_parent

        except Exception as e:
            self.logger.debug(f"Fallback detection failed: {e}")
            return False

    def _get_llvm_hash_change(self, commit: str) -> Tuple[str, str]:
        """
        Get the old and new LLVM hashes from a bump commit.

        Args:
            commit: The LLVM bump commit hash.

        Returns:
            Tuple of (old_hash, new_hash).

        Raises:
            CommitDetectorError: If unable to extract hashes.
        """
        # Get hash at parent commit
        result_parent = self.executor.run_command(
            ["git", "show", f"{commit}~1:{self.LLVM_HASH_FILE}"],
            cwd=str(self.triton_dir),
        )

        if not result_parent.success:
            raise CommitDetectorError(
                f"Failed to get LLVM hash at {commit}~1: {result_parent.stderr}"
            )

        old_hash = self._extract_hash_from_content(result_parent.stdout)

        # Get hash at commit
        result_at = self.executor.run_command(
            ["git", "show", f"{commit}:{self.LLVM_HASH_FILE}"],
            cwd=str(self.triton_dir),
        )

        if not result_at.success:
            raise CommitDetectorError(
                f"Failed to get LLVM hash at {commit}: {result_at.stderr}"
            )

        new_hash = self._extract_hash_from_content(result_at.stdout)

        return old_hash, new_hash

    def _extract_hash_from_content(self, content: str) -> str:
        """
        Extract the LLVM commit hash from file content.

        The llvm-hash.txt file typically contains just a commit hash,
        possibly with whitespace or comments.

        Args:
            content: Content of the llvm-hash.txt file.

        Returns:
            The extracted LLVM commit hash.

        Raises:
            CommitDetectorError: If no valid hash found.
        """
        # Remove whitespace and comments
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Validate it looks like a git hash (hex string, 7-40 chars)
            if re.match(r"^[0-9a-fA-F]{7,40}$", line):
                return line

        raise CommitDetectorError(f"No valid hash found in content: {content[:100]}")

    def get_llvm_hash_at_commit(self, commit: str) -> str:
        """
        Get the LLVM hash at a specific Triton commit.

        This is useful for finding the LLVM version used by any Triton commit,
        not just bump commits.

        Args:
            commit: The Triton commit hash.

        Returns:
            The LLVM commit hash used at that Triton commit.

        Raises:
            CommitDetectorError: If unable to get the hash.
        """
        result = self.executor.run_command(
            ["git", "show", f"{commit}:{self.LLVM_HASH_FILE}"],
            cwd=str(self.triton_dir),
        )

        if not result.success:
            raise CommitDetectorError(
                f"Failed to get LLVM hash at {commit}: {result.stderr}"
            )

        return self._extract_hash_from_content(result.stdout)
