# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Git utility functions for bisect operations.

This module provides shared helper functions for git repository management
used by both EnvironmentManager and LLVMBisector.
"""

from pathlib import Path
from typing import Optional, Tuple

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger


def verify_git_repo(
    repo_dir: Path,
    executor: ShellExecutor,
) -> Tuple[bool, str]:
    """
    Verify a directory is a valid git repository.

    Args:
        repo_dir: Path to the repository directory.
        executor: ShellExecutor instance for running commands.

    Returns:
        Tuple of (is_valid, current_commit_hash).
        If not valid, current_commit_hash is empty string.
    """
    result = executor.run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_dir),
    )
    if result.success:
        return (True, result.stdout.strip())
    return (False, "")


def ensure_git_repo(
    repo_dir: Path,
    repo_url: str,
    repo_name: str,
    executor: ShellExecutor,
    logger: BisectLogger,
    use_streaming: bool = False,
    fetch_updates: bool = True,
    cwd: Optional[Path] = None,
) -> str:
    """
    Clone or verify a git repository.

    This function handles the common pattern of:
    - If directory doesn't exist: git clone
    - If directory exists but not valid git repo: raise error
    - If directory exists and is valid: optionally git fetch origin

    Args:
        repo_dir: Directory where repo should be located.
        repo_url: URL to clone from (e.g., "https://github.com/llvm/llvm-project").
        repo_name: Human-readable name for logging (e.g., "LLVM", "Triton").
        executor: ShellExecutor instance for running commands.
        logger: BisectLogger instance for logging.
        use_streaming: Use streaming output for clone (for large repos).
        fetch_updates: Whether to fetch updates if repo already exists.
        cwd: Working directory for clone command (defaults to repo_dir's parent).

    Returns:
        Current commit hash of the repository.

    Raises:
        RuntimeError: If repo is invalid or operations fail.
    """
    if not repo_dir.exists():
        logger.info(f"{repo_name} repo not found at {repo_dir}")
        logger.info(f"Cloning {repo_name} repo from {repo_url}...")

        clone_cmd = ["git", "clone", repo_url, str(repo_dir)]
        clone_cwd = str(cwd) if cwd else None

        if use_streaming:
            result = executor.run_command_streaming(clone_cmd, cwd=clone_cwd)
        else:
            result = executor.run_command(clone_cmd, cwd=clone_cwd)

        if not result.success:
            raise RuntimeError(f"Failed to clone {repo_name} repo: {result.stderr}")

        logger.info(f"{repo_name} repo cloned successfully")

    # Verify it's a valid git repository
    is_valid, current_commit = verify_git_repo(repo_dir, executor)

    if not is_valid:
        raise RuntimeError(
            f"{repo_name} directory exists but is not a valid git repository: {repo_dir}\n"
            f"Please remove it and retry: rm -rf {repo_dir}"
        )

    logger.info(f"{repo_name} repo verified at: {repo_dir}")
    logger.info(f"Current {repo_name} commit: {current_commit[:12]}")

    # Optionally fetch updates
    if fetch_updates and repo_dir.exists():
        logger.info(f"Fetching {repo_name} updates...")
        result = executor.run_command(
            ["git", "fetch", "origin"],
            cwd=str(repo_dir),
        )
        if not result.success:
            raise RuntimeError(f"Failed to fetch {repo_name} updates: {result.stderr}")
        logger.info("Fetch complete.")

    return current_commit
