# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Environment Manager for LLVM bisect.

Manages the setup and maintenance of environment required for LLVM bisect:
- Triton repository (from GitHub)
- LLVM repository (from GitHub)
- Conda environment

This is useful for auto-setup when running bisect from environments that
don't already have the required repositories cloned.

Usage:
    With --auto-env-setup flag:
    tritonparseoss bisect --llvm-only --auto-env-setup \
        --triton-dir ~/oss-triton \
        --good-llvm abc123 --bad-llvm def456 \
        --test-script ~/test.py
"""

from pathlib import Path
from typing import Dict

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger


class EnvironmentManager:
    """
    Manages environment for LLVM bisect.

    Responsibilities:
    - Clone/update Triton repository from GitHub
    - Clone/update LLVM repository from GitHub
    - Create/verify conda environment with required dependencies

    Example:
        >>> logger = BisectLogger("./logs")
        >>> manager = EnvironmentManager(Path("~/oss-triton"), logger)
        >>> manager.ensure_environment("triton_bisect")
    """

    TRITON_REPO = "https://github.com/triton-lang/triton"
    LLVM_REPO = "https://github.com/llvm/llvm-project"

    def __init__(self, triton_dir: Path, logger: BisectLogger) -> None:
        """
        Initialize EnvironmentManager.

        Args:
            triton_dir: Directory where Triton will be cloned.
            logger: BisectLogger instance for logging.
        """
        self.triton_dir = triton_dir
        self.llvm_dir = triton_dir / "llvm-project"
        self.logger = logger

        # Create executor for running commands
        self.executor = ShellExecutor(self.logger)

    def ensure_environment(self, conda_env: str) -> None:
        """
        Ensure environment is properly set up.

        This method:
        1. Clones or updates Triton repository
        2. Clones or updates LLVM repository
        3. Creates or verifies conda environment

        Args:
            conda_env: Name of conda environment to use.

        Raises:
            subprocess.CalledProcessError: If git or conda commands fail.
            RuntimeError: If conda is not available.
        """
        # TODO: Phase 2 implementation
        raise NotImplementedError("Will be implemented in Phase 2")

    def _ensure_triton_repo(self) -> None:
        """Clone or update Triton repository."""
        # TODO: Phase 2 implementation
        raise NotImplementedError("Will be implemented in Phase 2")

    def _ensure_llvm_repo(self) -> None:
        """Clone or update LLVM repository."""
        # TODO: Phase 2 implementation
        raise NotImplementedError("Will be implemented in Phase 2")

    def _ensure_conda_env(self, env_name: str) -> None:
        """Create or verify conda environment."""
        # TODO: Phase 2 implementation
        raise NotImplementedError("Will be implemented in Phase 2")

    def check_environment_status(self) -> Dict[str, bool]:
        """
        Check environment status (for diagnostics).

        Returns:
            Dictionary containing status of each component:
            - triton_exists: Whether Triton directory exists
            - triton_is_valid_repo: Whether it's a valid git repo
            - llvm_exists: Whether LLVM directory exists
            - llvm_is_valid_repo: Whether it's a valid git repo
            - conda_available: Whether conda is available
        """
        # TODO: Phase 2 implementation
        raise NotImplementedError("Will be implemented in Phase 2")

    def print_status(self) -> None:
        """Print formatted environment status (for CLI --status option)."""
        # TODO: Phase 2 implementation
        raise NotImplementedError("Will be implemented in Phase 2")
