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
from tritonparse.bisect.git_utils import ensure_git_repo, verify_git_repo
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
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Setting up environment...")
        self.logger.info("=" * 60)

        self._ensure_triton_repo()
        self._ensure_llvm_repo()
        self._ensure_conda_env(conda_env)

        self.logger.info("")
        self.logger.info("✅ Environment setup complete!")
        self.logger.info(f"   Triton: {self.triton_dir}")
        self.logger.info(f"   LLVM:   {self.llvm_dir}")
        self.logger.info(f"   Conda:  {conda_env}")
        self.logger.info("=" * 60)
        self.logger.info("")

    def _ensure_triton_repo(self) -> None:
        """
        Clone or update Triton repository.

        Behavior:
        - If directory doesn't exist: git clone
        - If directory exists but not valid git repo: raise error
        - If directory exists and is valid: git fetch origin (update remote refs)

        Raises:
            RuntimeError: If git command fails or repo is invalid.
        """
        self.logger.info("")
        ensure_git_repo(
            repo_dir=self.triton_dir,
            repo_url=self.TRITON_REPO,
            repo_name="Triton",
            executor=self.executor,
            logger=self.logger,
            use_streaming=False,
            fetch_updates=True,
        )

    def _ensure_llvm_repo(self) -> None:
        """
        Clone or update LLVM repository.

        Note: LLVM repo is large (~2GB), first clone may take a while.

        Behavior:
        - If directory doesn't exist: git clone
        - If directory exists but not valid git repo: raise error
        - If directory exists and is valid: git fetch origin (update remote refs)

        Raises:
            RuntimeError: If git command fails or repo is invalid.
        """
        self.logger.info("")
        self.logger.info(
            "(This may take a while, LLVM repo is ~2GB)"
            if not self.llvm_dir.exists()
            else ""
        )
        ensure_git_repo(
            repo_dir=self.llvm_dir,
            repo_url=self.LLVM_REPO,
            repo_name="LLVM",
            executor=self.executor,
            logger=self.logger,
            use_streaming=False,
            fetch_updates=True,
        )

    def _ensure_conda_env(self, env_name: str) -> None:
        """
        Ensure conda environment exists.

        Behavior:
        - Check if conda is available
        - Check if environment exists
        - If not, create new environment with Python 3.12

        Note: You may need to accept the conda license on first run.
        Dependencies are installed later when running `make dev-install` in Triton.

        Args:
            env_name: Name of conda environment.

        Raises:
            RuntimeError: If conda is not installed.
        """
        self.logger.info("")
        self.logger.info(f"🐍 Checking conda environment: {env_name}")

        # Check if conda is available
        result = self.executor.run_command(
            ["conda", "--version"],
        )
        if not result.success:
            raise RuntimeError(
                "conda not found. Please install miniconda:\n"
                "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\n"
                "  bash Miniconda3-latest-Linux-x86_64.sh"
            )

        # Check if environment exists
        result = self.executor.run_command(
            ["conda", "env", "list"],
        )

        # Parse conda env list output to check for exact environment name
        env_exists = False
        for line in result.stdout.splitlines():
            # Each line is like: "env_name    /path/to/env" or "env_name *  /path/to/env"
            parts = line.split()
            if parts and parts[0] == env_name:
                env_exists = True
                break

        if not env_exists:
            self.logger.info(f"   Creating conda environment: {env_name}")
            # Note: You may need to accept the conda license on first run
            result = self.executor.run_command(
                ["conda", "create", "-n", env_name, "python=3.12", "-y"],
            )
            if not result.success:
                raise RuntimeError(
                    f"Failed to create conda environment: {result.stderr}"
                )
            self.logger.info(f"   ✅ Conda environment '{env_name}' created")
        else:
            self.logger.info(f"   ✅ Conda environment '{env_name}' already exists")

    def check_environment_status(self) -> Dict[str, bool]:
        """
        Check environment status (for diagnostics).

        Returns:
            Dictionary containing status of each component:
            - triton_exists: Whether Triton directory exists
            - triton_is_valid_repo: Whether it's a valid git repo (verified via git rev-parse)
            - llvm_exists: Whether LLVM directory exists
            - llvm_is_valid_repo: Whether it's a valid git repo (verified via git rev-parse)
            - conda_available: Whether conda is available
        """
        status: Dict[str, bool] = {}

        # Triton
        status["triton_exists"] = self.triton_dir.exists()
        if status["triton_exists"]:
            is_valid, _ = verify_git_repo(self.triton_dir, self.executor)
            status["triton_is_valid_repo"] = is_valid
        else:
            status["triton_is_valid_repo"] = False

        # LLVM
        status["llvm_exists"] = self.llvm_dir.exists()
        if status["llvm_exists"]:
            is_valid, _ = verify_git_repo(self.llvm_dir, self.executor)
            status["llvm_is_valid_repo"] = is_valid
        else:
            status["llvm_is_valid_repo"] = False

        # Conda
        result = self.executor.run_command(
            ["conda", "--version"],
        )
        status["conda_available"] = result.success

        return status

    def print_status(self) -> None:
        """Print formatted environment status (for CLI --status option)."""
        status = self.check_environment_status()

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Environment Status")
        self.logger.info("=" * 60)
        self.logger.info(f"📁 Triton Directory: {self.triton_dir}")
        self.logger.info(
            f"   Exists:         {'✅' if status['triton_exists'] else '❌'}"
        )
        self.logger.info(
            f"   Valid Git Repo: {'✅' if status['triton_is_valid_repo'] else '❌'}"
        )

        self.logger.info(f"📁 LLVM Directory: {self.llvm_dir}")
        self.logger.info(
            f"   Exists:         {'✅' if status['llvm_exists'] else '❌'}"
        )
        self.logger.info(
            f"   Valid Git Repo: {'✅' if status['llvm_is_valid_repo'] else '❌'}"
        )

        self.logger.info("🐍 Conda")
        self.logger.info(
            f"   Available:      {'✅' if status['conda_available'] else '❌'}"
        )
        self.logger.info("=" * 60)
        self.logger.info("")
