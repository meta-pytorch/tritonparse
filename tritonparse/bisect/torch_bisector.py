# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
PyTorch bisect executor for finding regression-causing commits.

This module implements a PyTorch-only bisect workflow using the existing
git-bisect shell runner infrastructure. It targets a PyTorch checkout and
reuses the generic build/test loop with PyTorch-specific logging.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, TYPE_CHECKING

from tritonparse.bisect.base_bisector import BaseBisector, BisectError
from tritonparse.bisect.scripts import (
    get_bisect_torch_script,
    get_build_pytorch_script,
    get_prepare_build_pytorch_script,
)

if TYPE_CHECKING:
    from tritonparse.bisect.logger import BisectLogger


class TorchBisectError(BisectError):
    """Exception raised for PyTorch bisect related errors."""

    pass


class TorchBisector(BaseBisector):
    """
    PyTorch bisect executor.

    This bisects a local PyTorch checkout to find the first bad commit that
    causes the user-provided test script to fail.
    """

    def __init__(
        self,
        torch_dir: str,
        test_script: str,
        conda_env: str,
        logger: "BisectLogger",
        build_command: Optional[str] = None,
        per_commit_log: bool = False,
    ) -> None:
        """
        Initialize the PyTorch bisector.

        Args:
            torch_dir: Path to the PyTorch repository.
            test_script: Path to the test script that determines pass/fail.
            conda_env: Name of the conda environment to use for builds.
            logger: BisectLogger instance for logging.
            build_command: Custom build command. Defaults to editable install.
        """
        # BaseBisector requires ``triton_dir`` as the primary repo directory.
        # For PyTorch bisect we reuse that slot for the PyTorch checkout so
        # the base-class pre-bisect checks, env-var setup, and git-bisect
        # invocation all operate on the correct repository.
        super().__init__(
            triton_dir=torch_dir,
            test_script=test_script,
            conda_env=conda_env,
            logger=logger,
            build_command=build_command,
            per_commit_log=per_commit_log,
        )

    @property
    def bisect_name(self) -> str:
        """Name of the bisect operation."""
        return "PyTorch Bisect"

    @property
    def default_build_command(self) -> str:
        """Default build command for PyTorch."""
        return f"bash {get_build_pytorch_script()}"

    @property
    def target_repo_dir(self) -> Path:
        """Directory where git bisect runs (PyTorch repo)."""
        return self.triton_dir

    def _get_bisect_script(self) -> Path:
        """Get the path to the PyTorch bisect script."""
        return get_bisect_torch_script()

    def _get_extra_env_vars(self) -> Dict[str, str]:
        """No additional environment variables are required."""
        return {}

    def _log_header(self, good_commit: str, bad_commit: str) -> None:
        """Log PyTorch-specific header information."""
        self.logger.info("=" * 60)
        self.logger.info(self.bisect_name)
        self.logger.info("=" * 60)
        self.logger.info(f"PyTorch directory: {self.target_repo_dir}")
        self.logger.info(f"Test script: {self.test_script}")
        self.logger.info(f"Good commit: {good_commit}")
        self.logger.info(f"Bad commit: {bad_commit}")
        self.logger.info(f"Conda environment: {self.conda_env}")
        self.logger.info(f"Build command: {self.build_command}")

    def _prepare_before_bisect(self) -> None:
        """Run the prepare_build_pytorch.sh script to install dependencies."""
        script = str(get_prepare_build_pytorch_script())
        self.logger.info(f"Running build preparation: {script}")
        result = self.executor.run_command(
            ["bash", script],
            cwd=str(self.target_repo_dir),
        )
        if result.exit_code != 0:
            raise TorchBisectError(
                f"Build preparation failed (exit {result.exit_code}): {result.stderr}"
            )

    def run(
        self,
        good_commit: str,
        bad_commit: str,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Execute PyTorch bisect to find the culprit commit.

        Args:
            good_commit: Known good commit hash or tag.
            bad_commit: Known bad commit hash or tag.
            output_callback: Optional callback for real-time output.

        Returns:
            The culprit commit hash (first bad commit).

        Raises:
            TorchBisectError: If bisect fails or cannot parse the result.
        """
        try:
            return self._run_bisect(good_commit, bad_commit, output_callback)
        except BisectError as e:
            raise TorchBisectError(str(e)) from e
