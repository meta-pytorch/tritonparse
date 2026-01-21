# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Triton bisect executor for finding regression-causing commits.

This module implements Phase 1 of the bisect workflow: bisecting Triton
commits to find the first bad commit that causes a test to fail.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Union

from tritonparse.bisect.base_bisector import BaseBisector, BisectError
from tritonparse.bisect.config import get_config
from tritonparse.bisect.scripts import get_bisect_triton_script


class TritonBisectError(BisectError):
    """Exception raised for Triton bisect related errors."""

    pass


class TritonBisector(BaseBisector):
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

    @property
    def bisect_name(self) -> str:
        """Name of the bisect operation."""
        return "Phase 1: Triton Bisect"

    @property
    def default_build_command(self) -> str:
        """Default build command for Triton."""
        if get_config().use_uv:
            return "uv pip install -e ."
        return "pip install -e ."

    @property
    def target_repo_dir(self) -> Path:
        """Directory where git bisect runs (Triton repo)."""
        return self.triton_dir

    def _get_bisect_script(self) -> Union[str, Path]:
        """Get the path to the Triton bisect script."""
        return get_bisect_triton_script()

    def _get_extra_env_vars(self) -> Dict[str, str]:
        """No extra environment variables needed for Triton bisect."""
        return {}

    def _log_header(self, good_commit: str, bad_commit: str) -> None:
        """Log Triton-specific header information."""
        self.logger.info("=" * 60)
        self.logger.info(self.bisect_name)
        self.logger.info("=" * 60)
        self.logger.info(f"Triton directory: {self.triton_dir}")
        self.logger.info(f"Test script: {self.test_script}")
        self.logger.info(f"Good commit: {good_commit}")
        self.logger.info(f"Bad commit: {bad_commit}")
        self.logger.info(f"Conda environment: {self.conda_env}")
        self.logger.info(f"Build command: {self.build_command}")

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
        try:
            return self._run_bisect(good_commit, bad_commit, output_callback)
        except BisectError as e:
            raise TritonBisectError(str(e)) from e
