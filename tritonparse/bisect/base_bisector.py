# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Base bisector class for git bisect operations.

This module provides the abstract base class that defines the common structure
and behavior for all bisector implementations (Triton, LLVM, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger


class BisectError(Exception):
    """Base exception for bisect related errors."""

    pass


class BaseBisector(ABC):
    """
    Abstract base class for bisect executors.

    This class implements the Template Method pattern, defining the common
    bisect workflow structure while allowing subclasses to customize specific
    steps.

    The bisect workflow consists of:
    1. Log header information
    2. Prepare before bisect (subclass hook)
    3. Pre-bisect validation checks
    4. Get the bisect script
    5. Set up environment variables
    6. Execute git bisect sequence
    7. Parse and return the culprit commit

    Subclasses must implement:
    - bisect_name: Name of the bisect (for logging)
    - default_build_command: Default build command
    - target_repo_dir: Directory where git bisect runs
    - _get_bisect_script(): Return the bisect script path
    - _get_extra_env_vars(): Return additional environment variables

    Subclasses may override:
    - _prepare_before_bisect(): Hook for pre-bisect preparation
    - _log_header(): Custom header logging
    """

    def __init__(
        self,
        triton_dir: str,
        test_script: str,
        conda_env: str,
        logger: BisectLogger,
        build_command: Optional[str] = None,
    ) -> None:
        """
        Initialize the bisector.

        Args:
            triton_dir: Path to the Triton repository.
            test_script: Path to the test script that determines pass/fail.
            conda_env: Name of the conda environment to use for builds.
            logger: BisectLogger instance for logging.
            build_command: Custom build command. Defaults to subclass default.
        """
        self.triton_dir = Path(triton_dir).resolve()
        self.test_script = Path(test_script).resolve()
        self.conda_env = conda_env
        self.logger = logger
        self.build_command = build_command or self.default_build_command
        self.executor = ShellExecutor(logger)

    @property
    @abstractmethod
    def bisect_name(self) -> str:
        """Name of the bisect operation (e.g., 'Triton Bisect', 'LLVM Bisect')."""
        pass

    @property
    @abstractmethod
    def default_build_command(self) -> str:
        """Default build command for this bisector."""
        pass

    @property
    @abstractmethod
    def target_repo_dir(self) -> Path:
        """Directory where git bisect will be executed."""
        pass

    @abstractmethod
    def _get_bisect_script(self) -> Union[str, Path]:
        """
        Get the path to the bisect script.

        Returns:
            Path to the bisect script.
        """
        pass

    @abstractmethod
    def _get_extra_env_vars(self) -> Dict[str, str]:
        """
        Get additional environment variables specific to this bisector.

        Returns:
            Dictionary of additional environment variables.
        """
        pass

    def _prepare_before_bisect(self) -> None:  # noqa: B027
        """
        Hook for subclasses to perform preparation before bisect.

        This method is called after logging the header but before
        pre-bisect checks. Override in subclasses if needed.

        For example, LLVMBisector uses this to checkout Triton commit
        and ensure LLVM repo exists.
        """
        pass
