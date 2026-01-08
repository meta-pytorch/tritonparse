# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Dual logging system for bisect operations.

Provides separate logging for:
- Module logs: Python logging -> stdout + file
- Command logs: subprocess output -> file
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class BisectLogger:
    """
    Dual logging system for bisect operations.

    This logger provides two separate logging streams:
    1. Module logs: Standard Python logging output to both stdout and a log file
    2. Command logs: Subprocess command output written to a separate file

    Example:
        >>> logger = BisectLogger("./logs")
        >>> logger.info("Starting bisect...")
        >>> logger.log_command_output("git status", "output...", 0)
    """

    def __init__(
        self,
        log_dir: str,
        session_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the dual logging system.

        Args:
            log_dir: Directory path where log files will be stored.
            session_name: Optional session identifier. If not provided,
                         a timestamp will be used (format: YYYYMMDD_HHMMSS).
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name

        # Module log: stdout + file
        self.module_log_path = self.log_dir / f"{session_name}_bisect.log"
        self._setup_module_logger()

        # Command log: file only
        self.command_log_path = self.log_dir / f"{session_name}_bisect_commands.log"

        # Print log file locations
        self.info(f"Log directory: {self.log_dir}")
        self.info(f"  Module log: {self.module_log_path.name}")
        self.info(f"  Command log: {self.command_log_path.name}")

    def _setup_module_logger(self) -> None:
        """Configure the Python logging system with file and stdout handlers."""
        self.logger = logging.getLogger(f"bisect.{self.session_name}")
        self.logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers if logger already exists
        if self.logger.handlers:
            return

        # File handler - captures all levels
        fh = logging.FileHandler(self.module_log_path)
        fh.setLevel(logging.DEBUG)

        # Stdout handler - INFO and above only
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def log_command_output(
        self,
        command: str,
        output: str,
        exit_code: int,
        include_wrapper: bool = True,
    ) -> None:
        """
        Log command execution output.

        Writes to the command log file.

        Args:
            command: The command that was executed.
            output: Combined stdout and stderr output from the command.
            exit_code: The exit code returned by the command.
            include_wrapper: If True, include header/footer wrapper around output.
                            If False, only write the output content (used for streaming).
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write to file
        with open(self.command_log_path, "a") as f:
            if include_wrapper:
                f.write(f"\n{'=' * 60}\n")
                f.write(f"[{timestamp}] Command: {command}\n")
                f.write(f"Exit code: {exit_code}\n")
                f.write(f"{'=' * 60}\n")
            f.write(output)
            if include_wrapper:
                f.write("\n")

    def info(self, msg: str) -> None:
        """Log an INFO level message."""
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        """Log a DEBUG level message."""
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        """Log a WARNING level message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an ERROR level message."""
        self.logger.error(msg)

    def exception(self, msg: str) -> None:
        """Log an ERROR level message with exception info."""
        self.logger.exception(msg)
