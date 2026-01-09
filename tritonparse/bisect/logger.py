# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Dual logging system for bisect operations.

Provides separate logging for:
- Module logs: Python logging -> stdout + file
- Command logs: subprocess output -> file (+ optional callback for TUI)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


class BisectLogger:
    """
    Dual logging system for bisect operations.

    This logger provides two separate logging streams:
    1. Module logs: Standard Python logging output to both stdout and a log file
    2. Command logs: Subprocess command output written to a separate file,
       with optional callback support for TUI integration

    Example:
        >>> logger = BisectLogger("./logs")
        >>> logger.info("Starting bisect...")
        >>> logger.log_command_output("git status", "output...", 0)

        # With TUI callback:
        >>> def on_output(line):
        ...     ui.append_output(line)
        >>> logger = BisectLogger("./logs", output_callback=on_output)
    """

    def __init__(
        self,
        log_dir: str,
        session_name: Optional[str] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize the dual logging system.

        Args:
            log_dir: Directory path where log files will be stored.
            session_name: Optional session identifier. If not provided,
                         a timestamp will be used (format: YYYYMMDD_HHMMSS).
            output_callback: Optional callback function that receives each line
                            of command output. Used for TUI integration.
                            Signature: callback(line: str) -> None
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name
        self.output_callback = output_callback
        self._tui_callback: Optional[Callable[[str], None]] = None
        self._stdout_handler: Optional[logging.Handler] = None

        # Module log: stdout + file
        self.module_log_path = self.log_dir / f"{session_name}_bisect.log"
        self._setup_module_logger()

        # Command log: file only
        self.command_log_path = self.log_dir / f"{session_name}_bisect_commands.log"

        # Print log file locations
        self.info(f"Log directory: {self.log_dir}")
        self.info(f"  Module log: {self.module_log_path.name}")
        self.info(f"  Command log: {self.command_log_path.name}")

    def set_output_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """
        Set or update the output callback for command logs.

        This allows setting the callback after initialization, useful when
        the TUI is created after the logger.

        Args:
            callback: Callback function that receives each line of command output,
                     or None to disable the callback.
        """
        self.output_callback = callback

    def _setup_module_logger(self) -> None:
        """Configure the Python logging system with file and stdout handlers."""
        # Use unique logger name with instance id to avoid sharing handlers
        # between different BisectLogger instances with the same session_name.
        # This ensures each instance logs to its own file path.
        self._logger_name = f"bisect.{self.session_name}.{id(self)}"
        self.logger = logging.getLogger(self._logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Prevent propagation to root logger to avoid duplicate output
        self.logger.propagate = False

        # File handler - captures all levels
        fh = logging.FileHandler(self.module_log_path)
        fh.setLevel(logging.DEBUG)

        # Stdout handler - INFO and above only
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        self._stdout_handler = sh

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

        Writes to the command log file, and if an output_callback is set,
        also sends each line to the callback (for TUI display).

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

        # Send to callback if set (for TUI)
        if self.output_callback is not None:
            for line in output.split("\n"):
                self.output_callback(line)

    def configure_for_tui(self, callback: Callable[[str], None]) -> None:
        """
        Configure logger for TUI mode.

        In TUI mode:
        - Stdout handler is disabled (Rich screen takes over the terminal)
        - Log messages are sent to the TUI callback for display
        - File logging continues as normal

        Args:
            callback: Callback function that receives log messages for TUI display.
        """
        self._tui_callback = callback

        # Remove stdout handler to avoid interference with Rich TUI
        if self._stdout_handler is not None:
            self.logger.removeHandler(self._stdout_handler)
            self._stdout_handler = None
        else:
            # Find and remove the StreamHandler
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    self.logger.removeHandler(handler)

    def info(self, msg: str) -> None:
        """Log an INFO level message."""
        self.logger.info(msg)
        if self._tui_callback:
            self._tui_callback(f"[INFO] {msg}")

    def debug(self, msg: str) -> None:
        """Log a DEBUG level message."""
        self.logger.debug(msg)
        # Debug messages are not sent to TUI (too verbose)

    def warning(self, msg: str) -> None:
        """Log a WARNING level message."""
        self.logger.warning(msg)
        if self._tui_callback:
            self._tui_callback(f"[WARNING] {msg}")

    def error(self, msg: str) -> None:
        """Log an ERROR level message."""
        self.logger.error(msg)
        if self._tui_callback:
            self._tui_callback(f"[ERROR] {msg}")

    def exception(self, msg: str) -> None:
        """Log an ERROR level message with exception info."""
        self.logger.exception(msg)
