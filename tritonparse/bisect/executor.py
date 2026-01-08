# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Shell command executor for bisect operations.

Provides a unified interface for executing shell commands with:
- Blocking mode (run_command): for short commands
- Timeout support
- Environment variable handling
- Integrated logging
"""

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from tritonparse.bisect.logger import BisectLogger


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


@dataclass
class CommandResult:
    """
    Result of a shell command execution.

    Attributes:
        command: The command that was executed (as a string).
        exit_code: The exit code returned by the command.
        stdout: Standard output from the command.
        stderr: Standard error output from the command.
        duration_seconds: Time taken to execute the command in seconds.
    """

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def success(self) -> bool:
        """Check if the command executed successfully (exit code 0)."""
        return self.exit_code == 0

    @property
    def output(self) -> str:
        """Get combined stdout and stderr output."""
        return self.stdout + self.stderr

    @property
    def duration_formatted(self) -> str:
        """Get duration in human-readable format."""
        return _format_duration(self.duration_seconds)


class ShellExecutor:
    """
    Shell command executor with logging integration.

    Provides execution mode:
    - run_command(): Blocking mode for short commands (e.g., git status)

    Example:
        >>> logger = BisectLogger("./logs")
        >>> executor = ShellExecutor(logger)
        >>> result = executor.run_command(["git", "status"])
        >>> if result.success:
        ...     print(result.stdout)
    """

    def __init__(self, logger: BisectLogger) -> None:
        """
        Initialize the shell executor.

        Args:
            logger: BisectLogger instance for logging command execution.
        """
        self.logger = logger

    def run_command(
        self,
        cmd: Union[str, List[str]],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        shell: bool = False,
    ) -> CommandResult:
        """
        Execute a shell command in blocking mode.

        Use this for short commands where you need the complete output.

        Args:
            cmd: Command to execute. Can be a string or list of arguments.
            cwd: Working directory for command execution.
            env: Additional environment variables (merged with current env).
            timeout: Maximum time in seconds to wait for completion.
            shell: If True, execute command through the shell.

        Returns:
            CommandResult containing exit code, stdout, stderr, and duration.
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.logger.debug(f"Executing: {cmd_str}")
        if cwd:
            self.logger.debug(f"  cwd: {cwd}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=full_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=shell,
            )

            duration = time.time() - start_time
            cmd_result = CommandResult(
                command=cmd_str,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
            )

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            stdout = e.stdout if e.stdout else ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")

            cmd_result = CommandResult(
                command=cmd_str,
                exit_code=-1,
                stdout=stdout,
                stderr=f"Command timed out after {timeout}s",
                duration_seconds=duration,
            )

        except OSError as e:
            duration = time.time() - start_time
            cmd_result = CommandResult(
                command=cmd_str,
                exit_code=-1,
                stdout="",
                stderr=f"OSError: {e}",
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            cmd_result = CommandResult(
                command=cmd_str,
                exit_code=-1,
                stdout="",
                stderr=f"Unexpected error: {e}",
                duration_seconds=duration,
            )

        # Log output and summary
        self.logger.log_command_output(cmd_str, cmd_result.output, cmd_result.exit_code)
        self.logger.info(
            f"Command completed in {cmd_result.duration_formatted} "
            f"(exit code: {cmd_result.exit_code})"
        )

        return cmd_result
