# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Shell command executor for bisect operations.

Provides CommandResult dataclass for representing command execution results.
"""

from dataclasses import dataclass


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
