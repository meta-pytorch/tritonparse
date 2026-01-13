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
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

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

    def run_command_streaming(
        self,
        cmd: Union[str, List[str]],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = False,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> CommandResult:
        """
        Execute a shell command with real-time streaming output.

        Use this for long-running commands like builds. Each line of output
        is sent to the logger's output_callback for TUI display.

        Args:
            cmd: Command to execute. Can be a string or list of arguments.
            cwd: Working directory for command execution.
            env: Additional environment variables (merged with current env).
            shell: If True, execute command through the shell.
            output_callback: Optional callback called for each output line.
                            Used by TUI to display real-time output.

        Returns:
            CommandResult containing exit code, stdout, and duration.
            Note: stderr is merged into stdout in streaming mode.
        """
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.logger.info(f"Executing (streaming): {cmd_str}")
        if cwd:
            self.logger.debug(f"  cwd: {cwd}")

        start_time = time.time()
        output_lines: List[str] = []

        # Write header to command log file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.logger.command_log_path, "a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[{timestamp}] Command: {cmd_str}\n")
            f.write(f"{'=' * 60}\n")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=full_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=shell,
                bufsize=1,
            )

            for line in process.stdout:
                line = line.rstrip("\n")
                output_lines.append(line)
                self.logger.log_command_output(
                    cmd_str, line + "\n", 0, include_wrapper=False
                )
                # Call output callback for TUI
                if output_callback:
                    output_callback(line)

            process.wait()
            exit_code = process.returncode

        except OSError as e:
            exit_code = -1
            error_msg = f"OSError: {e}"
            output_lines.append(error_msg)
            self.logger.log_command_output(
                cmd_str, error_msg + "\n", exit_code, include_wrapper=False
            )
            if output_callback:
                output_callback(error_msg)

        except Exception as e:
            exit_code = -1
            error_msg = f"Unexpected error: {e}"
            output_lines.append(error_msg)
            self.logger.log_command_output(
                cmd_str, error_msg + "\n", exit_code, include_wrapper=False
            )
            if output_callback:
                output_callback(error_msg)

        duration = time.time() - start_time

        # Write footer to command log file
        with open(self.logger.command_log_path, "a") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Exit code: {exit_code}, Duration: {_format_duration(duration)}\n")

        self.logger.info(
            f"Command completed in {_format_duration(duration)} "
            f"(exit code: {exit_code})"
        )

        return CommandResult(
            command=cmd_str,
            exit_code=exit_code,
            stdout="\n".join(output_lines),
            stderr="",
            duration_seconds=duration,
        )

    def run_git_bisect_sequence(
        self,
        repo_path: str,
        good_commit: str,
        bad_commit: str,
        run_script: Union[str, Path],
        env: Optional[Dict[str, str]] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> CommandResult:
        """
        Execute a complete git bisect sequence.

        This method runs the full git bisect workflow:
        1. git bisect start
        2. git bisect good <good_commit>
        3. git bisect bad <bad_commit>
        4. git bisect run bash <run_script>
        5. git bisect reset (always, even on failure)

        Note: Git bisect state is persisted in .git/ directory, so multiple
        subprocess calls work correctly.

        Args:
            repo_path: Path to the git repository (also used as cwd).
            good_commit: Known good commit hash or tag.
            bad_commit: Known bad commit hash or tag.
            run_script: Path to the script to run for each bisect step.
            env: Additional environment variables for the run script.
            output_callback: Optional callback called for each output line.
                            Used by TUI to display real-time output.

        Returns:
            CommandResult from the 'git bisect run' command, containing
            the bisect output with the culprit commit information.

        Example:
            >>> result = executor.run_git_bisect_sequence(
            ...     repo_path="/path/to/repo",
            ...     good_commit="v1.0.0",
            ...     bad_commit="HEAD",
            ...     run_script="/path/to/test.sh",
            ...     env={"TEST_SCRIPT": "/path/to/test.py"},
            ... )
            >>> if result.success:
            ...     # Parse culprit from result.stdout
            ...     print(result.stdout)
        """
        self.logger.info(f"Starting git bisect: {good_commit} -> {bad_commit}")
        self.logger.info(f"  Repository: {repo_path}")
        self.logger.info(f"  Run script: {run_script}")

        try:
            # Step 1: git bisect start
            self.logger.info("Step 1/4: git bisect start")
            result = self.run_command(
                ["git", "bisect", "start"],
                cwd=repo_path,
            )
            if not result.success:
                self.logger.error(f"git bisect start failed: {result.stderr}")
                return result

            # Step 2: git bisect good
            self.logger.info(f"Step 2/4: git bisect good {good_commit}")
            result = self.run_command(
                ["git", "bisect", "good", good_commit],
                cwd=repo_path,
            )
            if not result.success:
                self.logger.error(f"git bisect good failed: {result.stderr}")
                self._bisect_reset(repo_path)
                return result
            # Pass output to callback for TUI parsing (e.g., "roughly N steps")
            if output_callback:
                for line in result.output.split("\n"):
                    if line.strip():
                        output_callback(line)

            # Step 3: git bisect bad
            self.logger.info(f"Step 3/4: git bisect bad {bad_commit}")
            result = self.run_command(
                ["git", "bisect", "bad", bad_commit],
                cwd=repo_path,
            )
            if not result.success:
                self.logger.error(f"git bisect bad failed: {result.stderr}")
                self._bisect_reset(repo_path)
                return result
            # Pass output to callback for TUI parsing (e.g., "roughly N steps")
            if output_callback:
                for line in result.output.split("\n"):
                    if line.strip():
                        output_callback(line)

            # Step 4: git bisect run (streaming for long-running builds)
            self.logger.info(f"Step 4/4: git bisect run bash {run_script}")
            result = self.run_command_streaming(
                ["git", "bisect", "run", "bash", str(run_script)],
                cwd=repo_path,
                env=env,
                output_callback=output_callback,
            )

            return result

        finally:
            # Always reset bisect state
            self._bisect_reset(repo_path)

    def _bisect_reset(self, repo_path: str) -> None:
        """
        Reset git bisect state.

        This is called automatically after run_git_bisect_sequence completes
        (whether successful or not).

        Args:
            repo_path: Path to the git repository.
        """
        self.logger.debug("Resetting git bisect state")
        self.run_command(
            ["git", "bisect", "reset"],
            cwd=repo_path,
        )
