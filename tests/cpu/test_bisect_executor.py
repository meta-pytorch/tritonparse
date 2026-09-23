# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for bisect executor module (CPU-only, no GPU required)."""

import io
import os
import select
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tritonparse.bisect import BisectLogger, CommandResult, ShellExecutor
from tritonparse.bisect.executor import _format_duration, ProcessCleanupError


class FormatDurationTest(unittest.TestCase):
    """Tests for _format_duration() helper function."""

    def test_seconds_format(self):
        """Test formatting when duration is less than 60 seconds."""
        self.assertEqual(_format_duration(0.0), "0.0s")
        self.assertEqual(_format_duration(30.5), "30.5s")
        self.assertEqual(_format_duration(59.9), "59.9s")

    def test_minutes_format(self):
        """Test formatting when duration is between 60 and 3600 seconds."""
        self.assertEqual(_format_duration(60.0), "1m 0.0s")
        self.assertEqual(_format_duration(90.5), "1m 30.5s")
        self.assertEqual(_format_duration(3599.9), "59m 59.9s")

    def test_hours_format(self):
        """Test formatting when duration is 3600 seconds or more."""
        self.assertEqual(_format_duration(3600.0), "1h 0m 0.0s")
        self.assertEqual(_format_duration(3661.5), "1h 1m 1.5s")
        self.assertEqual(_format_duration(7325.0), "2h 2m 5.0s")


class CommandResultTest(unittest.TestCase):
    """Tests for CommandResult dataclass."""

    def test_success_property(self):
        """Test success property returns True only when exit_code is 0."""
        self.assertTrue(CommandResult("cmd", 0, "", "", 1.0).success)
        self.assertFalse(CommandResult("cmd", 1, "", "", 1.0).success)
        self.assertFalse(CommandResult("cmd", -1, "", "", 1.0).success)

    def test_output_combines_stdout_stderr(self):
        """Test output property combines stdout and stderr."""
        result = CommandResult("cmd", 0, "stdout_text", "stderr_text", 1.0)
        self.assertEqual(result.output, "stdout_textstderr_text")

    def test_duration_formatted(self):
        """Test duration_formatted property uses _format_duration."""
        result = CommandResult("cmd", 0, "", "", 90.5)
        self.assertEqual(result.duration_formatted, "1m 30.5s")


class ShellExecutorTest(unittest.TestCase):
    """Tests for ShellExecutor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = BisectLogger(self.temp_dir)
        self.executor = ShellExecutor(self.logger)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_run_command_success(self):
        """Test run_command executes command and returns result."""
        result = self.executor.run_command(["echo", "hello"])
        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.stdout)

    @patch("subprocess.run")
    def test_run_command_timeout(self, mock_run):
        """Test run_command handles timeout correctly."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=1)
        result = self.executor.run_command(["sleep", "10"], timeout=1)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("timed out", result.stderr)

    def test_run_command_streaming_success(self):
        """Test run_command_streaming executes and streams output."""
        lines = []
        result = self.executor.run_command_streaming(
            ["echo", "hello"],
            output_callback=lines.append,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.exit_code, 0)
        self.assertGreater(len(lines), 0)

    def test_run_command_streaming_callback_receives_lines(self):
        """Test run_command_streaming calls callback for each line."""
        lines = []
        # Use printf to output multiple lines
        self.executor.run_command_streaming(
            'printf "line1\nline2\nline3"',
            shell=True,
            output_callback=lines.append,
        )
        self.assertEqual(len(lines), 3)
        self.assertIn("line1", lines)
        self.assertIn("line2", lines)
        self.assertIn("line3", lines)

    def test_callback_failure_stops_the_child_process(self):
        child_pids = []

        def fail_on_child(line):
            if line.startswith("child="):
                child_pids.append(int(line.split("=")[1]))
                raise RuntimeError("callback failed")

        result = self.executor.run_command_streaming(
            ["bash", "-c", 'printf "child=%s\\n" "$$"\nsleep 30'],
            output_callback=fail_on_child,
        )
        self.assertFalse(result.success)
        self.assertIn("callback failed", result.output)
        self.assertEqual(len(child_pids), 1)
        with self.assertRaises(ProcessLookupError):
            os.kill(child_pids[0], 0)

    def test_cleanup_stops_children_that_ignore_sigterm(self):
        pipe_path = Path(self.temp_dir) / "child.pipe"
        os.mkfifo(pipe_path)
        reader = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
        groups = []

        def fail_when_ready(line):
            if line.startswith("group="):
                groups.append(int(line.split("=")[1]))
            elif line == "ready":
                raise RuntimeError("callback failed")

        try:
            result = self.executor.run_command_streaming(
                [
                    "bash",
                    "-c",
                    'printf "group=%s\\n" "$$"\n'
                    '(trap "" TERM\n'
                    'exec 3>"$CHILD_PIPE"\n'
                    'printf "ready\\n"\n'
                    "sleep 30) &\n"
                    "wait",
                ],
                env={"CHILD_PIPE": str(pipe_path)},
                output_callback=fail_when_ready,
            )
            self.assertFalse(result.success)
            self.assertEqual(len(groups), 1)
            # EOF proves every writer in the child group has closed its fd.
            # Merely reaping the parent leaves the pipe open in its children.
            ready, _, _ = select.select([reader], [], [], 2)
            self.assertEqual(ready, [reader], "A child survived bisect cleanup")
            self.assertEqual(os.read(reader, 1), b"")
        finally:
            for group in groups:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os.close(reader)

    def test_cleanup_waits_are_bounded_when_sigkill_cannot_stop_the_process(self):
        process = MagicMock()
        process.pid = 12345
        process.returncode = None
        process.stdout = io.StringIO("ready\n")

        def stuck_wait(timeout=None):
            self.assertIsNotNone(timeout, "Cleanup used an unbounded wait")
            raise subprocess.TimeoutExpired("stuck process", timeout)

        def fail_when_ready(line):
            if line == "ready":
                raise RuntimeError("callback failed")

        process.wait.side_effect = stuck_wait
        with (
            patch("subprocess.Popen", return_value=process),
            patch("os.killpg") as killpg,
            patch("tritonparse.bisect.executor.time.sleep"),
            self.assertRaisesRegex(RuntimeError, "did not stop after SIGKILL"),
        ):
            self.executor.run_command_streaming(
                ["test-command"], output_callback=fail_when_ready
            )
        self.assertTrue(process.stdout.closed)
        self.assertEqual(
            [call.args[1] for call in killpg.call_args_list],
            [signal.SIGTERM, signal.SIGKILL],
        )

    def test_group_signals_finish_before_the_leader_pid_is_released(self):
        process = MagicMock()
        process.pid = 12345
        process.returncode = None
        process.stdout = io.StringIO("ready\n")
        reaped = False
        signals = []

        def reap(timeout):
            nonlocal reaped
            reaped = True
            process.returncode = 0
            return 0

        def signal_owned_group(group, signum):
            self.assertFalse(reaped, "Signalled a group after releasing its leader PID")
            signals.append(signum)

        def fail_when_ready(line):
            if line == "ready":
                raise RuntimeError("callback failed")

        process.wait.side_effect = reap
        with (
            patch("subprocess.Popen", return_value=process),
            patch("os.killpg", side_effect=signal_owned_group),
            patch("tritonparse.bisect.executor.time.sleep"),
        ):
            result = self.executor.run_command_streaming(
                ["test-command"], output_callback=fail_when_ready
            )
        self.assertFalse(result.success)
        self.assertTrue(reaped)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])

    def test_cleanup_failure_keeps_interrupt_output_and_command_footer(self):
        process = MagicMock()
        process.pid = 12345
        process.returncode = None
        process.stdout = io.StringIO("build evidence\nready\n")

        def interrupt(line):
            if line == "ready":
                raise KeyboardInterrupt()

        with (
            patch("subprocess.Popen", return_value=process),
            patch.object(
                self.executor,
                "_stop_process_group",
                side_effect=ProcessCleanupError("cleanup unavailable"),
            ),
            self.assertRaises(ProcessCleanupError) as error,
        ):
            self.executor.run_command_streaming(
                ["test-command"], output_callback=interrupt
            )
        self.assertIsInstance(error.exception.__cause__, KeyboardInterrupt)
        result = error.exception.result
        self.assertEqual(result.exit_code, 130)
        self.assertIn("build evidence", result.output)
        self.assertIn("Command interrupted", result.output)
        self.assertIn("cleanup unavailable", result.output)
        self.assertTrue(process.stdout.closed)
        commands = Path(self.logger.command_log_path).read_text()
        self.assertIn("build evidence", commands)
        self.assertIn("cleanup unavailable", commands)
        self.assertIn("Exit code: 130, Duration:", commands)

    def test_cleanup_refuses_to_signal_an_already_reaped_group_leader(self):
        process = MagicMock()
        process.pid = 12345
        process.returncode = 0
        process.stdout = io.StringIO("ready\n")

        def fail_when_ready(line):
            if line == "ready":
                raise RuntimeError("callback failed")

        with (
            patch("subprocess.Popen", return_value=process),
            patch("os.killpg") as killpg,
            self.assertRaisesRegex(RuntimeError, "already reaped"),
        ):
            self.executor.run_command_streaming(
                ["test-command"], output_callback=fail_when_ready
            )
        killpg.assert_not_called()
        self.assertTrue(process.stdout.closed)


if __name__ == "__main__":
    unittest.main()
