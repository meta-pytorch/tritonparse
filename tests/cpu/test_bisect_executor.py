# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for bisect executor module (CPU-only, no GPU required)."""

import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tritonparse.bisect import BisectLogger, CommandResult, ShellExecutor
from tritonparse.bisect.executor import _format_duration


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


if __name__ == "__main__":
    unittest.main()
