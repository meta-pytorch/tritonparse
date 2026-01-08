# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for bisect logger module (CPU-only, no GPU required)."""

import shutil
import tempfile
import unittest

from tritonparse.bisect import BisectLogger


class BisectLoggerTest(unittest.TestCase):
    """Tests for BisectLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_creates_log_directory_and_files(self):
        """Test that logger creates log directory and module log file."""
        logger = BisectLogger(self.temp_dir)
        self.assertTrue(logger.module_log_path.exists())
        self.assertTrue(logger.log_dir.exists())

    def test_auto_generates_session_name(self):
        """Test that session name is auto-generated if not provided."""
        logger = BisectLogger(self.temp_dir)
        self.assertIsNotNone(logger.session_name)
        self.assertIn(logger.session_name, str(logger.module_log_path))

    def test_uses_provided_session_name(self):
        """Test that provided session name is used."""
        logger = BisectLogger(self.temp_dir, session_name="test_session")
        self.assertEqual(logger.session_name, "test_session")
        self.assertIn("test_session", str(logger.module_log_path))

    def test_output_callback_receives_lines(self):
        """Test that output_callback receives each line of command output."""
        captured = []
        logger = BisectLogger(self.temp_dir, output_callback=captured.append)
        logger.log_command_output("test_cmd", "line1\nline2\nline3", 0)
        self.assertEqual(captured, ["line1", "line2", "line3"])

    def test_tui_callback_on_info(self):
        """Test that info() triggers TUI callback when configured."""
        tui_msgs = []
        logger = BisectLogger(self.temp_dir)
        logger.configure_for_tui(tui_msgs.append)
        logger.info("test message")
        self.assertEqual(len(tui_msgs), 1)
        self.assertIn("[INFO] test message", tui_msgs[0])

    def test_tui_callback_not_triggered_by_debug(self):
        """Test that debug() does NOT trigger TUI callback (too verbose)."""
        tui_msgs = []
        logger = BisectLogger(self.temp_dir)
        logger.configure_for_tui(tui_msgs.append)
        logger.debug("debug message")
        # TUI callback should not be called for debug messages
        self.assertEqual(len(tui_msgs), 0)

    def test_log_command_output_writes_to_file(self):
        """Test that log_command_output writes to command log file."""
        logger = BisectLogger(self.temp_dir)
        logger.log_command_output("echo hello", "hello world", 0)
        self.assertTrue(logger.command_log_path.exists())
        content = logger.command_log_path.read_text()
        self.assertIn("echo hello", content)
        self.assertIn("hello world", content)


if __name__ == "__main__":
    unittest.main()
