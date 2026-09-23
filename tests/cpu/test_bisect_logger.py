# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for bisect logger module (CPU-only, no GPU required)."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tritonparse.bisect import BisectLogger


class BisectLoggerTest(unittest.TestCase):
    """Tests for BisectLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def _make_logger(self, log_dir=None, **kwargs):
        logger = BisectLogger(str(log_dir or self.temp_dir), **kwargs)
        self.addCleanup(self._close_logger, logger)
        return logger

    @staticmethod
    def _close_logger(logger):
        for handler in logger.logger.handlers[:]:
            logger.logger.removeHandler(handler)
            handler.close()

    def test_creates_log_directory_and_files(self):
        """Test that logger creates log directory and module log file."""
        logger = self._make_logger()
        self.assertTrue(logger.module_log_path.exists())
        self.assertTrue(logger.log_dir.exists())

    def test_auto_generates_session_name(self):
        """Test that session name is auto-generated if not provided."""
        logger = self._make_logger()
        self.assertIsNotNone(logger.session_name)
        self.assertIn(logger.session_name, str(logger.module_log_path))

    def test_uses_provided_session_name(self):
        """Test that provided session name is used."""
        logger = self._make_logger(session_name="test_session")
        self.assertEqual(logger.session_name, "test_session")
        self.assertIn("test_session", str(logger.module_log_path))

    def test_output_callback_receives_lines(self):
        """Test that output_callback receives each line of command output."""
        captured = []
        logger = self._make_logger(output_callback=captured.append)
        logger.log_command_output("test_cmd", "line1\nline2\nline3", 0)
        self.assertEqual(captured, ["line1", "line2", "line3"])

    def test_tui_callback_on_info(self):
        """Test that info() triggers TUI callback when configured."""
        tui_msgs = []
        logger = self._make_logger()
        logger.configure_for_tui(tui_msgs.append)
        logger.info("test message")
        self.assertEqual(len(tui_msgs), 1)
        self.assertIn("[INFO] test message", tui_msgs[0])

    def test_tui_callback_not_triggered_by_debug(self):
        """Test that debug() does NOT trigger TUI callback (too verbose)."""
        tui_msgs = []
        logger = self._make_logger()
        logger.configure_for_tui(tui_msgs.append)
        logger.debug("debug message")
        # TUI callback should not be called for debug messages
        self.assertEqual(len(tui_msgs), 0)

    def test_log_command_output_writes_to_file(self):
        """Test that log_command_output writes to command log file."""
        logger = self._make_logger()
        logger.log_command_output("echo hello", "hello world", 0)
        self.assertTrue(logger.command_log_path.exists())
        content = logger.command_log_path.read_text()
        self.assertIn("echo hello", content)
        self.assertIn("hello world", content)

    def test_reused_instance_id_keeps_logs_isolated(self):
        """A replacement logger must not reopen a previous instance's log file."""
        for remove_previous_directory in (False, True):
            with self.subTest(remove_previous_directory=remove_previous_directory):
                previous_dir = Path(self.temp_dir) / f"old_{remove_previous_directory}"
                current_dir = Path(self.temp_dir) / f"new_{remove_previous_directory}"
                session_name = f"reused_id_{remove_previous_directory}"
                # Force ID reuse without depending on allocator timing.
                with patch("tritonparse.bisect.logger.id", return_value=1, create=True):
                    previous = self._make_logger(
                        previous_dir, session_name=session_name
                    )
                    # Reproduce the original cleanup: close without detaching.
                    for handler in previous.logger.handlers:
                        handler.close()
                    if remove_previous_directory:
                        shutil.rmtree(previous_dir)

                    current = self._make_logger(current_dir, session_name=session_name)
                    message = "message from the replacement logger"
                    current.info(message)

                self.assertIn(message, current.module_log_path.read_text())
                if remove_previous_directory:
                    self.assertFalse(previous_dir.exists())
                else:
                    self.assertNotIn(message, previous.module_log_path.read_text())


if __name__ == "__main__":
    unittest.main()
