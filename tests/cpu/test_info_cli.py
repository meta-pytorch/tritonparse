# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for info CLI functionality."""

import sys
import unittest
from io import StringIO

from tests.test_utils import get_test_ndjson_file
from tritonparse.info.cli import info_command


class TestInfoCLI(unittest.TestCase):
    """Tests for info command line interface."""

    def test_info_list_kernels(self):
        """Integration test: info command lists all kernels."""
        gz_file = get_test_ndjson_file()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            info_command(str(gz_file), kernel_name=None)
            output = captured_output.getvalue()
            self.assertIn("Kernels in", output)
            self.assertIn("launches", output)
        finally:
            sys.stdout = old_stdout

    def test_info_kernel_launches(self):
        """Integration test: info command lists launches for specific kernel."""
        gz_file = get_test_ndjson_file()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            info_command(str(gz_file), kernel_name="fused_op_kernel")
            output = captured_output.getvalue()
            self.assertIn("Launches for 'fused_op_kernel'", output)
            self.assertIn("id=", output)
            self.assertIn("line", output)
        finally:
            sys.stdout = old_stdout

    def test_info_kernel_not_found(self):
        """Integration test: info command handles kernel not found."""
        gz_file = get_test_ndjson_file()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            with self.assertRaises(ValueError):
                info_command(str(gz_file), kernel_name="nonexistent_kernel")
            output = captured_output.getvalue()
            self.assertIn("not found", output)
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    unittest.main()
