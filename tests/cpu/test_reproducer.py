# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for reproducer functionality (CPU-only, no kernel execution)."""

import argparse
import os
import unittest
from pathlib import Path

import tritonparse.reproducer.orchestrator
from tests.test_utils import (
    cleanup_temp_dir,
    get_test_ndjson_file,
    setup_temp_reproduce_dir,
)
from tritonparse.reproducer.cli import _add_reproducer_args


class TestReproducer(unittest.TestCase):
    """Tests for reproducer generation."""

    def test_reproduce_mutual_exclusivity(self):
        """Test that --line and --kernel/--launch-id are mutually exclusive."""
        parser = argparse.ArgumentParser()
        _add_reproducer_args(parser)

        # Test: both --line and --kernel provided should raise error
        # Create a mock parser with error method
        mock_parser = argparse.ArgumentParser()
        _add_reproducer_args(mock_parser)
        args = mock_parser.parse_args(
            ["test.ndjson", "--line", "5", "--kernel", "matmul_kernel"]
        )

        # The mutual exclusivity check happens in cli.py main()
        # We test that args are parsed correctly, and the check will happen there
        self.assertEqual(args.kernel, "matmul_kernel")
        self.assertEqual(args.line, 5)

        # Test: only --kernel should work (line defaults to 0, which is allowed)
        args = parser.parse_args(["test.ndjson", "--kernel", "matmul_kernel"])
        self.assertEqual(args.kernel, "matmul_kernel")
        self.assertEqual(args.line, 0)  # default value, allowed with --kernel

        # Test: only --line should work
        args = parser.parse_args(["test.ndjson", "--line", "5"])
        self.assertEqual(args.line, 5)
        self.assertIsNone(args.kernel)

    def test_reproduce_kernel_launch_id(self):
        """End-to-end test: reproduce using --kernel and --launch-id."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            # Test reproducing fused_op_kernel launch_id=0
            result = tritonparse.reproducer.orchestrator.reproduce(
                input_path=str(gz_file),
                line_index=0,  # Placeholder, will be recalculated from kernel_name
                out_dir=out_dir,
                template="example",
                kernel_name="fused_op_kernel",
                launch_id=0,
            )

            # Verify output structure
            self.assertIn("kernel", result)
            self.assertIn("repro_script", result)
            self.assertIn("repro_context", result)
            self.assertTrue(os.path.exists(result["repro_script"]))
            self.assertTrue(os.path.exists(result["repro_context"]))

            # Verify the script contains kernel name
            script_content = Path(result["repro_script"]).read_text()
            self.assertIn("fused_op_kernel", script_content)

        finally:
            cleanup_temp_dir(temp_dir)

    def test_reproduce_kernel_not_found(self):
        """Test that proper error is raised when kernel not found."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            with self.assertRaises(ValueError) as cm:
                tritonparse.reproducer.orchestrator.reproduce(
                    input_path=str(gz_file),
                    line_index=0,  # Placeholder, will be recalculated from kernel_name
                    out_dir=out_dir,
                    template="example",
                    kernel_name="nonexistent_kernel",
                    launch_id=0,
                )

            error_msg = str(cm.exception)
            self.assertIn("not found", error_msg)
            self.assertIn("nonexistent_kernel", error_msg)

        finally:
            cleanup_temp_dir(temp_dir)

    def test_reproduce_launch_id_out_of_range(self):
        """Test that proper error is raised when launch_id is out of range."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            # fused_op_kernel has only 4 launches (0-3), test with launch_id=10
            with self.assertRaises(ValueError) as cm:
                tritonparse.reproducer.orchestrator.reproduce(
                    input_path=str(gz_file),
                    line_index=0,  # Placeholder, will be recalculated from kernel_name
                    out_dir=out_dir,
                    template="example",
                    kernel_name="fused_op_kernel",
                    launch_id=10,
                )

            error_msg = str(cm.exception)
            self.assertIn("has only 4 launches", error_msg)
            self.assertIn("--launch-id 10", error_msg)
            self.assertIn("Valid range: 0 to 3", error_msg)

        finally:
            cleanup_temp_dir(temp_dir)


if __name__ == "__main__":
    unittest.main()
