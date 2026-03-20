#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for diff CLI commands (single-file, dual-file, and trace modes)."""

import json
import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from tritonparse.diff.cli import _parse_event_indices, diff_command

from .test_fixtures import COMP_EVENT_A, COMP_EVENT_B, create_compilation_event


class TestDiffCLI(unittest.TestCase):
    """Tests for diff CLI commands."""

    def setUp(self) -> None:
        """Create temporary ndjson file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ndjson", delete=False
        )
        self.temp_file.write(json.dumps(COMP_EVENT_A) + "\n")
        self.temp_file.write(json.dumps(COMP_EVENT_B) + "\n")
        self.temp_file.close()
        self.temp_path = self.temp_file.name
        self.output_path = self.temp_path.replace(".ndjson", "_diff.ndjson")

    def tearDown(self) -> None:
        """Clean up temporary files."""
        for path in [self.temp_path, self.output_path]:
            if os.path.exists(path):
                os.unlink(path)

    def test_parse_occurrence_ids(self) -> None:
        """Test occurrence ID parsing."""
        self.assertEqual(_parse_event_indices("0,1"), (0, 1))
        self.assertEqual(_parse_event_indices("2, 5"), (2, 5))

        with self.assertRaises(ValueError):
            _parse_event_indices("invalid")

    def test_single_file_diff(self) -> None:
        """Test basic single-file diff."""
        with patch("sys.stdout", new_callable=StringIO):
            diff_command(
                input_paths=[self.temp_path],
                events="0,1",
                skip_logger=True,
            )

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[2])["event_type"], "compilation_diff")

    def test_list_compilations(self) -> None:
        """Test --list mode."""
        with self.assertLogs("tritonparse.diff.cli", level="INFO") as cm:
            diff_command(
                input_paths=[self.temp_path],
                list_compilations_flag=True,
                skip_logger=True,
            )

        output_text = "\n".join(cm.output)
        self.assertIn("Compilations in", output_text)
        self.assertIn("add_kernel", output_text)

    def test_in_place_mode(self) -> None:
        """Test --in-place appends to input file."""
        with patch("sys.stdout", new_callable=StringIO):
            diff_command(
                input_paths=[self.temp_path],
                events="0,1",
                in_place=True,
                skip_logger=True,
            )

        with open(self.temp_path) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[2])["event_type"], "compilation_diff")

    def test_occurrence_id_not_found(self) -> None:
        """Test error on invalid occurrence ID."""
        with self.assertRaises(ValueError) as ctx:
            diff_command(
                input_paths=[self.temp_path],
                events="0,10",
                skip_logger=True,
            )
        self.assertIn("Event index 10 out of range", str(ctx.exception))


class TestDiffCLIDualFile(unittest.TestCase):
    """Tests for dual-file diff mode."""

    def setUp(self) -> None:
        """Create temporary ndjson files for testing."""
        self.temp_file_a = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ndjson", delete=False
        )
        self.temp_file_a.write(json.dumps(COMP_EVENT_A) + "\n")
        self.temp_file_a.close()

        self.temp_file_b = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ndjson", delete=False
        )
        self.temp_file_b.write(json.dumps(COMP_EVENT_B) + "\n")
        self.temp_file_b.close()

        self.temp_path_a = self.temp_file_a.name
        self.temp_path_b = self.temp_file_b.name
        self.output_path = self.temp_path_a.replace(".ndjson", "_diff.ndjson")

    def tearDown(self) -> None:
        """Clean up temporary files."""
        for path in [self.temp_path_a, self.temp_path_b, self.output_path]:
            if os.path.exists(path):
                os.unlink(path)

    def test_dual_file_diff(self) -> None:
        """Test comparing events across two files."""
        with patch("sys.stdout", new_callable=StringIO):
            diff_command(
                input_paths=[self.temp_path_a, self.temp_path_b],
                events="0,0",  # index 0 from file A, index 0 from file B
                skip_logger=True,
            )

        self.assertTrue(os.path.exists(self.output_path))

        with open(self.output_path) as f:
            lines = f.readlines()

        diff_event = json.loads(lines[2])
        self.assertEqual(diff_event["hash_a"], COMP_EVENT_A["kernel_hash"])
        self.assertEqual(diff_event["hash_b"], COMP_EVENT_B["kernel_hash"])


class TestTraceDiffCLI(unittest.TestCase):
    """Tests for trace diff CLI integration."""

    def setUp(self) -> None:
        """Create temporary ndjson files for testing."""
        self.temp_file_a = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ndjson", delete=False
        )
        self.temp_file_a.write(
            json.dumps(create_compilation_event(kernel_name="k1", kernel_hash="h1"))
            + "\n"
        )
        self.temp_file_a.write(
            json.dumps(create_compilation_event(kernel_name="k2", kernel_hash="h2"))
            + "\n"
        )
        self.temp_file_a.close()

        self.temp_file_b = tempfile.NamedTemporaryFile(
            mode="w", suffix=".ndjson", delete=False
        )
        self.temp_file_b.write(
            json.dumps(create_compilation_event(kernel_name="k1", kernel_hash="h1"))
            + "\n"
        )
        self.temp_file_b.write(
            json.dumps(create_compilation_event(kernel_name="k3", kernel_hash="h3"))
            + "\n"
        )
        self.temp_file_b.close()

        self.temp_path_a = self.temp_file_a.name
        self.temp_path_b = self.temp_file_b.name
        self.output_path = self.temp_path_a.replace(".ndjson", "_diff.ndjson")

    def tearDown(self) -> None:
        """Clean up temporary files."""
        for path in [self.temp_path_a, self.temp_path_b, self.output_path]:
            if os.path.exists(path):
                os.unlink(path)

    def test_trace_flag_two_files(self) -> None:
        """--trace with two temp ndjson files produces output."""
        with self.assertLogs("tritonparse.diff.cli", level="INFO") as cm:
            diff_command(
                input_paths=[self.temp_path_a, self.temp_path_b],
                trace=True,
                skip_logger=True,
            )

        output_text = "\n".join(cm.output)
        self.assertIn("Trace Diff Summary", output_text)
        self.assertTrue(os.path.exists(self.output_path))

    def test_trace_flag_one_file_error(self) -> None:
        """--trace with one file raises error."""
        with self.assertRaises(ValueError) as ctx:
            diff_command(
                input_paths=[self.temp_path_a],
                trace=True,
                skip_logger=True,
            )
        self.assertIn("exactly 2 input files", str(ctx.exception))

    def test_trace_in_place_error(self) -> None:
        """--trace --in-place raises error."""
        with self.assertRaises(ValueError) as ctx:
            diff_command(
                input_paths=[self.temp_path_a, self.temp_path_b],
                trace=True,
                in_place=True,
                skip_logger=True,
            )
        self.assertIn("cannot be used together", str(ctx.exception))

    def test_trace_output_file(self) -> None:
        """Output ndjson contains trace_diff event type."""
        with patch("sys.stdout", new_callable=StringIO):
            diff_command(
                input_paths=[self.temp_path_a, self.temp_path_b],
                trace=True,
                skip_logger=True,
            )

        self.assertTrue(os.path.exists(self.output_path))
        with open(self.output_path) as f:
            lines = f.readlines()

        # Should have compilation events + trace_diff event
        event_types = [json.loads(line)["event_type"] for line in lines]
        self.assertIn("trace_diff", event_types)

    def test_trace_diff_basic(self) -> None:
        """Trace diff with two files produces summary output."""
        with self.assertLogs("tritonparse.diff.cli", level="INFO") as cm:
            diff_command(
                input_paths=[self.temp_path_a, self.temp_path_b],
                trace=True,
                skip_logger=True,
            )

        output_text = "\n".join(cm.output)
        self.assertIn("Trace Diff Summary", output_text)


if __name__ == "__main__":
    unittest.main()
