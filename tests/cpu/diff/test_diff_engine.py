#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the DiffEngine core integration."""

import json
import unittest

from tritonparse.diff.core.diff_engine import DiffEngine
from tritonparse.diff.output.event_writer import create_diff_event

from .test_fixtures import COMP_EVENT_A, COMP_EVENT_B, create_compilation_event


class TestDiffEngine(unittest.TestCase):
    """Tests for the main DiffEngine integration."""

    def test_full_diff_workflow(self) -> None:
        """Test complete diff workflow produces valid result."""
        engine = DiffEngine(
            COMP_EVENT_A,
            COMP_EVENT_B,
            source_trace_a="/path/a.ndjson",
            source_trace_b="/path/b.ndjson",
            event_index_a=0,
            event_index_b=1,
        )
        result = engine.run()

        # Check identifiers
        self.assertNotEqual(result.diff_id, "")
        self.assertEqual(result.kernel_name_a, "add_kernel")
        self.assertTrue(result.kernel_names_identical)
        self.assertEqual(result.event_index_a, 0)
        self.assertEqual(result.event_index_b, 1)

        # Check metadata diff detected
        self.assertIn("num_stages", result.metadata_diff.diffs)
        self.assertIn("num_warps", result.metadata_diff.diffs)

        # Check Python source identical
        self.assertEqual(result.python_source_diff.status, "identical")

        # Check IR stats computed
        self.assertIn("ttir", result.ir_stats)
        self.assertNotEqual(result.ir_stats["ttir"].line_diff, 0)

        # Check by_python_line computed
        self.assertGreater(len(result.by_python_line), 0)

        # Check summary generated
        self.assertIn(
            result.summary.status, ["identical", "minor_diff", "significant_diff"]
        )
        self.assertGreater(len(result.summary.highlights), 0)

    def test_different_python_sources_skips_line_comparison(self) -> None:
        """Test that different Python sources skip by_python_line."""
        comp_different = create_compilation_event(python_source="def different(): pass")
        engine = DiffEngine(COMP_EVENT_A, comp_different)
        result = engine.run()

        self.assertEqual(result.python_source_diff.status, "different")
        self.assertEqual(result.by_python_line, {})
        self.assertIsNotNone(result.summary.warning)

    def test_identical_events(self) -> None:
        """Test that identical events produce identical status."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_A)
        result = engine.run()

        self.assertEqual(result.summary.status, "identical")
        self.assertEqual(result.metadata_diff.diffs, {})

    def test_event_serialization(self) -> None:
        """Test that result can be serialized to JSON."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B)
        result = engine.run()
        diff_event = create_diff_event(result)

        # Should serialize without error
        json_str = json.dumps(diff_event)
        self.assertIn("compilation_diff", json_str)

        # by_python_line keys should be strings
        self.assertIn("by_python_line", diff_event)
        for key in diff_event["by_python_line"].keys():
            self.assertIsInstance(key, str)

    def test_summary_includes_dtype_mismatch_note(self) -> None:
        """Summary includes a correctness note when dtype mismatches detected."""
        from .test_fixtures import create_launch_event

        launch_a = create_launch_event(
            extracted_args={
                "a_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.bfloat16"},
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.float16"},
            }
        )
        engine = DiffEngine(
            COMP_EVENT_A,
            COMP_EVENT_B,
            launch_a=launch_a,
            launch_b=launch_b,
            tensor_values=True,
        )
        result = engine.run()
        correctness_notes = [
            n for n in result.summary.notes if n.category == "correctness"
        ]
        self.assertEqual(len(correctness_notes), 1)
        self.assertIn("dtype mismatch", correctness_notes[0].content.lower())


if __name__ == "__main__":
    unittest.main()
