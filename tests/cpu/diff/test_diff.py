#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Essential tests for the diff module.

Tests cover:
1. DiffEngine integration (full workflow)
2. CLI commands (single-file, dual-file, --list)
3. Key edge cases (missing data, different sources)
"""

import json
import os
import tempfile
import unittest
from io import StringIO
from typing import Any
from unittest.mock import patch

from tritonparse.diff.cli import _parse_event_indices, diff_command
from tritonparse.diff.core.diff_engine import DiffEngine
from tritonparse.diff.core.diff_types import TensorArgDiff, TensorValueDiff
from tritonparse.diff.core.event_matcher import find_launches_for_compilation
from tritonparse.diff.core.tensor_value_analyzer import TensorValueAnalyzer
from tritonparse.diff.output.event_writer import create_diff_event
from tritonparse.diff.output.summary_formatter import format_summary


# --- Test Fixtures ---

DEFAULT_PYTHON_SOURCE = """\
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""

DEFAULT_TTIR = """\
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
    %4 = tt.store %2, %3 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
"""

LONGER_TTIR = """\
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
    %4 = arith.addf %3, %3 : tensor<1024xf32>
    %5 = arith.mulf %4, %3 : tensor<1024xf32>
    %6 = tt.store %2, %5 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
"""


def create_compilation_event(
    kernel_name: str = "add_kernel",
    kernel_hash: str = "abc123def456",
    num_stages: int = 3,
    num_warps: int = 4,
    occurrence_id: int = 0,
    python_source: str | None = None,
    ttir: str | None = None,
    source_mappings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a sample compilation event for testing."""
    return {
        "event_type": "compilation",
        "kernel_name": kernel_name,
        "kernel_hash": kernel_hash,
        "occurrence_id": occurrence_id,
        "compilation_metadata": {
            "num_stages": num_stages,
            "num_warps": num_warps,
            "num_ctas": 1,
        },
        "payload": {
            "python_source": {
                "content": python_source or DEFAULT_PYTHON_SOURCE,
                "start_line": 1,
            },
            "ttir": ttir or DEFAULT_TTIR,
            "source_mappings": source_mappings
            or {
                "python": {
                    "3": {"ttir_lines": [3, 4]},
                    "7": {"ttir_lines": [5, 6]},
                }
            },
        },
    }


COMP_EVENT_A = create_compilation_event(
    kernel_hash="abc123def456789",
    num_stages=3,
    num_warps=4,
    occurrence_id=0,
)

COMP_EVENT_B = create_compilation_event(
    kernel_hash="xyz789ghi012345",
    num_stages=5,
    num_warps=8,
    occurrence_id=1,
    ttir=LONGER_TTIR,
    source_mappings={
        "python": {
            "3": {"ttir_lines": [3, 4, 5]},
            "7": {"ttir_lines": [5, 6, 7, 8]},
        }
    },
)


# --- DiffEngine Tests ---


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


# --- CLI Tests ---


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
        output = StringIO()
        with patch("sys.stdout", output):
            diff_command(
                input_paths=[self.temp_path],
                list_compilations_flag=True,
                skip_logger=True,
            )

        output_text = output.getvalue()
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


# --- Tensor Value Diff Tests ---


def create_launch_event(
    kernel_hash: str = "abc123def456",
    extracted_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a sample launch event for testing."""
    return {
        "event_type": "launch",
        "compilation_metadata": {"hash": kernel_hash},
        "extracted_args": extracted_args
        or {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
            },
            "y_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
            },
            "n_elements": {"type": "int", "value": 1024},
        },
    }


class TestTensorValueDiff(unittest.TestCase):
    """Tests for tensor value comparison (Level 5)."""

    def test_types_default_construction(self) -> None:
        """Test default construction of tensor value types."""
        arg_diff = TensorArgDiff()
        self.assertEqual(arg_diff.status, "skipped")
        self.assertEqual(arg_diff.metrics, {})

        val_diff = TensorValueDiff()
        self.assertEqual(val_diff.status, "skipped")
        self.assertEqual(val_diff.args_compared, 0)
        self.assertEqual(val_diff.per_arg, {})

    def test_types_serialization(self) -> None:
        """Test tensor value types can be serialized to JSON."""
        from dataclasses import asdict

        val_diff = TensorValueDiff(
            status="divergent",
            args_compared=1,
            args_divergent=1,
            per_arg={
                "x_ptr": TensorArgDiff(
                    arg_name="x_ptr",
                    status="divergent",
                    shape_a=[1024],
                    shape_b=[1024],
                    dtype_a="torch.float32",
                    dtype_b="torch.float32",
                    metrics={"max_abs_error": 0.001, "allclose": False},
                )
            },
        )
        json_str = json.dumps(asdict(val_diff))
        parsed = json.loads(json_str)
        self.assertEqual(parsed["status"], "divergent")
        self.assertIn("x_ptr", parsed["per_arg"])

    def test_compilation_diff_result_has_tensor_value_diff(self) -> None:
        """Test CompilationDiffResult includes tensor_value_diff field."""
        from tritonparse.diff.core.diff_types import CompilationDiffResult

        result = CompilationDiffResult()
        self.assertEqual(result.tensor_value_diff.status, "skipped")

    def test_find_launches_for_compilation(self) -> None:
        """Test finding launch events by compilation hash."""
        events = [
            {"event_type": "compilation", "kernel_hash": "abc"},
            create_launch_event(kernel_hash="abc"),
            create_launch_event(kernel_hash="xyz"),
            create_launch_event(kernel_hash="abc"),
        ]
        launches = find_launches_for_compilation(events, "abc")
        self.assertEqual(len(launches), 2)

        launches = find_launches_for_compilation(events, "xyz")
        self.assertEqual(len(launches), 1)

        launches = find_launches_for_compilation(events, "nonexistent")
        self.assertEqual(len(launches), 0)

        launches = find_launches_for_compilation(events, "")
        self.assertEqual(len(launches), 0)

    def test_analyzer_no_blobs(self) -> None:
        """Test analyzer skips when no blob_path present."""
        launch_a = create_launch_event(kernel_hash="abc")
        launch_b = create_launch_event(kernel_hash="xyz")
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "skipped")
        self.assertIn("TRITONPARSE_SAVE_TENSOR_BLOBS", result.warning)

    def test_analyzer_no_extracted_args(self) -> None:
        """Test analyzer skips when no extracted_args."""
        launch_a = {"event_type": "launch"}
        launch_b = {"event_type": "launch"}
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "skipped")
        self.assertIn("extracted_args", result.warning)

    def test_analyzer_no_common_tensors(self) -> None:
        """Test analyzer skips when no common tensor args."""
        launch_a = create_launch_event(
            extracted_args={"a": {"type": "tensor", "shape": [1]}}
        )
        launch_b = create_launch_event(
            extracted_args={"b": {"type": "tensor", "shape": [1]}}
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "skipped")
        self.assertIn("common tensor", result.warning)

    def test_analyzer_shape_mismatch(self) -> None:
        """Test analyzer detects shape mismatch."""
        launch_a = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/a.bin",
                }
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [2048],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/b.bin",
                }
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.per_arg["x_ptr"].status, "shape_mismatch")

    def test_analyzer_dtype_mismatch(self) -> None:
        """Test analyzer detects dtype mismatch."""
        launch_a = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/a.bin",
                }
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float16",
                    "blob_path": "/tmp/b.bin",
                }
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.per_arg["x_ptr"].status, "dtype_mismatch")

    def test_analyzer_with_mocked_blobs_identical(self) -> None:
        """Test analyzer with identical tensors (mocked load_tensor)."""
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not available")

        tensor = torch.ones(1024)

        launch_a = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/a.bin",
                }
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/b.bin",
                }
            }
        )

        with patch(
            "tritonparse.tools.load_tensor.load_tensor",
            return_value=tensor,
        ):
            result = TensorValueAnalyzer(launch_a, launch_b).analyze()

        self.assertEqual(result.status, "identical")
        self.assertEqual(result.args_compared, 1)
        self.assertEqual(result.args_identical, 1)
        self.assertEqual(result.per_arg["x_ptr"].status, "identical")
        self.assertEqual(result.per_arg["x_ptr"].metrics["max_abs_error"], 0.0)

    def test_analyzer_with_mocked_blobs_divergent(self) -> None:
        """Test analyzer with divergent tensors (mocked load_tensor)."""
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not available")

        tensor_a = torch.zeros(1024)
        tensor_b = torch.ones(1024)

        launch_a = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/a.bin",
                }
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/b.bin",
                }
            }
        )

        with patch(
            "tritonparse.tools.load_tensor.load_tensor",
            side_effect=[tensor_a, tensor_b],
        ):
            result = TensorValueAnalyzer(launch_a, launch_b).analyze()

        self.assertEqual(result.status, "divergent")
        self.assertEqual(result.args_divergent, 1)
        self.assertEqual(result.per_arg["x_ptr"].status, "divergent")
        self.assertEqual(result.per_arg["x_ptr"].metrics["max_abs_error"], 1.0)
        self.assertFalse(result.per_arg["x_ptr"].metrics["allclose"])

    def test_analyzer_with_mocked_blobs_close(self) -> None:
        """Test analyzer with close tensors (within tolerance)."""
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not available")

        tensor_a = torch.ones(1024)
        tensor_b = torch.ones(1024) + 1e-7  # Within default atol=1e-5

        launch_a = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/a.bin",
                }
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/b.bin",
                }
            }
        )

        with patch(
            "tritonparse.tools.load_tensor.load_tensor",
            side_effect=[tensor_a, tensor_b],
        ):
            result = TensorValueAnalyzer(launch_a, launch_b).analyze()

        self.assertEqual(result.status, "close")
        self.assertEqual(result.args_close, 1)
        self.assertTrue(result.per_arg["x_ptr"].metrics["allclose"])

    def test_diff_engine_tensor_values_disabled(self) -> None:
        """Test DiffEngine with tensor_values=False (default)."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B)
        result = engine.run()
        self.assertEqual(result.tensor_value_diff.status, "skipped")

    def test_diff_engine_no_launches(self) -> None:
        """Test DiffEngine with tensor_values=True but no launch events."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B, tensor_values=True)
        result = engine.run()
        self.assertEqual(result.tensor_value_diff.status, "skipped")
        self.assertIn("No launch events", result.tensor_value_diff.warning)

    def test_diff_engine_with_launches(self) -> None:
        """Test DiffEngine with tensor_values=True and launch events."""
        launch_a = create_launch_event(kernel_hash="abc123def456789")
        launch_b = create_launch_event(kernel_hash="xyz789ghi012345")

        engine = DiffEngine(
            COMP_EVENT_A,
            COMP_EVENT_B,
            launch_a=launch_a,
            launch_b=launch_b,
            tensor_values=True,
        )
        result = engine.run()
        # No blobs -> skipped
        self.assertEqual(result.tensor_value_diff.status, "skipped")
        self.assertIn(
            "TRITONPARSE_SAVE_TENSOR_BLOBS",
            result.tensor_value_diff.warning,
        )

    def test_serialization_includes_tensor_value_diff(self) -> None:
        """Test that create_diff_event includes tensor_value_diff."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B)
        result = engine.run()
        diff_event = create_diff_event(result)
        json_str = json.dumps(diff_event)
        self.assertIn("tensor_value_diff", json_str)
        self.assertIn("skipped", diff_event["tensor_value_diff"]["status"])

    def test_format_summary_with_tensor_warning(self) -> None:
        """Test format_summary includes tensor warning when present."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B, tensor_values=True)
        result = engine.run()
        output = format_summary(result)
        self.assertIn("Tensor Values:", output)

    def test_format_summary_with_active_tensor_comparison(self) -> None:
        """Test format_summary formats active tensor comparison."""
        from tritonparse.diff.core.diff_types import CompilationDiffResult

        result = CompilationDiffResult()
        result.tensor_value_diff = TensorValueDiff(
            status="divergent",
            args_compared=2,
            args_identical=1,
            args_divergent=1,
            per_arg={
                "x_ptr": TensorArgDiff(arg_name="x_ptr", status="identical"),
                "y_ptr": TensorArgDiff(
                    arg_name="y_ptr",
                    status="divergent",
                    metrics={
                        "max_abs_error": 0.001,
                        "mean_abs_error": 0.0005,
                        "mismatch_ratio": 0.1,
                    },
                ),
            },
        )
        output = format_summary(result)
        self.assertIn("Tensor Value Comparison", output)
        self.assertIn("1 identical", output)
        self.assertIn("1 divergent", output)
        self.assertIn("[=] x_ptr", output)
        self.assertIn("[!] y_ptr", output)
        self.assertIn("Max Abs Error", output)


if __name__ == "__main__":
    unittest.main()
