#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for tensor value comparison (Level 5)."""

import json
import unittest
from unittest.mock import patch

from tritonparse.diff.core.diff_engine import DiffEngine
from tritonparse.diff.core.diff_types import TensorArgDiff, TensorValueDiff
from tritonparse.diff.core.event_matcher import find_launches_for_compilation
from tritonparse.diff.core.tensor_value_analyzer import TensorValueAnalyzer
from tritonparse.diff.output.event_writer import create_diff_event
from tritonparse.diff.output.summary_formatter import format_summary

from .test_fixtures import COMP_EVENT_A, COMP_EVENT_B, create_launch_event


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

    def test_dtype_mismatch_fields_default(self) -> None:
        """Test new dtype fields have correct defaults."""
        val_diff = TensorValueDiff()
        self.assertEqual(val_diff.dtype_inventory_a, {})
        self.assertEqual(val_diff.dtype_inventory_b, {})
        self.assertEqual(val_diff.dtype_mismatches, [])

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
        """Test analyzer skips when no blob_path and no inline stats present."""
        launch_a = create_launch_event(kernel_hash="abc")
        launch_b = create_launch_event(kernel_hash="xyz")
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "skipped")
        self.assertIn("TRITONPARSE_MORE_TENSOR_INFORMATION", result.warning)

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
        # No blobs and no inline stats -> skipped
        self.assertEqual(result.tensor_value_diff.status, "skipped")
        self.assertIn(
            "TRITONPARSE_MORE_TENSOR_INFORMATION",
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


class TestTensorValueDiffInlineStats(unittest.TestCase):
    """Tests for tensor value comparison using inline statistics fallback."""

    def test_analyzer_inline_stats_identical(self) -> None:
        """Test analyzer with identical inline stats."""
        args = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.29,
            },
        }
        launch_a = create_launch_event(extracted_args=dict(args))
        launch_b = create_launch_event(extracted_args=dict(args))
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "identical")
        self.assertEqual(result.args_compared, 1)
        self.assertEqual(result.per_arg["x_ptr"].status, "identical")
        self.assertEqual(result.per_arg["x_ptr"].metrics["comparison_mode"], "stats")

    def test_analyzer_inline_stats_close(self) -> None:
        """Test analyzer with close inline stats (within atol)."""
        args_a = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.29,
            },
        }
        args_b = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 1e-7,
                "max": 1.0 + 1e-7,
                "mean": 0.5 + 1e-7,
                "std": 0.29 + 1e-7,
            },
        }
        launch_a = create_launch_event(extracted_args=args_a)
        launch_b = create_launch_event(extracted_args=args_b)
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "close")
        self.assertEqual(result.per_arg["x_ptr"].metrics["comparison_mode"], "stats")

    def test_analyzer_inline_stats_divergent(self) -> None:
        """Test analyzer with divergent inline stats."""
        args_a = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.29,
            },
        }
        args_b = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 10.0,
                "max": 20.0,
                "mean": 15.0,
                "std": 2.9,
            },
        }
        launch_a = create_launch_event(extracted_args=args_a)
        launch_b = create_launch_event(extracted_args=args_b)
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "divergent")
        self.assertEqual(result.per_arg["x_ptr"].status, "divergent")
        self.assertEqual(result.per_arg["x_ptr"].metrics["comparison_mode"], "stats")
        self.assertAlmostEqual(result.per_arg["x_ptr"].metrics["mean_diff"], 14.5)

    def test_analyzer_mixed_blob_and_stats(self) -> None:
        """Test analyzer with some args having blobs and some only stats."""
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not available")

        tensor = torch.ones(1024)
        args_a = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "blob_path": "/tmp/a.bin",
            },
            "y_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.29,
            },
        }
        args_b = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "blob_path": "/tmp/b.bin",
            },
            "y_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.29,
            },
        }
        launch_a = create_launch_event(extracted_args=args_a)
        launch_b = create_launch_event(extracted_args=args_b)

        with patch(
            "tritonparse.tools.load_tensor.load_tensor",
            return_value=tensor,
        ):
            result = TensorValueAnalyzer(launch_a, launch_b).analyze()

        self.assertEqual(result.args_compared, 2)
        self.assertEqual(result.per_arg["x_ptr"].metrics["comparison_mode"], "blob")
        self.assertEqual(result.per_arg["y_ptr"].metrics["comparison_mode"], "stats")

    def test_analyzer_no_blobs_no_stats(self) -> None:
        """Test analyzer skips when neither blobs nor stats are present."""
        launch_a = create_launch_event()
        launch_b = create_launch_event()
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "skipped")
        self.assertIn("TRITONPARSE_MORE_TENSOR_INFORMATION", result.warning)
        self.assertIn("TRITONPARSE_SAVE_TENSOR_BLOBS", result.warning)

    def test_analyzer_partial_stats_skips(self) -> None:
        """Test that tensors with only some stats fields are skipped."""
        args_a = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 1.0,
                # missing mean and std
            },
        }
        args_b = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.29,
            },
        }
        launch_a = create_launch_event(extracted_args=args_a)
        launch_b = create_launch_event(extracted_args=args_b)
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        # No tensor has full stats on both sides and no blobs -> skipped
        self.assertEqual(result.status, "skipped")

    def test_analyzer_inline_stats_zero_values(self) -> None:
        """Test that zero stat values (which are falsy) are handled correctly."""
        args = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
            },
        }
        launch_a = create_launch_event(extracted_args=dict(args))
        launch_b = create_launch_event(extracted_args=dict(args))
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        # Should not skip — 0.0 is a valid stat value
        self.assertEqual(result.status, "identical")
        self.assertEqual(result.per_arg["x_ptr"].metrics["comparison_mode"], "stats")

    def test_format_summary_with_stats_comparison(self) -> None:
        """Test format_summary formats stats-based tensor comparison."""
        from tritonparse.diff.core.diff_types import CompilationDiffResult

        result = CompilationDiffResult()
        result.tensor_value_diff = TensorValueDiff(
            status="divergent",
            args_compared=1,
            args_divergent=1,
            per_arg={
                "x_ptr": TensorArgDiff(
                    arg_name="x_ptr",
                    status="divergent",
                    metrics={
                        "comparison_mode": "stats",
                        "min_a": 0.0,
                        "min_b": 10.0,
                        "min_diff": 10.0,
                        "max_a": 1.0,
                        "max_b": 20.0,
                        "max_diff": 19.0,
                        "mean_a": 0.5,
                        "mean_b": 15.0,
                        "mean_diff": 14.5,
                        "std_a": 0.29,
                        "std_b": 2.9,
                        "std_diff": 2.61,
                        "atol": 1e-5,
                    },
                ),
            },
        )
        output = format_summary(result)
        self.assertIn("(stats)", output)
        self.assertIn("Min", output)
        self.assertIn("Mean", output)


class TestDtypeMismatchDetection(unittest.TestCase):
    """Tests for cross-side dtype mismatch detection."""

    def test_no_common_names_different_dtypes(self) -> None:
        """Dtype mismatch detected when names differ and dtypes differ."""
        launch_a = create_launch_event(
            extracted_args={
                "a_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.bfloat16"},
                "b_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.bfloat16"},
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.float16"},
                "y_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.float16"},
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "divergent")
        self.assertEqual(len(result.dtype_mismatches), 1)
        self.assertIn("torch.bfloat16", result.dtype_mismatches[0].dtypes_a)
        self.assertIn("torch.float16", result.dtype_mismatches[0].dtypes_b)
        self.assertEqual(
            result.dtype_inventory_a,
            {"a_ptr": "torch.bfloat16", "b_ptr": "torch.bfloat16"},
        )
        self.assertEqual(
            result.dtype_inventory_b,
            {"x_ptr": "torch.float16", "y_ptr": "torch.float16"},
        )

    def test_no_common_names_same_dtypes_no_false_alarm(self) -> None:
        """No false alarm when names differ but dtypes are the same."""
        launch_a = create_launch_event(
            extracted_args={
                "a_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.float32"},
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.float32"},
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.dtype_mismatches, [])
        self.assertEqual(result.dtype_inventory_a, {"a_ptr": "torch.float32"})
        self.assertEqual(result.dtype_inventory_b, {"x_ptr": "torch.float32"})

    def test_no_common_names_partial_overlap_no_false_alarm(self) -> None:
        """No false alarm when dtype sets partially overlap."""
        launch_a = create_launch_event(
            extracted_args={
                "a_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.bfloat16"},
                "bias": {"type": "tensor", "shape": [64], "dtype": "torch.float32"},
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {"type": "tensor", "shape": [1024], "dtype": "torch.float16"},
                "scale": {"type": "tensor", "shape": [64], "dtype": "torch.float32"},
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.dtype_mismatches, [])

    def test_inventories_populated_on_happy_path(self) -> None:
        """Dtype inventories populated even when common names exist."""
        args = {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
                "min": 0.0,
                "max": 1.0,
                "mean": 0.5,
                "std": 0.29,
            },
        }
        launch_a = create_launch_event(extracted_args=dict(args))
        launch_b = create_launch_event(extracted_args=dict(args))
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "identical")
        self.assertEqual(result.dtype_inventory_a, {"x_ptr": "torch.float32"})
        self.assertEqual(result.dtype_inventory_b, {"x_ptr": "torch.float32"})

    def test_cross_side_dtype_mismatch_in_highlights(self) -> None:
        """Cross-side dtype mismatches appear in DiffEngine highlights."""
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
        highlights_text = " ".join(result.summary.highlights)
        self.assertIn("DTYPE MISMATCH", highlights_text)
        self.assertIn("torch.bfloat16", highlights_text)
        self.assertIn("torch.float16", highlights_text)

    def test_per_arg_dtype_mismatch_in_highlights(self) -> None:
        """Per-arg dtype_mismatch status appears in DiffEngine highlights."""
        launch_a = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/a.bin",
                },
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float16",
                    "blob_path": "/tmp/b.bin",
                },
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
        highlights_text = " ".join(result.summary.highlights)
        self.assertIn("x_ptr", highlights_text)
        self.assertIn("dtype mismatch", highlights_text.lower())

    def test_format_summary_shows_dtype_mismatch(self) -> None:
        """format_summary displays dtype mismatch prominently."""
        from tritonparse.diff.core.diff_types import (
            CompilationDiffResult,
            DtypeMismatch,
        )

        result = CompilationDiffResult()
        result.tensor_value_diff = TensorValueDiff(
            status="divergent",
            warning="No common tensor arguments found between launch events",
            dtype_inventory_a={"a_ptr": "torch.bfloat16", "b_ptr": "torch.bfloat16"},
            dtype_inventory_b={"x_ptr": "torch.float16", "y_ptr": "torch.float16"},
            dtype_mismatches=[
                DtypeMismatch(
                    dtypes_a=["torch.bfloat16"],
                    dtypes_b=["torch.float16"],
                    description="Trace A tensors use ['torch.bfloat16'], Trace B tensors use ['torch.float16'] — likely cause of accuracy issues",
                )
            ],
        )
        output = format_summary(result)
        self.assertIn("DTYPE MISMATCH", output)
        self.assertIn("torch.bfloat16", output)
        self.assertIn("torch.float16", output)
        self.assertIn("a_ptr", output)
        self.assertIn("x_ptr", output)

    def test_format_summary_shows_dtype_inventories(self) -> None:
        """format_summary shows dtype inventories when no per-arg comparisons."""
        from tritonparse.diff.core.diff_types import (
            CompilationDiffResult,
            DtypeMismatch,
        )

        result = CompilationDiffResult()
        result.tensor_value_diff = TensorValueDiff(
            status="divergent",
            dtype_inventory_a={"a_ptr": "torch.bfloat16"},
            dtype_inventory_b={"x_ptr": "torch.float16"},
            dtype_mismatches=[
                DtypeMismatch(
                    dtypes_a=["torch.bfloat16"],
                    dtypes_b=["torch.float16"],
                    description="mismatch",
                )
            ],
        )
        output = format_summary(result)
        self.assertIn("Trace A tensor args", output)
        self.assertIn("Trace B tensor args", output)

    def test_common_names_per_arg_dtype_mismatch(self) -> None:
        """Per-arg dtype_mismatch status is counted as divergent."""
        launch_a = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float32",
                    "blob_path": "/tmp/a.bin",
                },
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024],
                    "dtype": "torch.float16",
                    "blob_path": "/tmp/b.bin",
                },
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "divergent")
        self.assertEqual(result.per_arg["x_ptr"].status, "dtype_mismatch")
        self.assertEqual(result.args_divergent, 1)

    def test_end_to_end_dtype_mismatch_output(self) -> None:
        """Full pipeline: DiffEngine -> format_summary shows dtype mismatch."""
        launch_a = create_launch_event(
            extracted_args={
                "a_ptr": {
                    "type": "tensor",
                    "shape": [1024, 1024],
                    "dtype": "torch.bfloat16",
                },
                "b_ptr": {
                    "type": "tensor",
                    "shape": [1024, 1024],
                    "dtype": "torch.bfloat16",
                },
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_ptr": {
                    "type": "tensor",
                    "shape": [1024, 1024],
                    "dtype": "torch.float16",
                },
                "y_ptr": {
                    "type": "tensor",
                    "shape": [1024, 1024],
                    "dtype": "torch.float16",
                },
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

        # Verify data model
        self.assertEqual(result.tensor_value_diff.status, "divergent")
        self.assertEqual(len(result.tensor_value_diff.dtype_mismatches), 1)

        # Verify highlights
        highlights_text = " ".join(result.summary.highlights)
        self.assertIn("DTYPE MISMATCH", highlights_text)

        # Verify notes
        correctness_notes = [
            n for n in result.summary.notes if n.category == "correctness"
        ]
        self.assertGreater(len(correctness_notes), 0)

        # Verify formatted output
        output = format_summary(result)
        self.assertIn("DTYPE MISMATCH", output)
        self.assertIn("torch.bfloat16", output)
        self.assertIn("torch.float16", output)
        self.assertIn("a_ptr", output)
        self.assertIn("x_ptr", output)


class TestTensorDescriptorSupport(unittest.TestCase):
    """Tests for TensorDescriptor arg type support."""

    def test_tensor_descriptor_recognized_as_tensor(self) -> None:
        """TensorDescriptor args are recognized and compared."""
        launch_a = create_launch_event(
            extracted_args={
                "a_desc": {
                    "type": "TensorDescriptor",
                    "base": {
                        "type": "tensor",
                        "shape": [1024],
                        "dtype": "torch.bfloat16",
                        "min": 0.0,
                        "max": 1.0,
                        "mean": 0.5,
                        "std": 0.29,
                    },
                    "shape": [1024],
                    "strides": [1],
                    "block_shape": [64],
                },
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "a_desc": {
                    "type": "TensorDescriptor",
                    "base": {
                        "type": "tensor",
                        "shape": [1024],
                        "dtype": "torch.bfloat16",
                        "min": 0.0,
                        "max": 1.0,
                        "mean": 0.5,
                        "std": 0.29,
                    },
                    "shape": [1024],
                    "strides": [1],
                    "block_shape": [64],
                },
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "identical")
        self.assertEqual(result.args_compared, 1)
        self.assertEqual(result.dtype_inventory_a, {"a_desc": "torch.bfloat16"})

    def test_tensor_descriptor_dtype_mismatch(self) -> None:
        """Dtype mismatch detected through TensorDescriptor base."""
        launch_a = create_launch_event(
            extracted_args={
                "a_desc": {
                    "type": "TensorDescriptor",
                    "base": {
                        "type": "tensor",
                        "shape": [1024],
                        "dtype": "torch.bfloat16",
                        "blob_path": "/tmp/a.bin",
                    },
                },
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "a_desc": {
                    "type": "TensorDescriptor",
                    "base": {
                        "type": "tensor",
                        "shape": [1024],
                        "dtype": "torch.float16",
                        "blob_path": "/tmp/b.bin",
                    },
                },
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "divergent")
        self.assertEqual(result.per_arg["a_desc"].status, "dtype_mismatch")
        self.assertEqual(result.per_arg["a_desc"].dtype_a, "torch.bfloat16")
        self.assertEqual(result.per_arg["a_desc"].dtype_b, "torch.float16")

    def test_tensor_descriptor_cross_side_mismatch(self) -> None:
        """Cross-side dtype mismatch with TensorDescriptor and no common names."""
        launch_a = create_launch_event(
            extracted_args={
                "a_desc": {
                    "type": "TensorDescriptor",
                    "base": {"type": "tensor", "dtype": "torch.bfloat16"},
                },
            }
        )
        launch_b = create_launch_event(
            extracted_args={
                "x_desc": {
                    "type": "TensorDescriptor",
                    "base": {"type": "tensor", "dtype": "torch.float16"},
                },
            }
        )
        result = TensorValueAnalyzer(launch_a, launch_b).analyze()
        self.assertEqual(result.status, "divergent")
        self.assertEqual(len(result.dtype_mismatches), 1)
        self.assertEqual(result.dtype_inventory_a, {"a_desc": "torch.bfloat16"})
        self.assertEqual(result.dtype_inventory_b, {"x_desc": "torch.float16"})


if __name__ == "__main__":
    unittest.main()
