#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for trace-level data types, event grouping, and TraceDiffEngine."""

import unittest

from .test_fixtures import (
    create_compilation_event,
    create_launch_event,
    DEFAULT_PYTHON_SOURCE,
    DEFAULT_TTIR,
    DIFFERENT_PYTHON_SOURCE_MATMUL,
    LONGER_TTIR,
)


# --- Trace Data Types Tests ---


class TestTraceDataTypes(unittest.TestCase):
    """Tests for trace-level data types."""

    def test_match_method_enum_values(self) -> None:
        """Test MatchMethod enum string values."""
        from tritonparse.diff.core.diff_types import MatchMethod

        self.assertEqual(MatchMethod.HASH.value, "hash")
        self.assertEqual(MatchMethod.NAME.value, "name")
        self.assertEqual(MatchMethod.SOURCE.value, "source")
        self.assertEqual(MatchMethod.FUZZY_NAME.value, "fuzzy_name")
        # str enum allows string comparison
        self.assertEqual(MatchMethod.HASH, "hash")

    def test_kernel_match_result_defaults(self) -> None:
        """Test KernelMatchResult default field values."""
        from tritonparse.diff.core.diff_types import KernelMatchResult, MatchMethod

        result = KernelMatchResult()
        self.assertEqual(result.kernel_name_a, "")
        self.assertEqual(result.kernel_name_b, "")
        self.assertEqual(result.match_method, MatchMethod.NAME)
        self.assertEqual(result.match_confidence, 1.0)
        self.assertEqual(result.status, "")
        self.assertIsNone(result.compilation_diff)
        self.assertEqual(result.launch_count_a, 0)
        self.assertEqual(result.launch_count_b, 0)
        self.assertEqual(result.metadata_changes, [])
        self.assertEqual(result.ir_stat_highlights, [])

    def test_compilation_diff_result_backward_compat(self) -> None:
        """Test that CompilationDiffResult still works with new fields defaulting to None."""
        from tritonparse.diff.core.diff_types import CompilationDiffResult

        result = CompilationDiffResult()
        # New fields default to None
        self.assertIsNone(result.match_method)
        self.assertIsNone(result.match_confidence)
        # Existing fields still have their defaults
        self.assertEqual(result.diff_id, "")
        self.assertEqual(result.kernel_name_a, "")
        self.assertFalse(result.kernel_names_identical)
        self.assertEqual(result.summary.status, "identical")


# --- Event Grouping Tests ---


class TestEventGrouping(unittest.TestCase):
    """Tests for event grouping functions."""

    def test_group_compilations_by_kernel(self) -> None:
        """Test grouping compilation events by kernel name."""
        from tritonparse.diff.core.event_matcher import group_compilations_by_kernel

        events = [
            create_compilation_event(kernel_name="add_kernel", kernel_hash="hash1"),
            create_compilation_event(kernel_name="matmul_kernel", kernel_hash="hash2"),
            create_compilation_event(kernel_name="add_kernel", kernel_hash="hash3"),
        ]
        groups = group_compilations_by_kernel(events)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups["add_kernel"]), 2)
        self.assertEqual(len(groups["matmul_kernel"]), 1)
        # Check compilation indices are correct
        self.assertEqual(groups["add_kernel"][0][0], 0)  # first compilation
        self.assertEqual(groups["add_kernel"][1][0], 2)  # third compilation
        self.assertEqual(groups["matmul_kernel"][0][0], 1)  # second compilation

    def test_group_compilations_empty(self) -> None:
        """Test grouping with empty events list."""
        from tritonparse.diff.core.event_matcher import group_compilations_by_kernel

        groups = group_compilations_by_kernel([])
        self.assertEqual(groups, {})

    def test_group_launches_by_kernel(self) -> None:
        """Test grouping launch events by kernel name."""
        from tritonparse.diff.core.event_matcher import group_launches_by_kernel

        events = [
            {
                "event_type": "launch",
                "compilation_metadata": {"name": "add_kernel", "hash": "h1"},
            },
            {
                "event_type": "launch",
                "compilation_metadata": {"name": "matmul_kernel", "hash": "h2"},
            },
            {
                "event_type": "launch",
                "compilation_metadata": {"name": "add_kernel", "hash": "h1"},
            },
            # non-launch events should be skipped
            create_compilation_event(kernel_name="add_kernel"),
        ]
        groups = group_launches_by_kernel(events)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups["add_kernel"]), 2)
        self.assertEqual(len(groups["matmul_kernel"]), 1)

    def test_group_unknown_kernels(self) -> None:
        """Test that events with no name are grouped under 'unknown'."""
        from tritonparse.diff.core.event_matcher import group_compilations_by_kernel

        event_no_name = {
            "event_type": "compilation",
            "payload": {"ttir": DEFAULT_TTIR, "source_mappings": {}},
        }
        events = [event_no_name]
        groups = group_compilations_by_kernel(events)
        self.assertIn("unknown", groups)
        self.assertEqual(len(groups["unknown"]), 1)

    def test_group_compilations_skips_non_compilation(self) -> None:
        """Test that non-compilation events are skipped."""
        from tritonparse.diff.core.event_matcher import group_compilations_by_kernel

        events = [
            {"event_type": "launch", "compilation_metadata": {"name": "k1"}},
            create_compilation_event(kernel_name="add_kernel"),
        ]
        groups = group_compilations_by_kernel(events)
        self.assertEqual(len(groups), 1)
        self.assertIn("add_kernel", groups)

    def test_group_launches_fallback_name_field(self) -> None:
        """Test that launch grouping falls back to top-level name field."""
        from tritonparse.diff.core.event_matcher import group_launches_by_kernel

        events = [
            {
                "event_type": "launch",
                "name": "my_kernel",
                "compilation_metadata": {"hash": "h1"},
            },
        ]
        groups = group_launches_by_kernel(events)
        self.assertIn("my_kernel", groups)

    def test_group_launches_unknown_fallback(self) -> None:
        """Test that launches with no name info are grouped under 'unknown'."""
        from tritonparse.diff.core.event_matcher import group_launches_by_kernel

        events = [
            {"event_type": "launch", "compilation_metadata": {"hash": "h1"}},
        ]
        groups = group_launches_by_kernel(events)
        self.assertIn("unknown", groups)


# --- TraceDiffEngine Tests ---


class TestTraceDiffEngine(unittest.TestCase):
    """Tests for the TraceDiffEngine orchestrator."""

    def test_identical_traces(self) -> None:
        """Same events -> all 'identical', matched by hash."""
        from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

        events = [
            create_compilation_event(kernel_name="k1", kernel_hash="h1"),
            create_compilation_event(kernel_name="k2", kernel_hash="h2"),
        ]
        engine = TraceDiffEngine(events, list(events), "a.ndjson", "b.ndjson")
        result = engine.run()

        self.assertEqual(len(result.matched_kernels), 2)
        self.assertEqual(len(result.only_in_a), 0)
        self.assertEqual(len(result.only_in_b), 0)
        for m in result.matched_kernels:
            self.assertEqual(m.status, "identical")
        self.assertEqual(result.summary.status, "identical")
        self.assertEqual(result.summary.identical, 2)

    def test_different_compilations(self) -> None:
        """Same kernel names, different hashes -> status 'different'."""
        from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

        events_a = [
            create_compilation_event(
                kernel_name="add_kernel",
                kernel_hash="hash_a",
                num_stages=3,
                num_warps=4,
            ),
        ]
        events_b = [
            create_compilation_event(
                kernel_name="add_kernel",
                kernel_hash="hash_b",
                num_stages=5,
                num_warps=8,
                ttir=LONGER_TTIR,
            ),
        ]
        engine = TraceDiffEngine(events_a, events_b)
        result = engine.run()

        self.assertEqual(len(result.matched_kernels), 1)
        # Different metadata means significant_diff -> "different"
        self.assertIn(result.matched_kernels[0].status, ["similar", "different"])
        self.assertIsNotNone(result.matched_kernels[0].compilation_diff)

    def test_only_in_a_and_b(self) -> None:
        """Non-overlapping kernels populate only_in_a/only_in_b."""
        from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

        events_a = [
            create_compilation_event(
                kernel_name="alpha_compute",
                kernel_hash="ha",
                python_source=DEFAULT_PYTHON_SOURCE,
            ),
        ]
        events_b = [
            create_compilation_event(
                kernel_name="zeta_transform",
                kernel_hash="hb",
                python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
            ),
        ]
        engine = TraceDiffEngine(events_a, events_b)
        result = engine.run()

        self.assertEqual(len(result.matched_kernels), 0)
        self.assertIn("alpha_compute", result.only_in_a)
        self.assertIn("zeta_transform", result.only_in_b)
        self.assertEqual(result.summary.status, "significant_diff")

    def test_mixed_scenario(self) -> None:
        """Mix of identical + only-in."""
        from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

        # Use very different names and sources to prevent fuzzy matching
        only_a_source = """\
@triton.jit
def alpha_reduction(input_ptr, output_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    vals = tl.load(input_ptr + pid * BLOCK + tl.arange(0, BLOCK))
    result = tl.sum(vals)
    tl.store(output_ptr + pid, result)
"""
        only_b_source = """\
@triton.jit
def zeta_scatter(src_ptr, idx_ptr, dst_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.load(idx_ptr + pid * BLOCK + tl.arange(0, BLOCK))
    val = tl.load(src_ptr + idx)
    tl.store(dst_ptr + idx, val)
"""
        events_a = [
            create_compilation_event(kernel_name="shared", kernel_hash="h1"),
            create_compilation_event(
                kernel_name="alpha_reduction_kernel",
                kernel_hash="ha_unique",
                python_source=only_a_source,
            ),
        ]
        events_b = [
            create_compilation_event(kernel_name="shared", kernel_hash="h1"),
            create_compilation_event(
                kernel_name="zeta_scatter_kernel",
                kernel_hash="hb_unique",
                python_source=only_b_source,
            ),
        ]
        engine = TraceDiffEngine(events_a, events_b)
        result = engine.run()

        self.assertEqual(len(result.matched_kernels), 1)
        self.assertEqual(result.matched_kernels[0].status, "identical")
        self.assertIn("alpha_reduction_kernel", result.only_in_a)
        self.assertIn("zeta_scatter_kernel", result.only_in_b)
        self.assertEqual(result.summary.status, "significant_diff")

    def test_launch_count_diff(self) -> None:
        """Same kernels, different launch counts reflected in result."""
        from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

        events_a = [
            create_compilation_event(kernel_name="k1", kernel_hash="h1"),
            create_launch_event(kernel_hash="h1"),
            create_launch_event(kernel_hash="h1"),
            create_launch_event(kernel_hash="h1"),
        ]
        events_b = [
            create_compilation_event(kernel_name="k1", kernel_hash="h1"),
            create_launch_event(kernel_hash="h1"),
        ]
        engine = TraceDiffEngine(events_a, events_b)
        result = engine.run()

        self.assertEqual(len(result.matched_kernels), 1)
        self.assertEqual(result.matched_kernels[0].launch_count_a, 3)
        self.assertEqual(result.matched_kernels[0].launch_count_b, 1)
        # Trace stats should reflect launch counts
        self.assertEqual(result.trace_a.total_launches, 3)
        self.assertEqual(result.trace_b.total_launches, 1)

    def test_representative_selection(self) -> None:
        """Multiple compilations with same name — matching picks correct hash."""
        from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

        # In trace A: kernel "k1" has two compilations with different hashes.
        # Hash "winner" exists in both traces, so it's matched by hash.
        # Hash "loser" only exists in A, so it's unmatched.
        events_a = [
            create_compilation_event(kernel_name="k1", kernel_hash="loser"),
            create_compilation_event(kernel_name="k1", kernel_hash="winner"),
            create_launch_event(kernel_hash="loser"),
            create_launch_event(kernel_hash="winner"),
            create_launch_event(kernel_hash="winner"),
            create_launch_event(kernel_hash="winner"),
        ]
        events_b = [
            create_compilation_event(kernel_name="k1", kernel_hash="winner"),
            create_launch_event(kernel_hash="winner"),
        ]
        engine = TraceDiffEngine(events_a, events_b)
        result = engine.run()

        # Only one match: the "winner" hash matches by hash
        self.assertEqual(len(result.matched_kernels), 1)
        self.assertEqual(result.matched_kernels[0].hash_a, "winner")
        self.assertEqual(result.matched_kernels[0].hash_b, "winner")

    def test_empty_traces(self) -> None:
        """No events -> empty result, no crash."""
        from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

        engine = TraceDiffEngine([], [])
        result = engine.run()

        self.assertEqual(len(result.matched_kernels), 0)
        self.assertEqual(len(result.only_in_a), 0)
        self.assertEqual(len(result.only_in_b), 0)
        self.assertEqual(result.summary.status, "identical")
        self.assertEqual(result.trace_a.total_events, 0)
        self.assertEqual(result.trace_b.total_events, 0)

    def test_match_stats_populated(self) -> None:
        """summary.match_stats has correct per-strategy counts."""
        from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

        events_a = [
            # Will match by hash
            create_compilation_event(kernel_name="k1_old", kernel_hash="shared_h"),
            # Will match by name
            create_compilation_event(kernel_name="k2", kernel_hash="h_a"),
        ]
        events_b = [
            create_compilation_event(kernel_name="k1_new", kernel_hash="shared_h"),
            create_compilation_event(kernel_name="k2", kernel_hash="h_b"),
        ]
        engine = TraceDiffEngine(events_a, events_b)
        result = engine.run()

        self.assertEqual(len(result.matched_kernels), 2)
        self.assertIn("hash", result.summary.match_stats)
        self.assertIn("name", result.summary.match_stats)
        self.assertEqual(result.summary.match_stats["hash"], 1)
        self.assertEqual(result.summary.match_stats["name"], 1)


if __name__ == "__main__":
    unittest.main()
