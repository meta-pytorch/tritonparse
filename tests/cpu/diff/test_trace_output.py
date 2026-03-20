#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for trace summary formatter and trace diff event writer."""

import json
import unittest
from typing import Any


class TestTraceSummaryFormatter(unittest.TestCase):
    """Tests for the trace summary formatter."""

    def _make_result(self) -> Any:
        """Create a TraceDiffResult for testing."""
        from tritonparse.diff.core.diff_types import (
            KernelMatchResult,
            MatchMethod,
            TraceDiffResult,
            TraceDiffSummary,
            TraceStats,
        )

        return TraceDiffResult(
            diff_id="test-id",
            trace_a=TraceStats(
                trace_path="a.ndjson",
                total_events=10,
                unique_kernels=3,
                total_compilations=3,
                total_launches=5,
                kernel_names=["add_kernel", "matmul_kernel", "old_kernel"],
            ),
            trace_b=TraceStats(
                trace_path="b.ndjson",
                total_events=8,
                unique_kernels=2,
                total_compilations=2,
                total_launches=4,
                kernel_names=["add_kernel", "matmul_kernel_v2"],
            ),
            matched_kernels=[
                KernelMatchResult(
                    kernel_name_a="add_kernel",
                    kernel_name_b="add_kernel",
                    match_method=MatchMethod.HASH,
                    match_confidence=1.0,
                    status="identical",
                    launch_count_a=3,
                    launch_count_b=3,
                ),
                KernelMatchResult(
                    kernel_name_a="matmul_kernel",
                    kernel_name_b="matmul_kernel_v2",
                    match_method=MatchMethod.SOURCE,
                    match_confidence=0.92,
                    status="different",
                    launch_count_a=2,
                    launch_count_b=1,
                ),
            ],
            only_in_a=["old_kernel"],
            only_in_b=[],
            summary=TraceDiffSummary(
                status="significant_diff",
                total_matched=2,
                identical=1,
                similar=0,
                different=1,
                only_a=1,
                only_b=0,
                match_stats={"hash": 1, "source": 1},
            ),
        )

    def test_format_trace_summary_sections(self) -> None:
        """Output has header, matching summary, matched kernels, only-in lists."""
        from tritonparse.diff.output.trace_summary_formatter import format_trace_summary

        result = self._make_result()
        output = format_trace_summary(result)

        self.assertIn("=== Trace Diff Summary ===", output)
        self.assertIn("Trace A: a.ndjson", output)
        self.assertIn("Trace B: b.ndjson", output)
        self.assertIn("Kernel Matching:", output)
        self.assertIn("2 matched", output)
        self.assertIn("Matched Kernels:", output)
        self.assertIn("Only in Trace A:", output)
        self.assertIn("old_kernel", output)

    def test_format_matched_kernel_icons(self) -> None:
        """Correct [=]/[~]/[!] icons for identical/similar/different."""
        from tritonparse.diff.output.trace_summary_formatter import format_trace_summary

        result = self._make_result()
        output = format_trace_summary(result)

        self.assertIn("[=] add_kernel", output)
        self.assertIn("[!] matmul_kernel", output)

    def test_format_match_method_annotations(self) -> None:
        """Non-name matches show [matched by X (Y%)]."""
        from tritonparse.diff.output.trace_summary_formatter import format_trace_summary

        result = self._make_result()
        output = format_trace_summary(result)

        # Hash match shows [matched by hash] (no percentage since confidence is 1.0)
        self.assertIn("[matched by hash]", output)
        # Source match shows [matched by source (92%)]
        self.assertIn("[matched by source (92%)]", output)

    def test_format_arrow_notation(self) -> None:
        """Different kernel names show name_a -> name_b."""
        from tritonparse.diff.output.trace_summary_formatter import format_trace_summary

        result = self._make_result()
        output = format_trace_summary(result)

        # matmul_kernel -> matmul_kernel_v2
        self.assertIn("\u2192", output)
        self.assertIn("matmul_kernel_v2", output)


# --- Trace Diff Event Writer Tests ---


class TestTraceDiffEventWriter(unittest.TestCase):
    """Tests for trace diff event serialization."""

    def _make_trace_result(self) -> Any:
        """Create a simple TraceDiffResult for testing."""
        from tritonparse.diff.core.diff_types import (
            KernelMatchResult,
            MatchMethod,
            TraceDiffResult,
            TraceDiffSummary,
            TraceStats,
        )

        return TraceDiffResult(
            diff_id="test-trace-diff",
            trace_a=TraceStats(trace_path="a.ndjson", unique_kernels=1),
            trace_b=TraceStats(trace_path="b.ndjson", unique_kernels=1),
            matched_kernels=[
                KernelMatchResult(
                    kernel_name_a="k1",
                    kernel_name_b="k1",
                    match_method=MatchMethod.NAME,
                    status="identical",
                ),
            ],
            summary=TraceDiffSummary(
                status="identical",
                total_matched=1,
                identical=1,
                match_stats={"name": 1},
            ),
        )

    def test_create_trace_diff_event_type(self) -> None:
        """event_type is 'trace_diff'."""
        from tritonparse.diff.output.event_writer import create_trace_diff_event

        result = self._make_trace_result()
        event = create_trace_diff_event(result)
        self.assertEqual(event["event_type"], "trace_diff")

    def test_trace_diff_event_has_match_stats(self) -> None:
        """match_stats dict present with correct counts."""
        from tritonparse.diff.output.event_writer import create_trace_diff_event

        result = self._make_trace_result()
        event = create_trace_diff_event(result)
        self.assertIn("summary", event)
        self.assertIn("match_stats", event["summary"])
        self.assertEqual(event["summary"]["match_stats"]["name"], 1)

    def test_serialization_roundtrip(self) -> None:
        """JSON serializable and deserializable."""
        from tritonparse.diff.output.event_writer import create_trace_diff_event

        result = self._make_trace_result()
        event = create_trace_diff_event(result)
        json_str = json.dumps(event)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["event_type"], "trace_diff")
        self.assertEqual(len(parsed["matched_kernels"]), 1)
        self.assertEqual(parsed["matched_kernels"][0]["status"], "identical")


if __name__ == "__main__":
    unittest.main()
