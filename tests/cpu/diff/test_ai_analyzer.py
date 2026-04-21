# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for AI diff analyzer."""

import unittest

from tritonparse.ai.client import MockClient
from tritonparse.diff.core.diff_engine import DiffEngine
from tritonparse.diff.core.diff_types import DiffNote
from tritonparse.diff.fb.ai.diff_analyzer import AIDiffAnalyzer

from .test_fixtures import COMP_EVENT_A, COMP_EVENT_B


MOCK_AI_RESPONSE = """
## Analysis

The compilation differs primarily due to autotuning configuration changes.

### Note 1
- **Root Cause**: Autotuner selected different parallelism parameters
- **Category**: performance
- **Confidence**: high
- **Explanation**: num_warps changed from 4 to 8, doubling thread-level parallelism.
  This typically improves occupancy on compute-bound kernels but increases register
  pressure. The IR expansion in TTIR (+3 lines) reflects additional warp management
  instructions.
- **Recommendation**: Benchmark both configurations to confirm the autotuner's choice.
"""


class AIDiffAnalyzerTest(unittest.TestCase):
    """Tests for AIDiffAnalyzer with MockClient."""

    def setUp(self) -> None:
        self.mock_client = MockClient(responses=[MOCK_AI_RESPONSE])
        self.analyzer = AIDiffAnalyzer(client=self.mock_client)
        self.engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B)
        self.result = self.engine.run()

    def test_analyze_returns_notes(self) -> None:
        notes = self.analyzer.analyze(self.result, COMP_EVENT_A, COMP_EVENT_B)
        self.assertIsInstance(notes, list)
        self.assertGreater(len(notes), 0)

    def test_notes_are_diff_notes(self) -> None:
        notes = self.analyzer.analyze(self.result, COMP_EVENT_A, COMP_EVENT_B)
        for note in notes:
            self.assertIsInstance(note, DiffNote)
            self.assertEqual(note.source, "ai")

    def test_notes_have_content(self) -> None:
        notes = self.analyzer.analyze(self.result, COMP_EVENT_A, COMP_EVENT_B)
        for note in notes:
            self.assertTrue(len(note.content) > 0)

    def test_mock_client_receives_messages(self) -> None:
        self.analyzer.analyze(self.result, COMP_EVENT_A, COMP_EVENT_B)
        self.assertIsNotNone(self.mock_client.last_messages)
        self.assertEqual(len(self.mock_client.last_messages), 2)
        self.assertEqual(self.mock_client.last_messages[0].role, "system")
        self.assertEqual(self.mock_client.last_messages[1].role, "user")

    def test_system_prompt_in_messages(self) -> None:
        self.analyzer.analyze(self.result, COMP_EVENT_A, COMP_EVENT_B)
        system_msg = self.mock_client.last_messages[0]
        self.assertIn("Triton", system_msg.content)

    def test_user_prompt_contains_context(self) -> None:
        self.analyzer.analyze(self.result, COMP_EVENT_A, COMP_EVENT_B)
        user_msg = self.mock_client.last_messages[1]
        self.assertIn("Phase 1", user_msg.content)
        self.assertIn("add_kernel", user_msg.content)

    def test_identical_events_returns_empty(self) -> None:
        """No AI analysis needed for identical compilations."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_A)
        result = engine.run()
        notes = self.analyzer.analyze(result, COMP_EVENT_A, COMP_EVENT_A)
        self.assertEqual(notes, [])


class AIDiffAnalyzerIntegrationTest(unittest.TestCase):
    """Integration tests: DiffEngine -> AIDiffAnalyzer -> DiffNote injection."""

    def test_kernel_diff_pipeline_with_mock(self) -> None:
        """Test kernel-level pipeline: diff -> AI -> notes in summary."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B)
        result = engine.run()
        self.assertNotEqual(result.summary.status, "identical")

        mock_client = MockClient(responses=[MOCK_AI_RESPONSE])
        analyzer = AIDiffAnalyzer(client=mock_client)
        notes = analyzer.analyze(result, COMP_EVENT_A, COMP_EVENT_B)

        result.summary.notes.extend(notes)

        ai_notes = [n for n in result.summary.notes if n.source == "ai"]
        self.assertGreater(len(ai_notes), 0)

        from tritonparse.diff.output.summary_formatter import format_summary

        output = format_summary(result)
        self.assertIn("[AI]", output)

    def test_trace_threshold_skips_identical(self) -> None:
        """Test that trace-level AI skips identical kernel matches."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_A)
        result = engine.run()
        self.assertEqual(result.summary.status, "identical")

        mock_client = MockClient(responses=[MOCK_AI_RESPONSE])
        analyzer = AIDiffAnalyzer(client=mock_client)
        notes = analyzer.analyze(result, COMP_EVENT_A, COMP_EVENT_A)

        self.assertEqual(notes, [])
        self.assertEqual(mock_client.call_count, 0)

    def test_trace_threshold_analyzes_significant(self) -> None:
        """Test that trace-level AI runs on significant diffs."""
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B)
        result = engine.run()

        mock_client = MockClient(responses=[MOCK_AI_RESPONSE])
        analyzer = AIDiffAnalyzer(client=mock_client)

        # Simulate trace-level threshold check
        is_significant = result.summary.status == "significant_diff"
        has_tensor_divergence = result.tensor_value_diff.status == "divergent"

        if is_significant or has_tensor_divergence:
            notes = analyzer.analyze(result, COMP_EVENT_A, COMP_EVENT_B)
            self.assertGreater(len(notes), 0)
            self.assertEqual(mock_client.call_count, 1)


if __name__ == "__main__":
    unittest.main()
