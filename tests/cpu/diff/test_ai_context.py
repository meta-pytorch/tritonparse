# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for AI diff context builder."""

import unittest

from tritonparse.diff.core.diff_engine import DiffEngine
from tritonparse.diff.fb.ai.context_builder import build_diff_context

from .test_fixtures import COMP_EVENT_A, COMP_EVENT_B


class DiffContextBuilderTest(unittest.TestCase):
    """Tests for build_diff_context."""

    def setUp(self) -> None:
        engine = DiffEngine(COMP_EVENT_A, COMP_EVENT_B)
        self.result = engine.run()

    def test_context_not_empty(self) -> None:
        context = build_diff_context(self.result, COMP_EVENT_A, COMP_EVENT_B)
        self.assertTrue(len(context) > 0)

    def test_context_contains_summary_section(self) -> None:
        context = build_diff_context(self.result, COMP_EVENT_A, COMP_EVENT_B)
        self.assertIn("Phase 1 Diff Results", context)
        self.assertIn("Status", context)

    def test_context_contains_metadata(self) -> None:
        context = build_diff_context(self.result, COMP_EVENT_A, COMP_EVENT_B)
        self.assertIn("Metadata", context)
        self.assertIn("num_stages", context)

    def test_context_contains_ir_stats(self) -> None:
        context = build_diff_context(self.result, COMP_EVENT_A, COMP_EVENT_B)
        self.assertIn("IR Statistics", context)

    def test_context_contains_python_source(self) -> None:
        context = build_diff_context(self.result, COMP_EVENT_A, COMP_EVENT_B)
        self.assertIn("Python Source", context)

    def test_context_contains_kernel_info(self) -> None:
        context = build_diff_context(self.result, COMP_EVENT_A, COMP_EVENT_B)
        self.assertIn("add_kernel", context)


if __name__ == "__main__":
    unittest.main()
