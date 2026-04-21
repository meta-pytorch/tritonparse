# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for AI diff analysis system prompt."""

import unittest

from tritonparse.diff.fb.ai.prompts import DIFF_ANALYSIS_SYSTEM_PROMPT


class DiffPromptTest(unittest.TestCase):
    """Tests for DIFF_ANALYSIS_SYSTEM_PROMPT content."""

    def test_prompt_not_empty(self) -> None:
        self.assertTrue(len(DIFF_ANALYSIS_SYSTEM_PROMPT) > 0)

    def test_prompt_contains_triton_concepts(self) -> None:
        prompt = DIFF_ANALYSIS_SYSTEM_PROMPT
        self.assertIn("num_warps", prompt)
        self.assertIn("num_stages", prompt)
        self.assertIn("TTIR", prompt)
        self.assertIn("autotuning", prompt)

    def test_prompt_contains_cause_categories(self) -> None:
        prompt = DIFF_ANALYSIS_SYSTEM_PROMPT
        self.assertIn("Autotuning", prompt)
        self.assertIn("Compiler", prompt)
        self.assertIn("Source Code", prompt)

    def test_prompt_contains_analysis_structure(self) -> None:
        prompt = DIFF_ANALYSIS_SYSTEM_PROMPT
        self.assertIn("Phase 1", prompt)
        self.assertIn("Root Cause", prompt)
        self.assertIn("Confidence", prompt)

    def test_prompt_contains_output_format(self) -> None:
        prompt = DIFF_ANALYSIS_SYSTEM_PROMPT
        self.assertIn("Category", prompt)
        self.assertIn("performance", prompt)
        self.assertIn("correctness", prompt)


if __name__ == "__main__":
    unittest.main()
