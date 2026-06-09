# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for compat_builder.prompts module."""

from __future__ import annotations

import unittest

from tritonparse.compat_builder.prompts import COMPAT_FIX_SYSTEM_PROMPT


class PromptsTest(unittest.TestCase):
    """Verify the system prompt contains required domain concepts."""

    def test_prompt_is_nonempty_string(self) -> None:
        self.assertIsInstance(COMPAT_FIX_SYSTEM_PROMPT, str)
        self.assertGreater(len(COMPAT_FIX_SYSTEM_PROMPT), 100)

    def test_prompt_contains_llvm_api_patterns(self) -> None:
        self.assertIn("renamed", COMPAT_FIX_SYSTEM_PROMPT)
        self.assertIn("Signature changed", COMPAT_FIX_SYSTEM_PROMPT)
        self.assertIn("Header moved", COMPAT_FIX_SYSTEM_PROMPT)

    def test_prompt_forbids_llvm_hash_modification(self) -> None:
        self.assertIn("llvm-hash.txt", COMPAT_FIX_SYSTEM_PROMPT)

    def test_prompt_contains_analysis_stage(self) -> None:
        self.assertIn("Stage 1", COMPAT_FIX_SYSTEM_PROMPT)
        self.assertIn("grep", COMPAT_FIX_SYSTEM_PROMPT.lower())

    def test_prompt_contains_execution_stage(self) -> None:
        self.assertIn("Stage 2", COMPAT_FIX_SYSTEM_PROMPT)
        self.assertIn("commit", COMPAT_FIX_SYSTEM_PROMPT.lower())

    def test_prompt_instructs_comprehensive_search(self) -> None:
        prompt_lower = COMPAT_FIX_SYSTEM_PROMPT.lower()
        self.assertTrue(
            "all call sites" in prompt_lower or "every call site" in prompt_lower
        )

    def test_prompt_specifies_commit_message_format(self) -> None:
        self.assertIn("compat fix:", COMPAT_FIX_SYSTEM_PROMPT)
