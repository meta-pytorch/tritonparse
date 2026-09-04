# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest
from unittest import mock

from tritonparse import structured_logging
from tritonparse.kernel_filter import matches_kernel_name, parse_kernel_patterns


class KernelPatternParsingTest(unittest.TestCase):
    def test_empty_specs_disable_filtering(self) -> None:
        for spec in (None, "", "  ", ", ,"):
            with self.subTest(spec=spec):
                self.assertIsNone(parse_kernel_patterns(spec))

    def test_patterns_are_trimmed_and_empty_items_removed(self) -> None:
        self.assertEqual(
            parse_kernel_patterns(" matmul* , , *attention* ,matmul* "),
            ("matmul*", "*attention*", "matmul*"),
        )


class KernelNameMatchingTest(unittest.TestCase):
    def test_no_patterns_match_every_name(self) -> None:
        self.assertTrue(matches_kernel_name("kernel", None))
        self.assertTrue(matches_kernel_name(None, None))

    def test_none_name_does_not_match_active_filter(self) -> None:
        self.assertFalse(matches_kernel_name(None, ("*",)))

    def test_patterns_use_fnmatch_or_semantics(self) -> None:
        patterns = ("matmul?", "*attention*", "kernel_[ab]")
        for kernel_name in ("matmul1", "flash_attention_fwd", "kernel_b"):
            with self.subTest(kernel_name=kernel_name):
                self.assertTrue(matches_kernel_name(kernel_name, patterns))
        self.assertFalse(matches_kernel_name("softmax", patterns))

    def test_empty_pattern_sequence_matches_nothing(self) -> None:
        self.assertFalse(matches_kernel_name("kernel", ()))


class StructuredLoggingCompatibilityTest(unittest.TestCase):
    def test_parse_kernel_allowlist_preserves_list_return_type(self) -> None:
        with mock.patch.object(
            structured_logging,
            "TRITONPARSE_KERNEL_ALLOWLIST",
            " matmul* ,, *attention* ",
        ):
            self.assertEqual(
                structured_logging.parse_kernel_allowlist(),
                ["matmul*", "*attention*"],
            )

    def test_should_trace_kernel_preserves_allowlist_behavior(self) -> None:
        self.assertTrue(structured_logging.should_trace_kernel(None, None))
        self.assertFalse(structured_logging.should_trace_kernel(None, ["*"]))
        self.assertTrue(
            structured_logging.should_trace_kernel("matmul_kernel", ["matmul*"])
        )
        self.assertFalse(
            structured_logging.should_trace_kernel("softmax_kernel", ["matmul*"])
        )
