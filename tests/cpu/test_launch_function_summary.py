# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for `summarize_launch_function` in `tritonparse.structured_logging`.

A launch event's `function` is an opaque backend handle. CUDA/HIP report an
int address or None, but some backends (e.g. MTIA) hand back the whole
compiled binary, which serializes to megabytes per launch event. These tests
pin the contract that scalars survive untouched while oversized handles are
replaced by a constant-size summary.
"""

import logging
import unittest

import tritonparse.structured_logging as structured_logging
from tritonparse._json_compat import dumps
from tritonparse.structured_logging import (
    MAX_LAUNCH_FUNCTION_CHARS,
    summarize_launch_function,
)


class ScalarHandlesTest(unittest.TestCase):
    """The documented CUDA/HIP shapes must round-trip unchanged."""

    def test_none_passes_through(self) -> None:
        self.assertIsNone(summarize_launch_function(None))

    def test_int_address_passes_through(self) -> None:
        self.assertEqual(summarize_launch_function(140234523443200), 140234523443200)

    def test_bool_is_not_treated_as_an_int_handle(self) -> None:
        # bool subclasses int, but a JSON boolean is not an integer:
        # json_validator rejects one for "integer" and it matches neither
        # "string" nor "null", so passing it through would emit a record that
        # fails schema validation. It has to be stringified instead.
        self.assertEqual(summarize_launch_function(True), "True")
        self.assertEqual(summarize_launch_function(False), "False")

    def test_zero_passes_through(self) -> None:
        # 0 is falsy but is still a valid handle value; it must not be
        # confused with "not captured".
        self.assertEqual(summarize_launch_function(0), 0)


class SmallNonScalarHandlesTest(unittest.TestCase):
    """Short non-scalar handles are kept, just normalized to a string."""

    def test_short_string_is_kept(self) -> None:
        self.assertEqual(summarize_launch_function("0x7f2a1c00"), "0x7f2a1c00")

    def test_short_bytes_is_stringified(self) -> None:
        self.assertEqual(summarize_launch_function(b"\x7fELF"), str(b"\x7fELF"))

    def test_short_memoryview_keeps_its_content(self) -> None:
        # str() of a memoryview is "<memory at 0x...>", which loses the content
        # and changes every run; it must render like the other bytes-like types.
        self.assertEqual(
            summarize_launch_function(memoryview(b"\x7fELF")), str(b"\x7fELF")
        )

    def test_equal_memoryviews_summarize_identically(self) -> None:
        # Two distinct objects over equal bytes must produce equal output, or
        # the field looks like it varies across otherwise identical launches.
        self.assertEqual(
            summarize_launch_function(memoryview(b"\x7fELF")),
            summarize_launch_function(memoryview(bytearray(b"\x7fELF"))),
        )

    def test_value_at_the_limit_is_kept(self) -> None:
        handle = "a" * MAX_LAUNCH_FUNCTION_CHARS
        self.assertEqual(summarize_launch_function(handle), handle)


class OversizedHandlesTest(unittest.TestCase):
    """Anything above the cap collapses to a constant-size summary."""

    def setUp(self) -> None:
        # The message fires once per process; reset so each test observes it.
        structured_logging._logged_oversized_launch_function = False

    def test_oversized_bytes_is_summarized(self) -> None:
        blob = b"\x7fELF" + b"\x00" * (4 * 1024 * 1024)
        summary = summarize_launch_function(blob)
        self.assertEqual(summary, f"<bytes: {len(blob)} bytes omitted>")
        self.assertLess(len(summary), MAX_LAUNCH_FUNCTION_CHARS)

    def test_oversized_bytearray_reports_its_own_type(self) -> None:
        blob = bytearray(4 * 1024 * 1024)
        self.assertEqual(
            summarize_launch_function(blob),
            f"<bytearray: {len(blob)} bytes omitted>",
        )

    def test_binary_handle_within_the_byte_count_but_over_the_char_cap(self) -> None:
        # 256 bytes clears a byte-count check, but its repr is 1027 characters
        # because each NUL renders as "\x00". The cap is on characters, so this
        # must be summarized rather than returned.
        blob = b"\x00" * MAX_LAUNCH_FUNCTION_CHARS
        self.assertGreater(len(str(blob)), MAX_LAUNCH_FUNCTION_CHARS)
        self.assertEqual(
            summarize_launch_function(blob),
            f"<bytes: {len(blob)} bytes omitted>",
        )

    def test_fall_through_log_names_the_rendered_length(self) -> None:
        # The byte count cleared the guard, so the log must not claim
        # "256 bytes, above the 256 limit" — that reads as a contradiction and
        # names the wrong quantity. The 1027-character repr is what tripped it.
        blob = b"\x00" * MAX_LAUNCH_FUNCTION_CHARS
        with self.assertLogs(structured_logging.log, level="DEBUG") as captured:
            summarize_launch_function(blob)
        self.assertIn(f"{len(str(blob))} chars", captured.output[0])
        self.assertNotIn(f"{len(blob)} bytes", captured.output[0])

    def test_printable_handle_of_the_same_byte_count_is_kept(self) -> None:
        # The same byte count renders short when the content is printable, so
        # the decision has to be content-dependent, not size-dependent.
        blob = b"a" * 250
        self.assertLessEqual(len(str(blob)), MAX_LAUNCH_FUNCTION_CHARS)
        self.assertEqual(summarize_launch_function(blob), str(blob))

    def test_result_never_exceeds_the_cap(self) -> None:
        handles = [
            b"\x00" * MAX_LAUNCH_FUNCTION_CHARS,
            b"\x7fELF" + b"\x00" * (4 * 1024 * 1024),
            bytearray(b"\xff" * 300),
            memoryview(bytearray(2048)).cast("q"),
            memoryview(b"\x00" * 300),
            "a" * (MAX_LAUNCH_FUNCTION_CHARS + 1),
            "0x7f2a1c00",
        ]
        for handle in handles:
            with self.subTest(handle=type(handle).__name__):
                result = summarize_launch_function(handle)
                self.assertIsInstance(result, str)
                self.assertLessEqual(len(result), MAX_LAUNCH_FUNCTION_CHARS)

    def test_wide_memoryview_is_measured_in_bytes_not_elements(self) -> None:
        # 2048 bytes viewed as 8-byte elements: len() is 256 — exactly the cap —
        # while nbytes is 2048. Measuring the cap against len() would take the
        # small-value path and return a ~8k-character string.
        wide = memoryview(bytearray(2048)).cast("q")
        self.assertEqual(len(wide), MAX_LAUNCH_FUNCTION_CHARS)
        summary = summarize_launch_function(wide)
        self.assertEqual(summary, f"<memoryview: {wide.nbytes} bytes omitted>")
        self.assertLessEqual(len(summary), MAX_LAUNCH_FUNCTION_CHARS)

    def test_oversized_memoryview_reports_its_own_type(self) -> None:
        blob = memoryview(bytes(4 * 1024 * 1024))
        self.assertEqual(
            summarize_launch_function(blob),
            f"<memoryview: {len(blob)} bytes omitted>",
        )

    def test_oversized_string_is_summarized(self) -> None:
        handle = "a" * (MAX_LAUNCH_FUNCTION_CHARS + 1)
        self.assertEqual(
            summarize_launch_function(handle),
            f"<str: {len(handle)} chars omitted>",
        )

    def test_summary_is_json_serializable(self) -> None:
        summary = summarize_launch_function(b"\x00" * (4 * 1024 * 1024))
        # The raw bytes object is not JSON serializable at all; the summary is.
        self.assertIn("bytes omitted", dumps(summary))

    def test_logs_once_per_process(self) -> None:
        blob = b"\x00" * (4 * 1024 * 1024)
        with self.assertLogs(structured_logging.log, level="DEBUG") as first:
            summarize_launch_function(blob)
        self.assertIn("size summary", first.output[0])

        # A second oversized handle must not re-log. assertLogs fails when
        # nothing is logged, so assert on the flag and on logger silence.
        self.assertTrue(structured_logging._logged_oversized_launch_function)
        with self.assertNoLogs(structured_logging.log, level="DEBUG"):
            summarize_launch_function(blob)

    def test_logs_at_debug_level_only(self) -> None:
        # `tp_logger` puts the `tritonparse` logger at INFO with its own handler
        # and propagate=False, so INFO and above reach the user by default. This
        # message must stay below that: a backend reporting a non-scalar handle
        # is expected and not actionable, and the trace already says so in the
        # field value. Anything louder is a permanent, unactionable log line for
        # every user of that backend.
        with self.assertLogs(structured_logging.log, level="DEBUG") as captured:
            summarize_launch_function(b"\x00" * (4 * 1024 * 1024))
        self.assertEqual([r.levelno for r in captured.records], [logging.DEBUG])

    def test_message_unit_matches_the_summary(self) -> None:
        # The summary says "bytes" for bytes-like and "chars" otherwise; the log
        # line used to hardcode "units" and disagree with it.
        with self.assertLogs(structured_logging.log, level="DEBUG") as captured:
            summarize_launch_function(b"\x00" * (4 * 1024 * 1024))
        self.assertIn("bytes,", captured.output[0])

        structured_logging._logged_oversized_launch_function = False
        with self.assertLogs(structured_logging.log, level="DEBUG") as captured:
            summarize_launch_function("a" * (MAX_LAUNCH_FUNCTION_CHARS + 1))
        self.assertIn("chars,", captured.output[0])
