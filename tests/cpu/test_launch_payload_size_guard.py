# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for the launch-payload size guard in `tritonparse.parse.event_diff`.

Traces recorded before the writer-side cap can carry multi-MB strings in a
launch event — an opaque `function` handle that some backends report as the
whole compiled binary, and `compilation_metadata.asm`, which duplicates IR the
compilation event already stores in `file_content`. These tests pin that such
values are summarized on both `_generate_launch_diff` paths (single launch and
multi launch), and that the single-launch path applies the same field
exclusions as the multi-launch one.
"""

import unittest

from tritonparse.parse.event_diff import (
    _generate_launch_diff,
    MAX_VALUE_CHARS,
    summarize_oversized_strings,
)

_HUGE = "x" * (MAX_VALUE_CHARS + 1)
_HUGE_SUMMARY = f"<{len(_HUGE)} chars omitted>"


def _launch_event(**overrides):
    """A minimal launch event with the fields the diff logic cares about."""
    event = {
        "event_type": "launch",
        "pid": 1,
        "name": "my_kernel",
        "function": 148350704,
        "stream": 0,
        "grid": [1, 1, 1],
        "timestamp": "2025-01-01T00:00:00Z",
        "occurrence_id": 7,
        "launch_group_hash": "abc123",
        "compilation_metadata": {"hash": "deadbeef", "name": "my_kernel"},
        "extracted_args": {"x": {"type": "tensor", "shape": [1024]}},
    }
    event.update(overrides)
    return event


class SummarizeOversizedStringsTest(unittest.TestCase):
    """Unit behavior of the recursive summarizer."""

    def test_short_string_untouched(self) -> None:
        self.assertEqual(summarize_oversized_strings("ttir"), "ttir")

    def test_string_at_the_limit_untouched(self) -> None:
        at_limit = "x" * MAX_VALUE_CHARS
        self.assertEqual(summarize_oversized_strings(at_limit), at_limit)

    def test_oversized_string_summarized(self) -> None:
        self.assertEqual(summarize_oversized_strings(_HUGE), _HUGE_SUMMARY)

    def test_non_strings_untouched(self) -> None:
        for value in (None, True, 0, 12345, 1.5):
            with self.subTest(value=value):
                self.assertEqual(summarize_oversized_strings(value), value)

    def test_recurses_into_nested_containers(self) -> None:
        value = {"compilation_metadata": {"asm": {"k.ttir": _HUGE}}, "l": [_HUGE, "ok"]}
        self.assertEqual(
            summarize_oversized_strings(value),
            {
                "compilation_metadata": {"asm": {"k.ttir": _HUGE_SUMMARY}},
                "l": [_HUGE_SUMMARY, "ok"],
            },
        )

    def test_input_is_not_mutated(self) -> None:
        value = {"function": _HUGE}
        summarize_oversized_strings(value)
        self.assertEqual(value["function"], _HUGE)


class SingleLaunchDiffTest(unittest.TestCase):
    """The single-launch shortcut must filter exactly like the general path."""

    def test_internal_fields_are_excluded(self) -> None:
        sames, diffs, _ = _generate_launch_diff([(_launch_event(), 0)])
        self.assertEqual(diffs, {})
        self.assertNotIn("occurrence_id", sames)
        self.assertNotIn("launch_group_hash", sames)
        self.assertEqual(sames["name"], "my_kernel")
        self.assertEqual(sames["grid"], [1, 1, 1])

    def test_oversized_function_is_summarized(self) -> None:
        sames, _, _ = _generate_launch_diff([(_launch_event(function=_HUGE), 0)])
        self.assertEqual(sames["function"], _HUGE_SUMMARY)

    def test_oversized_nested_asm_is_summarized(self) -> None:
        event = _launch_event()
        # Triton keys `asm` by artifact filename. The dot in "my_kernel.ttir"
        # is a separator to _flatten_dict, so the round trip re-nests it as
        # {"my_kernel": {"ttir": ...}} — long-standing behavior, unrelated to
        # the size guard.
        event["compilation_metadata"]["asm"] = {"my_kernel.ttir": _HUGE}
        sames, _, _ = _generate_launch_diff([(event, 0)])
        self.assertEqual(
            sames["compilation_metadata"]["asm"]["my_kernel"]["ttir"], _HUGE_SUMMARY
        )
        # Sibling metadata must survive the rewrite.
        self.assertEqual(sames["compilation_metadata"]["hash"], "deadbeef")


class MultiLaunchDiffTest(unittest.TestCase):
    """Both the `sames` and the `distribution` branch get the same guard."""

    def test_oversized_unchanged_value_is_summarized(self) -> None:
        launches = [
            (_launch_event(function=_HUGE), 0),
            (_launch_event(function=_HUGE, grid=[2, 1, 1]), 1),
        ]
        sames, diffs, _ = _generate_launch_diff(launches)
        self.assertEqual(sames["function"], _HUGE_SUMMARY)
        # grid genuinely differs, so it belongs in diffs, not sames.
        self.assertIn("grid", diffs)

    def test_oversized_differing_value_is_summarized(self) -> None:
        other = "y" * (MAX_VALUE_CHARS + 2)
        launches = [
            (_launch_event(name="k", compilation_metadata={"asm": _HUGE}), 0),
            (_launch_event(name="k", compilation_metadata={"asm": other}), 1),
        ]
        _, diffs, _ = _generate_launch_diff(launches)
        values = diffs["compilation_metadata"]["asm"]["values"]
        self.assertEqual(
            sorted(v["value"] for v in values),
            sorted([_HUGE_SUMMARY, f"<{len(other)} chars omitted>"]),
        )

    def test_small_values_are_unaffected(self) -> None:
        launches = [(_launch_event(), 0), (_launch_event(), 1)]
        sames, diffs, index_map = _generate_launch_diff(launches)
        self.assertEqual(sames["function"], 148350704)
        self.assertEqual(diffs, {})
        self.assertEqual(index_map, [{"start": 0, "end": 1}])
