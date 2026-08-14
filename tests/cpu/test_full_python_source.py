# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for the full-Python-source capture switch.

Covers the ``init(enable_full_python_source=...)`` keyword, its precedence over
the ``TRITON_FULL_PYTHON_SOURCE`` environment variable, and the effect the switch
has on ``extract_python_source_info``.
"""

import inspect
import types
import unittest
from unittest.mock import patch

from tritonparse import shared_vars, structured_logging
from tritonparse.shared_vars import set_runtime_sass_dump_override
from tritonparse.structured_logging import (
    extract_python_source_info,
    is_full_python_source_enabled,
    set_runtime_full_python_source_override,
)

# Marker that lives inside the "real" kernel body only. Function-only capture of
# the thin wrapper below must not contain it; full-file capture must.
#
# Do NOT fold this back into a single literal: the assertions below compare the
# marker against the *contents of this very file*, so a literal spelling here
# would put the marker outside the inner kernel too, and every full-file
# assertion would then pass whether or not the kernel body was captured.
NESTED_BODY_MARKER = "tlx_barrier_wait" + "_stand_in"


def _nested_inner_kernel(a, b):
    """Stand-in for the real kernel body of a nested-kernel pair."""
    tlx_barrier_wait_stand_in = a + b  # noqa: F841
    return tlx_barrier_wait_stand_in


def _nested_kernel_wrapper(a, b):
    return _nested_inner_kernel(a, b)


# Resolve the path the same way extract_python_source_info() does, so the
# assertions hold regardless of how the test binary lays sources out on disk.
SOURCE_FILE = inspect.getfile(_nested_kernel_wrapper)


class FullPythonSourceOverrideTest(unittest.TestCase):
    """Precedence between the runtime override and TRITON_FULL_PYTHON_SOURCE."""

    def setUp(self):
        # Every test starts with no override and restores that on the way out, so
        # the module-level default (env var decides) is never leaked between tests.
        set_runtime_full_python_source_override(None)
        self.addCleanup(set_runtime_full_python_source_override, None)

    def test_no_override_follows_env_var_when_unset(self):
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            self.assertFalse(is_full_python_source_enabled())

    def test_no_override_follows_env_var_when_set(self):
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", True):
            self.assertTrue(is_full_python_source_enabled())

    def test_override_true_wins_over_env_var(self):
        set_runtime_full_python_source_override(True)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            self.assertTrue(is_full_python_source_enabled())

    def test_override_false_wins_over_env_var(self):
        set_runtime_full_python_source_override(False)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", True):
            self.assertFalse(is_full_python_source_enabled())

    def test_clearing_override_restores_env_var(self):
        set_runtime_full_python_source_override(True)
        set_runtime_full_python_source_override(None)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            self.assertFalse(is_full_python_source_enabled())


class InitFullPythonSourceTest(unittest.TestCase):
    """init() wiring for the enable_full_python_source keyword."""

    def setUp(self):
        set_runtime_full_python_source_override(None)
        self.addCleanup(set_runtime_full_python_source_override, None)
        # init() also calls set_runtime_sass_dump_override(), and with
        # enable_sass_dump defaulting to False that clears whatever override was
        # installed process-wide. Snapshot and restore it so these tests do not
        # silently reconfigure SASS dumping for whatever runs after them.
        self.addCleanup(
            set_runtime_sass_dump_override, shared_vars._RUNTIME_SASS_DUMP_OVERRIDE
        )

    def _init(self, **kwargs):
        # init_basic() would create trace folders and install log handlers, and
        # triton.knobs would install a real compilation listener; neither is
        # needed to observe how the keyword is plumbed through.
        with patch.object(structured_logging, "init_basic"), patch("triton.knobs"):
            structured_logging.init(**kwargs)

    def test_init_without_keyword_leaves_env_var_in_charge(self):
        self._init()
        self.assertIsNone(structured_logging._RUNTIME_FULL_PYTHON_SOURCE_OVERRIDE)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", True):
            self.assertTrue(is_full_python_source_enabled())
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            self.assertFalse(is_full_python_source_enabled())

    def test_init_true_forces_full_source(self):
        self._init(enable_full_python_source=True)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            self.assertTrue(is_full_python_source_enabled())

    def test_init_false_forces_function_only(self):
        self._init(enable_full_python_source=False)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", True):
            self.assertFalse(is_full_python_source_enabled())

    def test_last_init_call_wins(self):
        # Same "last call wins" semantics as enable_sass_dump: a later init()
        # that omits the keyword hands control back to the environment variable.
        self._init(enable_full_python_source=True)
        self._init()
        self.assertIsNone(structured_logging._RUNTIME_FULL_PYTHON_SOURCE_OVERRIDE)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            self.assertFalse(is_full_python_source_enabled())

    def test_clear_logging_config_drops_the_override(self):
        # clear_logging_config() is how a session ends -- it is what
        # TritonParseManager.__exit__ calls, and what the test suite calls
        # between cases for isolation -- so the override must not outlive it.
        self._init(enable_full_python_source=True)
        with patch("triton.knobs"):
            structured_logging.clear_logging_config()
        # Back to "the environment variable decides", in both directions.
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            self.assertFalse(is_full_python_source_enabled())
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", True):
            self.assertTrue(is_full_python_source_enabled())


class ExtractPythonSourceInfoTest(unittest.TestCase):
    """End effect of the switch on the captured python_source payload."""

    def setUp(self):
        set_runtime_full_python_source_override(None)
        self.addCleanup(set_runtime_full_python_source_override, None)
        # ASTSource duck type: extract_python_source_info() only reads `.fn`.
        self.source = types.SimpleNamespace(fn=_nested_kernel_wrapper)

    def _extract(self):
        trace_data = {}
        extract_python_source_info(trace_data, self.source)
        return trace_data["python_source"]

    def test_default_captures_only_the_wrapper(self):
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            python_source = self._extract()
        self.assertEqual(python_source["file_path"], SOURCE_FILE)
        # The wrapper does not start at line 1 of this file, proving this is a
        # function-only slice rather than the whole file.
        self.assertGreater(python_source["start_line"], 1)
        self.assertIn("_nested_kernel_wrapper", python_source["code"])
        # The nested kernel's real body is missing -- this is the bug that
        # enable_full_python_source exists to fix.
        self.assertNotIn(NESTED_BODY_MARKER, python_source["code"])

    def test_override_captures_the_whole_file(self):
        set_runtime_full_python_source_override(True)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            python_source = self._extract()

        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            expected_code = f.read()

        self.assertEqual(python_source["file_path"], SOURCE_FILE)
        self.assertEqual(python_source["start_line"], 1)
        self.assertEqual(python_source["end_line"], len(expected_code.split("\n")))
        self.assertEqual(python_source["code"], expected_code)
        # The nested kernel's real body is now present.
        self.assertIn(NESTED_BODY_MARKER, python_source["code"])
        # Full-file mode still reports where the traced function itself lives.
        self.assertGreater(python_source["function_start_line"], 1)
        self.assertGreaterEqual(
            python_source["function_end_line"], python_source["function_start_line"]
        )

    def test_function_range_matches_function_only_capture(self):
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            function_only = self._extract()
        set_runtime_full_python_source_override(True)
        with patch.object(structured_logging, "TRITON_FULL_PYTHON_SOURCE", False):
            full_file = self._extract()

        self.assertEqual(full_file["function_start_line"], function_only["start_line"])
        self.assertEqual(full_file["function_end_line"], function_only["end_line"])

    def test_oversized_file_falls_back_to_function_only(self):
        # TRITON_MAX_SOURCE_SIZE still caps full-source extraction.
        set_runtime_full_python_source_override(True)
        with patch.object(structured_logging, "TRITON_MAX_SOURCE_SIZE", 1):
            python_source = self._extract()
        self.assertGreater(python_source["start_line"], 1)
        self.assertNotIn(NESTED_BODY_MARKER, python_source["code"])
        self.assertNotIn("function_start_line", python_source)


if __name__ == "__main__":
    unittest.main()
