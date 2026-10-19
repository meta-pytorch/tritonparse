# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for kernel arguments launched as `tl.constexpr` objects.

Triton accepts a `triton.language.constexpr` object, such as a module-level
`tl.constexpr` global, as a launch argument. The trace must record the wrapped
value, and the reproducer must pass a `tl.constexpr` holding that value so the
kernel is compiled with the same specialization.
"""

import json
import math
import unittest
from unittest.mock import Mock, patch

import triton
import triton.language as tl
from triton import knobs
from triton.runtime.jit import JITFunction
from tritonparse import structured_logging
from tritonparse._json_compat import dumps, loads
from tritonparse.reproducer.function_extractor import extract_utility_functions
from tritonparse.reproducer.ingestion.ndjson import build_context_bundle
from tritonparse.reproducer.stub_generator import get_constexpr_values
from tritonparse.reproducer.utils import _create_arg_from_info, create_args_from_json
from tritonparse.structured_logging import convert, extract_arg_info

# Canonical literals used for legacy repr type inference.
_LITERALS = (4, -1, True, 0.08838834764831845, float("-inf"), None)


def _capture(value):
    """Extract one launch argument and serialize it the way a trace does."""
    return loads(dumps(convert(extract_arg_info({"ARG": value})["ARG"])))


class ConstexprCaptureTest(unittest.TestCase):
    def test_records_wrapped_value(self) -> None:
        cases = [
            (4, {"type": "int", "value": 4}),
            (True, {"type": "bool", "value": True}),
            (0.5, {"type": "float", "value": 0.5}),
            ("ieee", {"type": "str", "value": "ieee", "length": 4}),
            (None, {"type": "NoneType", "repr": "None"}),
            (tl.float16, {"type": "dtype", "repr": "fp16"}),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(_capture(value), expected)
                self.assertEqual(
                    _capture(tl.constexpr(value)), {**expected, "constexpr": True}
                )

    def test_compilation_constants_unwrap_without_warning(self) -> None:
        with self.assertNoLogs("tritonparse.structured_logging", level="WARNING"):
            constants = convert({(22,): tl.constexpr(1), (23,): tl.constexpr(4)})
        self.assertEqual(constants, {"(22,)": 1, "(23,)": 4})


class ConstexprJITHookTest(unittest.TestCase):
    def setUp(self) -> None:
        structured_logging.clear_logging_config()
        self.addCleanup(structured_logging.clear_logging_config)

    def test_only_tritonparse_hook_bypasses_specialization_serialization(self) -> None:
        @triton.jit
        def kernel(ARG):
            pass

        with patch.object(JITFunction, "_call_hook") as original:
            structured_logging.enable_launch_tracing()
            hook = knobs.runtime.jit_post_compile_hook
            self.assertTrue(kernel._call_hook(hook))
            self.assertIs(
                kernel.launch_metadata.func, structured_logging.add_launch_metadata
            )
            original.assert_not_called()

            other_hook = Mock()
            constants = {(0,): tl.constexpr(4)}
            result = kernel._call_hook(other_hook, "key", constants=constants)
            self.assertIs(result, original.return_value)
            original.assert_called_once_with(
                kernel, other_hook, "key", constants=constants
            )
            with self.assertRaises(TypeError):
                json.dumps(tl.constexpr(4))
            structured_logging.disable_launch_tracing()
            self.assertIs(JITFunction._call_hook, original)

    def test_hook_restored_after_disable_and_clear(self) -> None:
        original = JITFunction._call_hook
        for cleanup in (
            structured_logging.disable_launch_tracing,
            structured_logging.clear_logging_config,
        ):
            with self.subTest(cleanup=cleanup.__name__):
                structured_logging.enable_launch_tracing()
                installed = JITFunction._call_hook
                self.assertIsNot(installed, original)
                structured_logging.enable_launch_tracing()
                self.assertIs(JITFunction._call_hook, installed)
                cleanup()
                self.assertIs(JITFunction._call_hook, original)

    def test_cleanup_preserves_a_later_hook_wrapper(self) -> None:
        with patch.object(JITFunction, "_call_hook"):
            structured_logging.enable_launch_tracing()
            later_wrapper = Mock()
            JITFunction._call_hook = later_wrapper
            structured_logging.clear_logging_config()
            self.assertIs(JITFunction._call_hook, later_wrapper)

    def test_tritonparse_hook_respects_kernel_allowlist(self) -> None:
        @triton.jit
        def kernel(ARG):
            pass

        with patch.object(
            structured_logging, "_KERNEL_ALLOWLIST_PATTERNS", ["unrelated_kernel"]
        ):
            structured_logging.enable_launch_tracing()
            self.assertTrue(kernel._call_hook(knobs.runtime.jit_post_compile_hook))
            self.assertIsNone(kernel.launch_metadata)


class ConstexprReproduceTest(unittest.TestCase):
    def assert_constexpr(self, arg, value) -> None:
        self.assertIsInstance(arg, tl.constexpr)
        self.assertEqual(arg.value, value)
        self.assertIs(type(arg.value), type(value))

    def test_round_trip(self) -> None:
        for value in _LITERALS + ("ieee", tl.float16):
            with self.subTest(value=value):
                arg = _create_arg_from_info(_capture(tl.constexpr(value)))
                self.assert_constexpr(arg, value)

    def test_plain_value_is_not_wrapped(self) -> None:
        self.assertEqual(_create_arg_from_info(_capture(4)), 4)
        self.assertEqual(_create_arg_from_info(_capture(float("inf"))), float("inf"))

    def test_float_missing_or_null_value(self) -> None:
        _, args = create_args_from_json(
            {
                "extracted_args": {
                    "MISSING": {"type": "float"},
                    "NULL": {"type": "float", "value": None},
                }
            }
        )
        self.assertEqual(args, {"MISSING": None, "NULL": None})

    def test_non_finite_float_round_trip(self) -> None:
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                plain = _create_arg_from_info(_capture(value))
                wrapped = _create_arg_from_info(_capture(tl.constexpr(value)))
                self.assertIsInstance(wrapped, tl.constexpr)
                for actual in (plain, wrapped.value):
                    self.assertIs(type(actual), float)
                    if math.isnan(value):
                        self.assertTrue(math.isnan(actual))
                    else:
                        self.assertEqual(actual, value)

    def test_legacy_repr(self) -> None:
        for value in _LITERALS:
            arg_repr = str(tl.constexpr(value))
            with self.subTest(arg_repr=arg_repr):
                arg = _create_arg_from_info({"type": "constexpr", "repr": arg_repr})
                self.assert_constexpr(arg, value)

    def test_literal_like_string_type_requires_new_trace(self) -> None:
        for text, inferred in (
            ("4", 4),
            ("True", True),
            ("False", False),
            ("None", None),
            ("0.5", 0.5),
            ("inf", float("inf")),
        ):
            with self.subTest(text=text):
                legacy_repr = str(tl.constexpr(text))
                self.assertEqual(legacy_repr, str(tl.constexpr(inferred)))
                legacy = _create_arg_from_info(
                    {"type": "constexpr", "repr": legacy_repr}
                )
                self.assert_constexpr(legacy, inferred)
                fresh = _create_arg_from_info(_capture(tl.constexpr(text)))
                self.assert_constexpr(fresh, text)

    def test_non_literal_legacy_repr_raises(self) -> None:
        # These reprs do not match a canonical int, float, bool or None.
        for arg_repr in (
            "constexpr[fp16]",
            "constexpr[ieee]",
            "constexpr[(1, 2)]",
            "constexpr[1e5]",
            "constexpr[" + "1" * 200 + "...",
        ):
            with self.subTest(arg_repr=arg_repr):
                with self.assertRaisesRegex(NotImplementedError, "only its repr"):
                    _create_arg_from_info({"type": "constexpr", "repr": arg_repr})

    def test_unsupported_wrapped_value_raises(self) -> None:
        info = _capture(tl.constexpr((1, 2)))
        self.assertEqual(info["type"], "tuple")
        with self.assertRaisesRegex(NotImplementedError, "type 'tuple'"):
            _create_arg_from_info(info)

    def test_error_names_the_argument(self) -> None:
        data = {
            "extracted_args": {
                "DTYPE": {"type": "constexpr", "repr": "constexpr[fp16]"}
            }
        }
        with self.assertRaisesRegex(NotImplementedError, "Argument 'DTYPE'"):
            create_args_from_json(data)

    def test_generated_utilities_rebuild_constexpr_args(self) -> None:
        namespace = {}
        exec(compile(extract_utility_functions(), "<reproducer>", "exec"), namespace)
        data = {
            "grid": [4, 308],
            "extracted_args": {
                "TASK_COUNT_INDEX": {"type": "constexpr", "repr": "constexpr[1]"},
                "PLAN_ERROR_INDEX": {"type": "int", "value": 4, "constexpr": True},
                "MISSING_FLOAT": {"type": "float"},
                "NULL_FLOAT": {"type": "float", "value": None},
                "INFINITE_FLOAT": {"type": "float", "value": "inf"},
                "NAN_FLOAT": {"type": "float", "value": "nan"},
            },
        }
        grid, args = namespace["create_args_from_json"](data)
        self.assertEqual(grid, [4, 308])
        self.assert_constexpr(args["TASK_COUNT_INDEX"], 1)
        self.assert_constexpr(args["PLAN_ERROR_INDEX"], 4)
        self.assertIsNone(args["MISSING_FLOAT"])
        self.assertIsNone(args["NULL_FLOAT"])
        self.assertEqual(args["INFINITE_FLOAT"], float("inf"))
        self.assertTrue(math.isnan(args["NAN_FLOAT"]))

    def test_override_ttir_constexpr_values(self) -> None:
        launch = {
            "event_type": "launch",
            "grid": [1],
            "extracted_args": {
                "x": {"type": "int", "value": 3},
                "INDEX": _capture(tl.constexpr(4)),
                "FLAG": _capture(tl.constexpr(True)),
            },
            "compilation_metadata": {"hash": "abc"},
        }
        compilation = {
            "event_type": "compilation",
            "payload": {
                "metadata": {"hash": "abc", "name": "k"},
                "python_source": {
                    "file_path": "/tmp/k.py",
                    "code": "@triton.jit\n"
                    "def k(x, INDEX: tl.constexpr, FLAG: tl.constexpr):\n"
                    "    pass\n",
                },
            },
        }
        bundle = build_context_bundle([launch, compilation], line_index=0)
        self.assertEqual(get_constexpr_values(bundle), {"INDEX": 4, "FLAG": True})
