# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest
from collections import namedtuple
from unittest.mock import patch

import torch
import triton.language as tl
from tritonparse import structured_logging
from tritonparse._json_compat import dumps, loads
from tritonparse.reproducer.function_extractor import extract_utility_functions
from tritonparse.reproducer.ingestion.ndjson import build_context_bundle
from tritonparse.reproducer.stub_generator import get_constexpr_values
from tritonparse.reproducer.utils import _create_arg_from_info, create_args_from_json
from tritonparse.structured_logging import convert, extract_arg_info


def _capture(value):
    return loads(dumps(convert(extract_arg_info({"ARG": value})["ARG"])))


class TupleArgsTest(unittest.TestCase):
    def setUp(self) -> None:
        for flag in (
            "TRITONPARSE_SAVE_TENSOR_BLOBS",
            "TRITONPARSE_MORE_TENSOR_INFORMATION",
        ):
            patcher = patch.object(structured_logging, flag, False)
            self.addCleanup(patcher.stop)
            patcher.start()

    def test_structured_capture(self) -> None:
        self.assertEqual(
            _capture((1, True, "ieee", None)),
            {
                "type": "tuple",
                "items": [
                    {"type": "int", "value": 1},
                    {"type": "bool", "value": True},
                    {"type": "str", "value": "ieee", "length": 4},
                    {"type": "NoneType", "repr": "None"},
                ],
            },
        )

    def test_round_trip_empty_singleton_long_and_nested_tuples(self) -> None:
        for value in (
            (),
            (3,),
            (1, True, "4", None, 0.5),
            tuple(range(256)),
            ("x" * 300,),
            ((1, (2,)), (), ("ieee", False)),
        ):
            with self.subTest(value=value):
                info = _capture(value)
                self.assertNotIn("repr", info)
                actual = _create_arg_from_info(info)
                self.assertIs(type(actual), tuple)
                self.assertEqual(actual, value)
                self.assertEqual(
                    [type(item) for item in actual], [type(item) for item in value]
                )

    def test_unsupported_nested_items_are_rejected(self) -> None:
        class UnsupportedArg:
            pass

        Config = namedtuple("Config", ["value"])
        values = (
            ([1, 2],),
            (0, ([1, 2],)),
            (UnsupportedArg(),),
            (0, (UnsupportedArg(),)),
            Config([1, 2]),
        )
        arguments = [*values, *(tl.constexpr(value) for value in values)]
        rebuilders = [("library", create_args_from_json)]
        for embed_context in (False, True):
            namespace = {}
            source = extract_utility_functions(embed_context=embed_context)
            exec(compile(source, "<reproducer>", "exec"), namespace)
            rebuilders.append(
                (f"generated_{embed_context}", namespace["create_args_from_json"])
            )
        for source_name, rebuild in rebuilders:
            for value in arguments:
                info = _capture(value)
                with (
                    self.subTest(source=source_name, value=value),
                    self.assertRaisesRegex(
                        NotImplementedError,
                        "Argument 'CONFIG': Unsupported argument type",
                    ),
                ):
                    rebuild({"extracted_args": {"CONFIG": info}})

    def test_top_level_unsupported_type_keeps_legacy_fallback(self) -> None:
        _, args = create_args_from_json({"extracted_args": {"ARG": _capture([1, 2])}})
        self.assertIsNone(args["ARG"])

    def test_tuple_subclass_is_rebuilt_structurally(self) -> None:
        value = torch.Size((2, 3))
        _, args = create_args_from_json({"extracted_args": {"SHAPE": _capture(value)}})
        self.assertIs(type(args["SHAPE"]), tuple)
        self.assertEqual(args["SHAPE"], value)

    def test_nested_constexpr_and_dtype(self) -> None:
        value = (tl.constexpr(32), (tl.float16, tl.constexpr(tl.float32)))
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                info = _capture(tl.constexpr(value) if wrapped else value)
                actual = _create_arg_from_info(info)
                if wrapped:
                    self.assertIsInstance(actual, tl.constexpr)
                    actual = actual.value
                self.assertIsInstance(actual[0], tl.constexpr)
                self.assertEqual(actual[0].value, 32)
                self.assertEqual(actual[1][0], tl.float16)
                self.assertIsInstance(actual[1][1], tl.constexpr)
                self.assertEqual(actual[1][1].value, tl.float32)

    def test_nested_namedtuple_fields_and_type_name(self) -> None:
        Tile = namedtuple("Tile", ["rows", "cols"])
        Config = namedtuple("Config", ["tile", "dtype", "metadata"])
        value = Config(Tile(64, 128), tl.float16, (tl.constexpr(2), ()))
        info = _capture(value)
        self.assertEqual(info["type"], "tuple")
        self.assertEqual(info["tuple_type"], "Config")
        self.assertEqual(info["fields"], ["tile", "dtype", "metadata"])
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                actual = _create_arg_from_info(
                    {**info, "constexpr": True} if wrapped else info
                )
                if wrapped:
                    self.assertIsInstance(actual, tl.constexpr)
                    actual = actual.value
                self.assertEqual(type(actual).__name__, "Config")
                self.assertIsNot(type(actual), Config)
                self.assertEqual(actual._fields, value._fields)
                self.assertEqual(type(actual.tile).__name__, "Tile")
                self.assertIsNot(type(actual.tile), Tile)
                self.assertEqual(actual.tile._fields, ("rows", "cols"))
                self.assertEqual(actual.tile.cols, 128)
                self.assertEqual(actual.dtype, tl.float16)
                self.assertIsInstance(actual.metadata[0], tl.constexpr)
                self.assertEqual(actual.metadata[0].value, 2)
                self.assertEqual(actual.metadata[1], ())

    def test_nested_tensor_metadata_round_trip(self) -> None:
        tensor = torch.empty_strided((2, 3), (1, 2), dtype=torch.int32, device="cpu")
        info = _capture((0, (tensor,)))
        tensor_info = info["items"][1]["items"][0]
        self.assertEqual(tensor_info["type"], "tensor")
        self.assertEqual(tensor_info["shape"], [2, 3])
        self.assertEqual(tensor_info["stride"], [1, 2])
        actual = _create_arg_from_info(info)[1][0]
        self.assertIsInstance(actual, torch.Tensor)
        self.assertEqual(actual.shape, tensor.shape)
        self.assertEqual(actual.stride(), tensor.stride())
        self.assertEqual(actual.dtype, tensor.dtype)
        self.assertEqual(actual.device, tensor.device)

    def test_legacy_tuple_repr_requires_recapture(self) -> None:
        for arg_repr in ("(1, 2)", "(0, 1, ...", "__import__('os').getcwd()"):
            with self.subTest(arg_repr=arg_repr):
                with self.assertRaisesRegex(NotImplementedError, "Re-capture"):
                    _create_arg_from_info({"type": "tuple", "repr": arg_repr})

    def test_malformed_tuple_metadata_is_rejected(self) -> None:
        for metadata in (
            {"items": "(1, 2)"},
            {"items": [1]},
            {"items": [{}]},
            {"items": [], "fields": ["x"], "tuple_type": "Config"},
            {"items": [], "fields": [], "tuple_type": "__import__('os')"},
            {"items": [], "fields": [], "tuple_type": "class"},
            {"items": [], "fields": []},
            {
                "items": [{"type": "int", "value": 1}],
                "fields": ["x.y"],
                "tuple_type": "Config",
            },
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaises(NotImplementedError):
                    _create_arg_from_info({"type": "tuple", **metadata})

    def test_nested_legacy_error_names_outer_argument(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Argument 'CONFIG'"):
            create_args_from_json(
                {
                    "extracted_args": {
                        "CONFIG": {
                            "type": "tuple",
                            "items": [{"type": "tuple", "repr": "(1, 2)"}],
                        }
                    }
                }
            )

    def test_reader_preserves_tuple_metadata(self) -> None:
        Config = namedtuple("Config", ["tile", "enabled"])
        scalar_info = _capture(tl.constexpr((64, (128, True))))
        config_info = _capture(Config((64, 128), True))
        tensor_info = _capture((torch.empty(2, device="cpu"), (3,)))
        launch = {
            "event_type": "launch",
            "grid": [1],
            "extracted_args": {
                "META": scalar_info,
                "CONFIG": config_info,
                "TENSORS": tensor_info,
            },
            "compilation_metadata": {"hash": "abc"},
        }
        compilation = {
            "event_type": "compilation",
            "payload": {
                "metadata": {"hash": "abc", "name": "kernel"},
                "python_source": {
                    "file_path": "/tmp/kernel.py",
                    "code": "@triton.jit\n"
                    "def kernel(META: tl.constexpr, CONFIG, TENSORS):\n"
                    "    pass\n",
                },
            },
        }
        bundle = build_context_bundle([launch, compilation], line_index=0)
        self.assertEqual(bundle.launch["kwargs"]["META"], (64, (128, True)))
        self.assertEqual(bundle.launch["kwargs"]["CONFIG"], ((64, 128), True))
        self.assertNotIn("TENSORS", bundle.launch["kwargs"])
        self.assertEqual(bundle.args["TENSORS"], tensor_info)
        self.assertEqual(bundle.args["CONFIG"]["fields"], config_info["fields"])
        self.assertEqual(bundle.args["META"]["items"], scalar_info["items"])
        self.assertTrue(bundle.args["META"]["constexpr"])
        self.assertEqual(get_constexpr_values(bundle), {"META": (64, (128, True))})
        for extracted_args in (bundle.raw_launch_event["extracted_args"], bundle.args):
            data = loads(dumps({"grid": [1], "extracted_args": extracted_args}))
            grid, args = create_args_from_json(data)
            self.assertEqual(grid, [1])
            self.assertEqual(args["CONFIG"].tile, (64, 128))
            self.assertEqual(args["META"].value, (64, (128, True)))
            self.assertIsInstance(args["TENSORS"][0], torch.Tensor)

    def test_generated_utilities_rebuild_tuple_args(self) -> None:
        Config = namedtuple("Config", ["tile", "dtype", "tensor"])
        value = Config(
            (tl.constexpr(64), (128,)),
            tl.float16,
            torch.empty(2, dtype=torch.int32, device="cpu"),
        )
        data = {
            "grid": [2],
            "extracted_args": {
                "CONFIG": _capture(value),
                "META": _capture(tl.constexpr((1, (2, 3)))),
                "NONE": _capture(tl.constexpr((None, (None,)))),
                "EMPTY": _capture(()),
                "LONG": _capture(tuple(range(256))),
            },
        }
        for embed_context in (False, True):
            with self.subTest(embed_context=embed_context):
                namespace = {}
                source = extract_utility_functions(embed_context=embed_context)
                exec(compile(source, "<reproducer>", "exec"), namespace)
                grid, args = namespace["create_args_from_json"](data)
                self.assertEqual(grid, [2])
                actual = args["CONFIG"]
                self.assertEqual(type(actual).__name__, "Config")
                self.assertEqual(actual._fields, value._fields)
                self.assertIsInstance(actual.tile[0], tl.constexpr)
                self.assertEqual(actual.tile[0].value, 64)
                self.assertEqual(actual.tile[1], (128,))
                self.assertEqual(actual.dtype, tl.float16)
                self.assertEqual(actual.tensor.shape, (2,))
                self.assertEqual(actual.tensor.dtype, torch.int32)
                self.assertIsInstance(args["META"], tl.constexpr)
                self.assertEqual(args["META"].value, (1, (2, 3)))
                self.assertIsInstance(args["NONE"], tl.constexpr)
                self.assertEqual(args["NONE"].value, (None, (None,)))
                self.assertEqual(args["EMPTY"], ())
                self.assertEqual(args["LONG"], tuple(range(256)))
