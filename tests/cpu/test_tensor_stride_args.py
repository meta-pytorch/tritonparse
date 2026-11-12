# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Regression tests for rebuilding tensor arguments with custom strides.

Expanded or broadcast views (``t.expand(...)``, ``t.broadcast_to(...)``) have
zero strides, and ``copy_()`` into such a view raises "unsupported operation:
more than one element of the written-to tensor refers to a single memory
location". Every captured layout must rebuild with its shape, stride and
storage offset, and with the saved values when a tensor blob exists. Generated
reproducer scripts embed their own copy of the helpers (see
``extract_utility_functions``), so each check also runs against that copy.
"""

import hashlib
import io
import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from tritonparse._json_compat import dumps, loads
from tritonparse.reproducer.function_extractor import extract_utility_functions
from tritonparse.reproducer.utils import create_args_from_json
from tritonparse.structured_logging import convert, extract_arg_info

# name -> (shape, stride, storage_offset)
LAYOUTS = {
    "contiguous": ([4, 8], [8, 1], 0),
    "scalar_with_offset": ([], [], 2),
    "offset_slice": ([4], [1], 10),
    "padded": ([2, 2], [4, 1], 3),
    "transposed": ([3, 2], [1, 3], 0),
    "gapped": ([2, 2, 2], [1, 4, 6], 0),
    "unfold": ([4, 4], [2, 1], 0),
    "outer_broadcast": ([4, 8], [0, 1], 0),
    "outer_broadcast_with_offset": ([4, 8], [0, 1], 5),
    "inner_broadcast": ([4, 8], [1, 0], 0),
    "middle_broadcast": ([2, 3, 8], [8, 0, 1], 0),
    "full_broadcast": ([4, 8], [0, 0], 0),
}


def _strided_tensor(
    shape: list[int], stride: list[int], storage_offset: int
) -> torch.Tensor:
    span = storage_offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    return torch.randn(span).as_strided(shape, stride, storage_offset)


def _save_blob(tensor: torch.Tensor, directory: str) -> str:
    """Save a tensor under the name tritonparse gives blobs: its content hash."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    data = buffer.getvalue()
    path = Path(directory) / f"{hashlib.blake2b(data).hexdigest()}.bin"
    path.write_bytes(data)
    return str(path)


def _tensor_arg(
    shape: list[int], stride: list[int], storage_offset: int, **extra: str
) -> dict[str, object]:
    return {
        "type": "tensor",
        "dtype": "torch.float32",
        "shape": shape,
        "stride": stride,
        "storage_offset": storage_offset,
        "device": "cpu",
        **extra,
    }


def _rebuilders() -> Iterator[tuple[str, Callable]]:
    """Yield create_args_from_json from the library and from generated scripts."""
    yield "library", create_args_from_json
    for embed_context in (False, True):
        namespace = {}
        source = extract_utility_functions(embed_context=embed_context)
        exec(compile(source, "<reproducer>", "exec"), namespace)
        yield (
            f"generated(embed_context={embed_context})",
            namespace["create_args_from_json"],
        )


def _rebuild(create_args: Callable, arg_info: dict[str, object]) -> torch.Tensor:
    _, args = create_args({"extracted_args": {"ARG": arg_info}})
    return args["ARG"]


class StridedTensorArgTest(unittest.TestCase):
    def test_saved_blob_rebuilds_exact_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for source, create_args in _rebuilders():
                for name, (shape, stride, offset) in LAYOUTS.items():
                    with self.subTest(source=source, layout=name):
                        original = _strided_tensor(shape, stride, offset)
                        arg_info = _tensor_arg(
                            shape,
                            stride,
                            offset,
                            blob_path=_save_blob(original, directory),
                        )
                        rebuilt = _rebuild(create_args, arg_info)
                        self.assertEqual(list(rebuilt.stride()), stride)
                        self.assertEqual(rebuilt.storage_offset(), offset)
                        self.assertTrue(torch.equal(rebuilt, original))

    def test_random_values_keep_the_layout(self) -> None:
        for source, create_args in _rebuilders():
            for name, (shape, stride, offset) in LAYOUTS.items():
                with self.subTest(source=source, layout=name):
                    rebuilt = _rebuild(create_args, _tensor_arg(shape, stride, offset))
                    self.assertEqual(list(rebuilt.shape), shape)
                    self.assertEqual(list(rebuilt.stride()), stride)
                    self.assertEqual(rebuilt.storage_offset(), offset)


class CaptureRoundTripTest(unittest.TestCase):
    """Capture a tensor the way a trace does, then rebuild it."""

    def _rebuild_from_capture(self, tensor: torch.Tensor) -> torch.Tensor:
        arg_info = loads(dumps(convert(extract_arg_info({"ARG": tensor})["ARG"])))
        return _rebuild(create_args_from_json, arg_info)

    def test_expanded_tensor(self) -> None:
        rebuilt = self._rebuild_from_capture(torch.randn(8).expand(4, 8))
        self.assertEqual(list(rebuilt.shape), [4, 8])
        self.assertEqual(list(rebuilt.stride()), [0, 1])

    def test_transposed_tensor(self) -> None:
        rebuilt = self._rebuild_from_capture(torch.randn(2, 3).t())
        self.assertEqual(list(rebuilt.shape), [3, 2])
        self.assertEqual(list(rebuilt.stride()), [1, 3])


if __name__ == "__main__":
    unittest.main()
