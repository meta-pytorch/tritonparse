# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for TensorDescriptor capture and reconstruction in tritonparse.

These tests verify that TensorDescriptor arguments from triton.tools.tensor_descriptor
are properly captured during tracing and reconstructed during reproducer generation.

Test Plan:
```
buck test -m ovr_config//triton:beta fbcode//pytorch/tritonparse/tests/gpu:test_tensor_descriptor
```
"""

import unittest

import torch
from tests.test_utils import GPUTestBase
from tritonparse.reproducer.utils import (
    _create_arg_from_info,
    _get_tensor_descriptor_type,
)
from tritonparse.structured_logging import _is_tensor_descriptor, extract_arg_info


class TestTensorDescriptorCapture(GPUTestBase):
    """Tests for TensorDescriptor capture in structured_logging."""

    def test_is_tensor_descriptor_true(self):
        """Test that _is_tensor_descriptor correctly identifies TensorDescriptor objects."""
        TensorDescriptor = _get_tensor_descriptor_type()
        if TensorDescriptor is None:
            self.skipTest("TensorDescriptor not available")

        tensor = torch.randn(128, 64, device=self.cuda_device)
        desc = TensorDescriptor(tensor, tensor.shape, tensor.stride(), [32, 32])

        self.assertTrue(_is_tensor_descriptor(desc))

    def test_is_tensor_descriptor_false_for_tensor(self):
        """Test that _is_tensor_descriptor returns False for regular tensors."""
        tensor = torch.randn(128, 64, device=self.cuda_device)
        self.assertFalse(_is_tensor_descriptor(tensor))

    def test_is_tensor_descriptor_false_for_primitives(self):
        """Test that _is_tensor_descriptor returns False for primitive types."""
        self.assertFalse(_is_tensor_descriptor(42))
        self.assertFalse(_is_tensor_descriptor("string"))
        self.assertFalse(_is_tensor_descriptor(3.14))
        self.assertFalse(_is_tensor_descriptor(None))

    def test_extract_arg_info_captures_tensor_descriptor(self):
        """Test that extract_arg_info properly captures TensorDescriptor metadata."""
        TensorDescriptor = _get_tensor_descriptor_type()
        if TensorDescriptor is None:
            self.skipTest("TensorDescriptor not available")

        tensor = torch.randn(128, 64, device=self.cuda_device, dtype=torch.float16)
        shape = (128, 64)
        strides = (64, 1)
        block_shape = [32, 32]
        desc = TensorDescriptor(tensor, shape, strides, block_shape)

        arg_dict = {"my_desc": desc}
        extracted = extract_arg_info(arg_dict)

        self.assertIn("my_desc", extracted)
        arg_info = extracted["my_desc"]

        self.assertEqual(arg_info["type"], "TensorDescriptor")
        self.assertIn("base", arg_info)
        self.assertEqual(arg_info["shape"], list(shape))
        self.assertEqual(arg_info["strides"], list(strides))
        self.assertEqual(arg_info["block_shape"], block_shape)
        self.assertIn("padding", arg_info)

        base_info = arg_info["base"]
        self.assertEqual(base_info["shape"], list(tensor.shape))
        self.assertIn("dtype", base_info)

    def test_extract_arg_info_with_nan_padding(self):
        """Test that extract_arg_info captures nan padding mode."""
        TensorDescriptor = _get_tensor_descriptor_type()
        if TensorDescriptor is None:
            self.skipTest("TensorDescriptor not available")

        tensor = torch.randn(256, 128, device=self.cuda_device)
        desc = TensorDescriptor(
            tensor, tensor.shape, tensor.stride(), [64, 64], padding="nan"
        )

        arg_dict = {"desc_with_padding": desc}
        extracted = extract_arg_info(arg_dict)

        arg_info = extracted["desc_with_padding"]
        self.assertEqual(arg_info["padding"], "nan")


class TestTensorDescriptorReconstruction(GPUTestBase):
    """Tests for TensorDescriptor reconstruction in reproducer utils."""

    def test_get_tensor_descriptor_type(self):
        """Test that _get_tensor_descriptor_type returns the correct type."""
        TensorDescriptor = _get_tensor_descriptor_type()
        self.assertIsNotNone(TensorDescriptor)
        self.assertEqual(TensorDescriptor.__name__, "TensorDescriptor")

    def test_create_arg_from_info_tensor_descriptor(self):
        """Test that _create_arg_from_info correctly reconstructs TensorDescriptor."""
        TensorDescriptor = _get_tensor_descriptor_type()
        if TensorDescriptor is None:
            self.skipTest("TensorDescriptor not available")

        arg_info = {
            "type": "TensorDescriptor",
            "base": {
                "type": "tensor",
                "dtype": "torch.float16",
                "shape": [128, 64],
                "device": "cuda:0",
            },
            "shape": [128, 64],
            "strides": [64, 1],
            "block_shape": [32, 32],
            "padding": "zero",
        }

        result = _create_arg_from_info(arg_info)

        self.assertIsInstance(result, TensorDescriptor)
        # TensorDescriptor stores shape as torch.Size (tuple-like), so compare as tuples
        self.assertEqual(tuple(result.shape), (128, 64))
        self.assertEqual(tuple(result.strides), (64, 1))
        self.assertEqual(result.block_shape, [32, 32])

    def test_roundtrip_tensor_descriptor(self):
        """Test full capture and reconstruction roundtrip for TensorDescriptor."""
        TensorDescriptor = _get_tensor_descriptor_type()
        if TensorDescriptor is None:
            self.skipTest("TensorDescriptor not available")

        original_tensor = torch.randn(
            256, 128, device=self.cuda_device, dtype=torch.float32
        )
        original_shape = (256, 128)
        original_strides = (128, 1)
        original_block_shape = [64, 32]
        original_desc = TensorDescriptor(
            original_tensor, original_shape, original_strides, original_block_shape
        )

        arg_dict = {"test_desc": original_desc}
        extracted = extract_arg_info(arg_dict)
        arg_info = extracted["test_desc"]

        reconstructed = _create_arg_from_info(arg_info)

        self.assertIsInstance(reconstructed, TensorDescriptor)
        # Compare as tuples since TensorDescriptor stores shape/strides as tuples
        self.assertEqual(tuple(reconstructed.shape), original_shape)
        self.assertEqual(tuple(reconstructed.strides), original_strides)
        self.assertEqual(reconstructed.block_shape, original_block_shape)
        self.assertEqual(reconstructed.base.shape, original_tensor.shape)

    def test_create_arg_from_info_missing_base_raises(self):
        """Test that missing base tensor info raises RuntimeError."""
        arg_info = {
            "type": "TensorDescriptor",
            "shape": [128, 64],
            "strides": [64, 1],
            "block_shape": [32, 32],
        }

        with self.assertRaises(RuntimeError) as cm:
            _create_arg_from_info(arg_info)

        self.assertIn("base", str(cm.exception))


class TestTensorDescriptorMultipleArgs(GPUTestBase):
    """Tests for handling multiple TensorDescriptor arguments."""

    def test_multiple_tensor_descriptors(self):
        """Test capture and reconstruction of multiple TensorDescriptor args."""
        TensorDescriptor = _get_tensor_descriptor_type()
        if TensorDescriptor is None:
            self.skipTest("TensorDescriptor not available")

        a = torch.randn(128, 64, device=self.cuda_device)
        b = torch.randn(64, 256, device=self.cuda_device)
        c = torch.randn(128, 256, device=self.cuda_device)

        a_desc = TensorDescriptor(a, a.shape, a.stride(), [128, 64])
        b_desc = TensorDescriptor(b, b.shape, b.stride(), [64, 256])
        c_desc = TensorDescriptor(c, c.shape, c.stride(), [128, 256])

        arg_dict = {
            "a_desc": a_desc,
            "b_desc": b_desc,
            "c_desc": c_desc,
            "M": 128,
            "N": 256,
            "K": 64,
        }

        extracted = extract_arg_info(arg_dict)

        self.assertEqual(extracted["a_desc"]["type"], "TensorDescriptor")
        self.assertEqual(extracted["b_desc"]["type"], "TensorDescriptor")
        self.assertEqual(extracted["c_desc"]["type"], "TensorDescriptor")
        self.assertEqual(extracted["M"]["type"], "int")
        self.assertEqual(extracted["N"]["type"], "int")
        self.assertEqual(extracted["K"]["type"], "int")

        for name in ["a_desc", "b_desc", "c_desc"]:
            reconstructed = _create_arg_from_info(extracted[name])
            self.assertIsInstance(reconstructed, TensorDescriptor)


if __name__ == "__main__":
    unittest.main()
