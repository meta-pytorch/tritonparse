#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""CPU tests for GPU-architecture helpers in tests.test_utils.

`is_amd_gpu()` / `is_nvidia_gpu()` drive backend-exclusive skips
(e.g. SASS assertions are NVIDIA-only, AMD e2e is AMD-only), so a
wrong answer here silently disables or breaks tests per-arch. torch
is mocked: no GPU needed.
"""

import unittest
from unittest import mock

from tests.test_utils import is_amd_gpu, is_nvidia_gpu


def _torch_mock(*, available, hip):
    torch_mock = mock.MagicMock()
    torch_mock.cuda.is_available.return_value = available
    torch_mock.version.hip = hip
    torch_mock.version.cuda = "12.8" if hip is None else None
    return torch_mock


class TestDeviceArchHelpers(unittest.TestCase):
    def test_rocm_build_with_gpu_is_amd(self):
        with mock.patch(
            "tests.test_utils.torch", _torch_mock(available=True, hip="7.1")
        ):
            self.assertTrue(is_amd_gpu())
            self.assertFalse(is_nvidia_gpu())

    def test_cuda_build_with_gpu_is_nvidia(self):
        with mock.patch(
            "tests.test_utils.torch", _torch_mock(available=True, hip=None)
        ):
            self.assertFalse(is_amd_gpu())
            self.assertTrue(is_nvidia_gpu())

    def test_no_gpu_is_neither(self):
        with mock.patch(
            "tests.test_utils.torch", _torch_mock(available=False, hip="7.1")
        ):
            self.assertFalse(is_amd_gpu())
            self.assertFalse(is_nvidia_gpu())

    def test_lookup_error_is_neither(self):
        torch_mock = mock.MagicMock()
        torch_mock.cuda.is_available.side_effect = RuntimeError("no driver")
        with mock.patch("tests.test_utils.torch", torch_mock):
            self.assertFalse(is_amd_gpu())
            self.assertFalse(is_nvidia_gpu())


if __name__ == "__main__":
    unittest.main()
