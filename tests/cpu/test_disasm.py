#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""CPU tests for tritonparse.tools.disasm availability checks.

`is_nvdisasm_available()` must reflect whether the nvdisasm binary can
actually run — not just whether triton's knob holds a path string. On
AMD/ROCm hosts the knob default can be set while the NVIDIA-only tool
is absent; reporting True there enables SASS dumping that silently
produces nothing and breaks tests asserting on SASS content.
"""

import os
import stat
import tempfile
import unittest
from unittest import mock

from tritonparse.tools import disasm


def _knobs_with_path(path):
    knobs = mock.MagicMock()
    knobs.nvidia.nvdisasm.path = path
    return knobs


class TestIsNvdisasmAvailable(unittest.TestCase):
    def test_empty_path_is_unavailable(self):
        for path in ("", None):
            with (
                self.subTest(path=path),
                mock.patch("triton.knobs", _knobs_with_path(path)),
            ):
                self.assertFalse(disasm.is_nvdisasm_available())

    def test_missing_file_is_unavailable(self):
        with mock.patch(
            "triton.knobs",
            _knobs_with_path("/nonexistent-dir/nvdisasm"),
        ):
            self.assertFalse(disasm.is_nvdisasm_available())

    def test_non_executable_file_is_unavailable(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        with mock.patch("triton.knobs", _knobs_with_path(path)):
            self.assertFalse(disasm.is_nvdisasm_available())

    def test_executable_file_is_available(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        self.addCleanup(os.unlink, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        with mock.patch("triton.knobs", _knobs_with_path(path)):
            self.assertTrue(disasm.is_nvdisasm_available())

    def test_bare_name_resolved_via_path(self):
        with (
            mock.patch("triton.knobs", _knobs_with_path("nvdisasm")),
            mock.patch("shutil.which", return_value=None) as which,
        ):
            self.assertFalse(disasm.is_nvdisasm_available())
            which.assert_called_once_with("nvdisasm")

    def test_knob_lookup_error_is_unavailable(self):
        knobs = mock.MagicMock()
        type(knobs).nvidia = mock.PropertyMock(side_effect=RuntimeError("no knobs"))
        with mock.patch("triton.knobs", knobs):
            self.assertFalse(disasm.is_nvdisasm_available())


if __name__ == "__main__":
    unittest.main()
