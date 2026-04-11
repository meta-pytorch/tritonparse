# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for tritonparse.bisect.torch_bisector module.

Tests cover:
- TorchBisector property accessors
- _log_header output
- _get_bisect_script returns a valid path
- _get_extra_env_vars is empty
- run() wraps BisectError as TorchBisectError
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tritonparse.bisect.torch_bisector import TorchBisectError, TorchBisector


class TorchBisectorPropertiesTest(unittest.TestCase):
    """Tests for TorchBisector property accessors."""

    def setUp(self) -> None:
        self.logger = MagicMock()
        self.logger.log_dir = Path("/tmp/logs")
        self.bisector = TorchBisector(
            torch_dir="/fake/pytorch",
            test_script="/fake/test.py",
            conda_env="test_env",
            logger=self.logger,
        )

    def test_bisect_name(self) -> None:
        self.assertEqual(self.bisector.bisect_name, "PyTorch Bisect")

    def test_default_build_command(self) -> None:
        self.assertTrue(self.bisector.build_command.startswith("bash "))
        self.assertTrue(self.bisector.build_command.endswith("build_pytorch.sh"))

    def test_target_repo_dir_is_torch_dir(self) -> None:
        self.assertEqual(
            self.bisector.target_repo_dir,
            Path("/fake/pytorch").resolve(),
        )

    def test_triton_dir_points_to_torch_dir(self) -> None:
        """triton_dir resolves to the PyTorch checkout passed as torch_dir."""
        self.assertEqual(
            self.bisector.triton_dir,
            Path("/fake/pytorch").resolve(),
        )

    def test_get_extra_env_vars_empty(self) -> None:
        self.assertEqual(self.bisector._get_extra_env_vars(), {})

    def test_get_bisect_script_returns_path(self) -> None:
        script = self.bisector._get_bisect_script()
        self.assertIsInstance(script, Path)
        self.assertTrue(str(script).endswith("bisect_torch.sh"))

    def test_custom_build_command(self) -> None:
        bisector = TorchBisector(
            torch_dir="/fake/pytorch",
            test_script="/fake/test.py",
            conda_env="test_env",
            logger=self.logger,
            build_command="python setup.py develop",
        )
        self.assertEqual(bisector.build_command, "python setup.py develop")


class TorchBisectorLogHeaderTest(unittest.TestCase):
    """Tests for TorchBisector._log_header."""

    def test_log_header_includes_pytorch_info(self) -> None:
        logger = MagicMock()
        logger.log_dir = Path("/tmp/logs")
        bisector = TorchBisector(
            torch_dir="/fake/pytorch",
            test_script="/fake/test.py",
            conda_env="test_env",
            logger=logger,
        )
        bisector._log_header("good123", "bad456")

        logged = " ".join(str(c) for c in logger.info.call_args_list)
        self.assertIn("PyTorch Bisect", logged)
        self.assertIn("good123", logged)
        self.assertIn("bad456", logged)
        self.assertIn("test_env", logged)


class TorchBisectorRunTest(unittest.TestCase):
    """Tests for TorchBisector.run error wrapping."""

    @patch.object(TorchBisector, "_run_bisect")
    def test_run_returns_culprit(self, mock_run: MagicMock) -> None:
        logger = MagicMock()
        logger.log_dir = Path("/tmp/logs")
        bisector = TorchBisector(
            torch_dir="/fake/pytorch",
            test_script="/fake/test.py",
            conda_env="test_env",
            logger=logger,
        )
        mock_run.return_value = "abc123deadbeef"
        result = bisector.run("good", "bad")
        self.assertEqual(result, "abc123deadbeef")

    @patch.object(TorchBisector, "_run_bisect")
    def test_run_wraps_bisect_error(self, mock_run: MagicMock) -> None:
        from tritonparse.bisect.base_bisector import BisectError

        logger = MagicMock()
        logger.log_dir = Path("/tmp/logs")
        bisector = TorchBisector(
            torch_dir="/fake/pytorch",
            test_script="/fake/test.py",
            conda_env="test_env",
            logger=logger,
        )
        mock_run.side_effect = BisectError("something failed")
        with self.assertRaises(TorchBisectError) as ctx:
            bisector.run("good", "bad")
        self.assertIn("something failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
