# Copyright (c) Meta Platforms, Inc. and affiliates.

"""CLI Integration Tests for --auto-env-setup with --llvm-only bisect.

These tests verify that the --auto-env-setup CLI option is correctly integrated
into the main tritonparse bisect CLI for automatic environment setup.
"""

import argparse
import unittest
from unittest.mock import patch

from tritonparse.bisect.cli import _add_bisect_args, bisect_command


class CLIArgumentParsingTest(unittest.TestCase):
    """Tests for CLI argument parsing."""

    def setUp(self):
        """Set up test parser."""
        self.parser = argparse.ArgumentParser()
        _add_bisect_args(self.parser)

    def test_auto_env_setup_option_exists(self):
        """Test that --auto-env-setup option is available."""
        args = self.parser.parse_args(
            [
                "--llvm-only",
                "--auto-env-setup",
                "--triton-dir",
                "/tmp/oss-triton",
                "--good-llvm",
                "def456",
                "--bad-llvm",
                "ghi789",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertTrue(args.auto_env_setup)
        self.assertTrue(args.llvm_only)

    def test_auto_env_setup_requires_llvm_only(self):
        """Test that --auto-env-setup is intended for use with --llvm-only."""
        # --auto-env-setup can be parsed without --llvm-only, but the help text
        # indicates it should be used with --llvm-only. The validation happens
        # at runtime in _handle_llvm_only.
        args = self.parser.parse_args(["--auto-env-setup"])
        self.assertTrue(args.auto_env_setup)

    def test_llvm_only_with_auto_env_setup_and_triton_dir(self):
        """Test --llvm-only with --auto-env-setup and --triton-dir."""
        args = self.parser.parse_args(
            [
                "--llvm-only",
                "--auto-env-setup",
                "--triton-dir",
                "~/oss-triton",
                "--good-llvm",
                "abc123",
                "--bad-llvm",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertTrue(args.llvm_only)
        self.assertTrue(args.auto_env_setup)
        self.assertEqual(args.triton_dir, "~/oss-triton")
        self.assertEqual(args.good_llvm, "abc123")
        self.assertEqual(args.bad_llvm, "def456")

    def test_torch_dir_option_exists(self):
        """Test that --torch-dir is available and parsed."""
        args = self.parser.parse_args(
            [
                "--llvm-only",
                "--triton-dir",
                "/existing/triton",
                "--torch-dir",
                "/existing/pytorch",
                "--good-llvm",
                "abc123",
                "--bad-llvm",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertEqual(args.torch_dir, "/existing/pytorch")

    def test_target_option_exists(self):
        """Test that --target is available and parsed."""
        args = self.parser.parse_args(
            [
                "--target",
                "torch",
                "--torch-dir",
                "/existing/pytorch",
                "--good",
                "abc123",
                "--bad",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertEqual(args.target, "torch")

    def test_llvm_only_without_auto_env_setup(self):
        """Test --llvm-only can still be used without --auto-env-setup."""
        args = self.parser.parse_args(
            [
                "--llvm-only",
                "--triton-dir",
                "/existing/triton",
                "--good-llvm",
                "abc123",
                "--bad-llvm",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertTrue(args.llvm_only)
        self.assertFalse(args.auto_env_setup)

    def test_fb_llvm_bisect_option_removed(self):
        """Test that --fb-llvm-bisect option no longer exists."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--fb-llvm-bisect"])


class CLIAutoSetupIntegrationTest(unittest.TestCase):
    """Tests for --auto-env-setup integration with --llvm-only."""

    def setUp(self):
        """Set up test parser."""
        self.parser = argparse.ArgumentParser()
        _add_bisect_args(self.parser)

    def test_auto_env_setup_with_conda_env(self):
        """Test --auto-env-setup with custom --conda-env."""
        args = self.parser.parse_args(
            [
                "--llvm-only",
                "--auto-env-setup",
                "--triton-dir",
                "~/oss-triton",
                "--conda-env",
                "my_custom_env",
                "--good-llvm",
                "abc123",
                "--bad-llvm",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertEqual(args.conda_env, "my_custom_env")

    def test_default_conda_env(self):
        """Test default --conda-env value."""
        args = self.parser.parse_args(
            [
                "--llvm-only",
                "--triton-dir",
                "~/oss-triton",
                "--good-llvm",
                "abc123",
                "--bad-llvm",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertEqual(args.conda_env, "triton_bisect")

    def test_torch_dir_defaults_to_none(self):
        """Test default --torch-dir value."""
        args = self.parser.parse_args(
            [
                "--llvm-only",
                "--triton-dir",
                "~/oss-triton",
                "--good-llvm",
                "abc123",
                "--bad-llvm",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertIsNone(args.torch_dir)

    def test_target_defaults_to_triton(self):
        """Test default --target value."""
        args = self.parser.parse_args(
            [
                "--triton-dir",
                "~/oss-triton",
                "--good",
                "abc123",
                "--bad",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )
        self.assertEqual(args.target, "triton")


class CLIDispatchTest(unittest.TestCase):
    """Tests for bisect command dispatch."""

    def setUp(self):
        self.parser = argparse.ArgumentParser()
        _add_bisect_args(self.parser)

    def test_target_torch_dispatches_to_torch_handler(self):
        """--target torch with --torch-dir should use _handle_torch_bisect."""
        args = self.parser.parse_args(
            [
                "--target",
                "torch",
                "--torch-dir",
                "/existing/pytorch",
                "--good",
                "abc123",
                "--bad",
                "def456",
                "--test-script",
                "/tmp/test.py",
            ]
        )

        with patch(
            "tritonparse.bisect.cli._handle_torch_bisect", return_value=7
        ) as mock_torch:
            result = bisect_command(args)

        self.assertEqual(result, 7)
        mock_torch.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
