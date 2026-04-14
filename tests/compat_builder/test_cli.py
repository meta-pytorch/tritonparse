# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for compat_builder.cli module."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tritonparse.compat_builder.cli import (
    _add_compat_build_args,
    _handle_status,
    _handle_verify,
    _validate_compat_build_args,
    compat_build_command,
)
from tritonparse.compat_builder.state import CompatBuildPhase, CompatBuildState


def _make_parser() -> argparse.ArgumentParser:
    """Create a parser with compat-build args for testing."""
    parser = argparse.ArgumentParser()
    _add_compat_build_args(parser)
    return parser


def _parse(argv: list[str]) -> argparse.Namespace:
    """Parse argv through the compat-build parser."""
    parser = _make_parser()
    return parser.parse_args(argv)


class ValidateArgsTest(unittest.TestCase):
    """Test argument validation logic."""

    def test_build_mode_requires_triton_dir(self) -> None:
        args = _parse(["--llvm-bump-commit", "abc123"])
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            _validate_compat_build_args(args, parser)

    def test_build_mode_requires_llvm_bump_commit(self) -> None:
        args = _parse(["--triton-dir", "/fake"])
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            _validate_compat_build_args(args, parser)

    def test_build_mode_valid(self) -> None:
        args = _parse(["--triton-dir", "/fake", "--llvm-bump-commit", "abc123"])
        parser = _make_parser()
        # Should not raise
        _validate_compat_build_args(args, parser)

    def test_resume_requires_state(self) -> None:
        args = _parse(["--resume", "--fix-commit", "abc123"])
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            _validate_compat_build_args(args, parser)

    def test_resume_requires_fix_commit(self) -> None:
        args = _parse(["--resume", "--state", "/fake/state.json"])
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            _validate_compat_build_args(args, parser)

    def test_resume_valid(self) -> None:
        args = _parse(
            ["--resume", "--state", "/fake/state.json", "--fix-commit", "abc123"]
        )
        parser = _make_parser()
        _validate_compat_build_args(args, parser)

    def test_verify_requires_csv(self) -> None:
        args = _parse(["--verify"])
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            _validate_compat_build_args(args, parser)

    def test_verify_valid(self) -> None:
        args = _parse(["--verify", "--csv", "/fake/commits.csv"])
        parser = _make_parser()
        _validate_compat_build_args(args, parser)

    def test_status_no_extra_args_needed(self) -> None:
        args = _parse(["--status"])
        parser = _make_parser()
        _validate_compat_build_args(args, parser)

    def test_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            _parse(["--resume", "--verify"])


class CommandRoutingTest(unittest.TestCase):
    """Test that compat_build_command routes to the correct handler."""

    @patch("tritonparse.compat_builder.cli._handle_status")
    def test_routes_to_status(self, mock_status: MagicMock) -> None:
        mock_status.return_value = 0
        args = _parse(["--status"])
        compat_build_command(args)
        mock_status.assert_called_once_with(args)

    @patch("tritonparse.compat_builder.cli._handle_verify")
    def test_routes_to_verify(self, mock_verify: MagicMock) -> None:
        mock_verify.return_value = 0
        args = _parse(["--verify", "--csv", "/fake.csv"])
        compat_build_command(args)
        mock_verify.assert_called_once_with(args)

    @patch("tritonparse.compat_builder.cli._handle_resume")
    def test_routes_to_resume(self, mock_resume: MagicMock) -> None:
        mock_resume.return_value = 0
        args = _parse(
            ["--resume", "--state", "/fake/state.json", "--fix-commit", "abc"]
        )
        compat_build_command(args)
        mock_resume.assert_called_once_with(args)

    @patch("tritonparse.compat_builder.cli._handle_build")
    def test_routes_to_build(self, mock_build: MagicMock) -> None:
        mock_build.return_value = 0
        args = _parse(["--triton-dir", "/fake", "--llvm-bump-commit", "abc123"])
        compat_build_command(args)
        mock_build.assert_called_once_with(args)


class HandleStatusTest(unittest.TestCase):
    """Test _handle_status handler."""

    def test_status_no_state_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _parse(["--status", "--log-dir", tmpdir])
            result = _handle_status(args)
            self.assertEqual(result, 1)

    def test_status_with_state_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = CompatBuildState(
                triton_dir="/fake/triton",
                llvm_bump_commit="abc123def456",
                output_csv="./commits.csv",
                old_llvm="111111111111",
                new_llvm="999999999999",
                current_triton="abc123def456~1",
                phase=CompatBuildPhase.WAITING_FOR_FIX,
                worktree_path="/fake/worktree",
            )
            state.pairs = [("t1", "l1"), ("t2", "l2")]
            state_path = state.save(path=Path(tmpdir) / "test_state.json")

            args = _parse(["--status", "--state", str(state_path)])
            result = _handle_status(args)
            self.assertEqual(result, 0)


class HandleVerifyTest(unittest.TestCase):
    """Test _handle_verify handler."""

    def test_verify_valid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "commits.csv"
            csv_path.write_text(
                "# schema_version=1\n"
                "# llvm_bump_commit=abc123\n"
                "# old_llvm=111111\n"
                "# new_llvm=999999\n"
                "# final_bad_triton_commit=abc123\n"
                "# final_bad_llvm=999999\n"
                "triton_commit,llvm_commit_last_compatible\n"
                "t1,l1\n"
            )
            args = _parse(["--verify", "--csv", str(csv_path)])
            result = _handle_verify(args)
            self.assertEqual(result, 0)

    def test_verify_invalid_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "commits.csv"
            # Missing required metadata
            csv_path.write_text(
                "# schema_version=1\ntriton_commit,llvm_commit_last_compatible\nt1,l1\n"
            )
            args = _parse(["--verify", "--csv", str(csv_path)])
            result = _handle_verify(args)
            self.assertEqual(result, 1)

    def test_verify_missing_file(self) -> None:
        args = _parse(["--verify", "--csv", "/nonexistent/commits.csv"])
        result = _handle_verify(args)
        self.assertEqual(result, 1)


class DefaultArgValuesTest(unittest.TestCase):
    """Test default argument values."""

    def test_ai_enabled_by_default(self) -> None:
        args = _parse(["--triton-dir", "/fake", "--llvm-bump-commit", "abc"])
        self.assertTrue(args.use_ai)

    def test_no_ai_flag(self) -> None:
        args = _parse(["--triton-dir", "/fake", "--llvm-bump-commit", "abc", "--no-ai"])
        self.assertFalse(args.use_ai)

    def test_default_output_csv(self) -> None:
        args = _parse(["--triton-dir", "/fake", "--llvm-bump-commit", "abc"])
        self.assertEqual(args.output_csv, "./commits.csv")

    def test_default_conda_env(self) -> None:
        args = _parse(["--triton-dir", "/fake", "--llvm-bump-commit", "abc"])
        self.assertEqual(args.conda_env, "triton_bisect")

    def test_tui_enabled_by_default(self) -> None:
        args = _parse(["--triton-dir", "/fake", "--llvm-bump-commit", "abc"])
        self.assertTrue(args.tui)

    def test_no_tui_flag(self) -> None:
        args = _parse(
            ["--triton-dir", "/fake", "--llvm-bump-commit", "abc", "--no-tui"]
        )
        self.assertFalse(args.tui)
