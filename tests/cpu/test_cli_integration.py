# Copyright (c) Meta Platforms, Inc. and affiliates.

"""CLI Integration Tests for --auto-env-setup with --llvm-only bisect.

These tests verify that the --auto-env-setup CLI option is correctly integrated
into the main tritonparse bisect CLI for automatic environment setup.
"""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tritonparse.bisect.cli import (
    _add_bisect_args,
    _handle_triton_bisect,
    _orchestrate_workflow,
    bisect_command,
)
from tritonparse.bisect.commit_detector import CommitDetectorError, LLVMBumpInfo
from tritonparse.bisect.llvm_bisector import LLVMBisectError
from tritonparse.bisect.result import BisectResult
from tritonparse.bisect.state import BisectPhase, BisectState
from tritonparse.bisect.triton_bisector import TritonBisectError


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


class LLVMDescriptorWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.ui = MagicMock()
        self.ui.is_tui_enabled = False
        self.ui._rich_enabled = False
        self.ui.progress.elapsed_seconds = 0
        self.logger = MagicMock()

    def test_descriptor_failure_is_not_success_after_finding_triton_commit(self):
        parser = argparse.ArgumentParser()
        _add_bisect_args(parser)
        args = parser.parse_args(
            [
                "--triton-dir",
                "/unused",
                "--test-script",
                "/unused/test.py",
                "--good",
                "a" * 40,
                "--bad",
                "b" * 40,
            ]
        )
        with (
            patch("tritonparse.bisect.cli._create_logger", return_value=self.logger),
            patch("tritonparse.bisect.ui.BisectUI", return_value=self.ui),
            patch("tritonparse.bisect.ui.print_final_summary") as summary,
            patch("tritonparse.bisect.triton_bisector.TritonBisector") as bisector,
            patch("tritonparse.bisect.commit_detector.CommitDetector") as detector,
        ):
            bisector.return_value.run.return_value = "b" * 40
            detector.return_value.detect.side_effect = CommitDetectorError(
                "invalid JSON"
            )
            self.assertEqual(_handle_triton_bisect(args), 1)
        self.assertIn("invalid JSON", summary.call_args.kwargs["error_msg"])

    def test_artifact_only_update_does_not_enter_llvm_pair_testing(self):
        state = BisectState(
            triton_dir="/unused",
            test_script="/unused/test.py",
            good_commit="a" * 40,
            bad_commit="b" * 40,
            phase=BisectPhase.TYPE_CHECK,
            triton_culprit="b" * 40,
        )
        info = LLVMBumpInfo(
            is_llvm_bump=False,
            old_hash="c" * 40,
            new_hash="c" * 40,
            artifact_changed=True,
        )
        with (
            patch.object(state, "save"),
            patch("tritonparse.bisect.ui.print_final_summary"),
            patch("tritonparse.bisect.commit_detector.CommitDetector") as detector,
            patch("tritonparse.bisect.pair_tester.PairTester") as tester,
        ):
            detector.return_value.detect.return_value = info
            self.assertEqual(_orchestrate_workflow(state, self.ui, self.logger), 0)
        self.assertEqual(state.phase, BisectPhase.COMPLETED)
        self.assertEqual(state.llvm_comparison, info.to_dict())
        tester.assert_not_called()


class BisectOutcomeWorkflowTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.log_dir = Path(temporary.name)
        self.ui = MagicMock()
        self.ui.is_tui_enabled = False
        self.ui._rich_enabled = False
        self.ui.progress.elapsed_seconds = 0
        self.logger = MagicMock()
        self.logger.log_dir = self.log_dir
        self.logger.module_log_path = self.log_dir / "module.log"
        self.logger.command_log_path = self.log_dir / "commands.log"
        self.logger.session_name = "workflow"

    def _state(self, **kwargs):
        return BisectState(
            triton_dir="/unused",
            test_script="/unused/test.py",
            good_commit="a" * 40,
            bad_commit="b" * 40,
            log_dir=str(self.log_dir),
            **kwargs,
        )

    def test_incomplete_triton_bisect_stops_before_llvm_detection(self):
        for status, phase in (
            ("ambiguous", BisectPhase.AMBIGUOUS),
            ("aborted", BisectPhase.ABORTED),
            ("error", BisectPhase.FAILED),
        ):
            with self.subTest(status=status):
                candidates = ["a" * 40, "b" * 40] if status == "ambiguous" else []
                result = BisectResult(
                    status=status,
                    candidates=candidates,
                    message=f"{status}\n" + "\n".join(candidates),
                )
                state = self._state()
                with (
                    patch("tritonparse.bisect.ui.print_final_summary") as summary,
                    patch(
                        "tritonparse.bisect.triton_bisector.TritonBisector"
                    ) as bisector,
                    patch(
                        "tritonparse.bisect.commit_detector.CommitDetector"
                    ) as detector,
                    patch(
                        "tritonparse.bisect.executor.ShellExecutor.run_command"
                    ) as command,
                ):
                    bisector.return_value.run.side_effect = TritonBisectError(
                        result.message, result=result
                    )
                    self.assertEqual(
                        _orchestrate_workflow(state, self.ui, self.logger), 1
                    )
                detector.assert_not_called()
                command.assert_not_called()
                self.assertEqual(state.phase, phase)
                self.assertIsNone(state.triton_culprit)
                self.assertIsNone(summary.call_args.kwargs["culprits"])
                self.assertEqual(state.triton_bisect_result, result.to_dict())
                restored = BisectState.load(self.log_dir / "workflow_state.json")
                self.assertEqual(restored.phase, phase)
                self.assertEqual(
                    restored.to_report()["triton_bisect_result"], result.to_dict()
                )

    def test_llvm_ambiguity_keeps_the_confirmed_triton_result(self):
        state = self._state(
            phase=BisectPhase.LLVM_BISECT,
            triton_culprit="c" * 40,
            is_llvm_bump=True,
            good_llvm="d" * 40,
            bad_llvm="e" * 40,
        )
        result = BisectResult(
            status="ambiguous",
            candidates=["d" * 40, "e" * 40],
            message="LLVM candidate set",
        )
        with (
            patch("tritonparse.bisect.ui.print_final_summary") as summary,
            patch("tritonparse.bisect.llvm_bisector.LLVMBisector") as bisector,
        ):
            bisector.return_value.run.side_effect = LLVMBisectError(
                result.message, result=result
            )
            self.assertEqual(_orchestrate_workflow(state, self.ui, self.logger), 1)
        self.assertEqual(state.phase, BisectPhase.AMBIGUOUS)
        self.assertIsNone(state.llvm_culprit)
        self.assertEqual(state.to_report()["llvm_bisect_result"], result.to_dict())
        self.assertEqual(summary.call_args.kwargs["culprits"], {"triton": "c" * 40})

    def test_unique_result_is_saved_before_llvm_detection(self):
        state = self._state()
        result = BisectResult(status="found", culprit="b" * 40, exit_code=0)
        with (
            patch("tritonparse.bisect.ui.print_final_summary"),
            patch("tritonparse.bisect.triton_bisector.TritonBisector") as bisector,
            patch("tritonparse.bisect.commit_detector.CommitDetector") as detector,
        ):
            bisector.return_value.run.return_value = result.culprit
            bisector.return_value.result = result
            detector.return_value.detect.return_value = LLVMBumpInfo(is_llvm_bump=False)
            self.assertEqual(_orchestrate_workflow(state, self.ui, self.logger), 0)
        detector.return_value.detect.assert_called_once_with(result.culprit)
        self.assertEqual(state.phase, BisectPhase.COMPLETED)
        self.assertEqual(state.triton_bisect_result, result.to_dict())

    def test_resuming_a_stopped_result_does_not_report_completion(self):
        for phase in (BisectPhase.AMBIGUOUS, BisectPhase.ABORTED, BisectPhase.FAILED):
            with self.subTest(phase=phase):
                state = self._state(phase=phase, error_message="prior stop reason")
                self.ui.reset_mock()
                with (
                    patch("tritonparse.bisect.ui.print_final_summary"),
                    patch(
                        "tritonparse.bisect.triton_bisector.TritonBisector"
                    ) as bisector,
                ):
                    self.assertEqual(
                        _orchestrate_workflow(state, self.ui, self.logger), 1
                    )
                bisector.assert_not_called()
                self.assertEqual(state.phase, phase)
                self.assertNotIn(
                    unittest.mock.call("Full Workflow Complete!"),
                    self.ui.append_output.call_args_list,
                )

    def test_single_mode_ambiguity_is_nonzero_without_a_culprit(self):
        parser = argparse.ArgumentParser()
        _add_bisect_args(parser)
        args = parser.parse_args(
            [
                "--triton-dir",
                "/unused",
                "--test-script",
                "/unused/test.py",
                "--good",
                "a" * 40,
                "--bad",
                "b" * 40,
            ]
        )
        result = BisectResult(
            status="ambiguous", candidates=["a" * 40, "b" * 40], message="candidate set"
        )
        with (
            patch("tritonparse.bisect.cli._create_logger", return_value=self.logger),
            patch("tritonparse.bisect.ui.BisectUI", return_value=self.ui),
            patch("tritonparse.bisect.ui.print_final_summary") as summary,
            patch("tritonparse.bisect.triton_bisector.TritonBisector") as bisector,
            patch("tritonparse.bisect.commit_detector.CommitDetector") as detector,
        ):
            bisector.return_value.run.side_effect = TritonBisectError(
                result.message, result=result
            )
            self.assertEqual(_handle_triton_bisect(args), 1)
        detector.assert_not_called()
        self.assertIsNone(summary.call_args.kwargs["culprits"])


if __name__ == "__main__":
    unittest.main()
