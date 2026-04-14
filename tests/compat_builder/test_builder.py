# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for tritonparse.compat_builder.builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from tritonparse.compat_builder.builder import CompatBuilder, WaitingForFixError
from tritonparse.compat_builder.state import CompatBuildPhase, CompatBuildState


def _make_builder(
    tmp_dir: str,
    ai_fixer_factory: object | None = None,
) -> CompatBuilder:
    """Return a CompatBuilder with an injected pre-built state (no real git/shell)."""
    logger = MagicMock()
    builder = CompatBuilder(
        triton_dir="/tmp/triton",
        llvm_bump_commit="bumpabc123def456",
        output_csv=str(Path(tmp_dir) / "commits.csv"),
        logger=logger,
        ai_fixer_factory=ai_fixer_factory,
    )
    builder.state = CompatBuildState(
        triton_dir="/tmp/triton",
        llvm_bump_commit="bumpabc123def456",
        output_csv=str(Path(tmp_dir) / "commits.csv"),
        log_dir=tmp_dir,
        worktree_path="/tmp/worktree",
        old_llvm="a" * 40,
        new_llvm="b" * 40,
        current_triton="c" * 40,
        current_llvm_good="d" * 40,
        phase=CompatBuildPhase.FINDING_INCOMPATIBLE,
    )
    return builder


class WaitingForFixErrorTest(unittest.TestCase):
    def test_attributes(self) -> None:
        path = Path("/tmp/state.json")
        err = WaitingForFixError(
            state_path=path,
            incompatible_llvm="abc123def456789abc",
            build_error="error: undefined symbol",
        )
        self.assertEqual(err.state_path, path)
        self.assertEqual(err.incompatible_llvm, "abc123def456789abc")
        self.assertEqual(err.build_error, "error: undefined symbol")

    def test_message_contains_short_hash_and_path(self) -> None:
        path = Path("/tmp/state.json")
        err = WaitingForFixError(
            state_path=path,
            incompatible_llvm="abc123def456789abc",
        )
        self.assertIn("abc123def456", str(err))
        self.assertIn(str(path), str(err))

    def test_build_error_defaults_to_none(self) -> None:
        err = WaitingForFixError(
            state_path=Path("/tmp/s.json"),
            incompatible_llvm="deadbeef1234abcd",
        )
        self.assertIsNone(err.build_error)

    def test_is_exception(self) -> None:
        err = WaitingForFixError(
            state_path=Path("/tmp/s.json"),
            incompatible_llvm="deadbeef1234abcd",
        )
        self.assertIsInstance(err, Exception)


class CompatBuilderRequireStateTest(unittest.TestCase):
    def test_raises_before_initialize(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/t",
            llvm_bump_commit="abc",
            output_csv="/tmp/o.csv",
            logger=logger,
        )
        with self.assertRaises(RuntimeError):
            builder._require_state()


class CompatBuilderRecordPairTest(unittest.TestCase):
    def test_record_pair_appends_current_triton_and_llvm_good(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            state = builder.state
            assert state is not None
            expected_triton = state.current_triton
            expected_llvm = state.current_llvm_good

            builder.record_pair()

            self.assertEqual(len(state.pairs), 1)
            self.assertEqual(state.pairs[0], (expected_triton, expected_llvm))

    def test_record_pair_multiple_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            state = builder.state
            assert state is not None
            builder.record_pair()
            state.current_triton = "e" * 40
            state.current_llvm_good = "f" * 40
            builder.record_pair()

            self.assertEqual(len(state.pairs), 2)
            self.assertEqual(state.pairs[1][0], "e" * 40)

    def test_record_pair_without_state_raises(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/t",
            llvm_bump_commit="abc",
            output_csv="/tmp/o.csv",
            logger=logger,
        )
        with self.assertRaises(RuntimeError):
            builder.record_pair()


class CompatBuilderApplyFixTest(unittest.TestCase):
    def test_apply_fix_advances_current_triton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.apply_fix("fix_commit_abc123")
            self.assertEqual(builder.state.current_triton, "fix_commit_abc123")

    def test_apply_fix_resets_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.state.ai_fix_attempted = True
            builder.state.last_incompatible_llvm = "bad_llvm"
            builder.state.last_build_error = "some error"

            builder.apply_fix("fix_commit_abc123")

            self.assertFalse(builder.state.ai_fix_attempted)
            self.assertIsNone(builder.state.last_incompatible_llvm)
            self.assertIsNone(builder.state.last_build_error)

    def test_apply_fix_sets_phase_to_finding_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.state.phase = CompatBuildPhase.WAITING_FOR_FIX
            builder.apply_fix("fix_commit_abc123")
            self.assertEqual(builder.state.phase, CompatBuildPhase.FINDING_INCOMPATIBLE)


class CompatBuilderIsCompleteTest(unittest.TestCase):
    def test_not_complete_when_good_differs_from_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            # good="d"*40, new="b"*40 — different
            self.assertFalse(builder.is_complete())

    def test_complete_when_good_equals_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.state.current_llvm_good = builder.state.new_llvm
            self.assertTrue(builder.is_complete())

    def test_not_complete_when_new_llvm_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.state.new_llvm = None
            self.assertFalse(builder.is_complete())

    def test_raises_without_state(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/t",
            llvm_bump_commit="abc",
            output_csv="/tmp/o.csv",
            logger=logger,
        )
        with self.assertRaises(RuntimeError):
            builder.is_complete()


class CompatBuilderGenerateCsvTest(unittest.TestCase):
    def test_csv_contains_metadata_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            csv_path = builder.generate_csv()

            content = csv_path.read_text()
            self.assertIn("# schema_version=", content)
            self.assertIn("# llvm_bump_commit=", content)
            self.assertIn("# old_llvm=", content)
            self.assertIn("# new_llvm=", content)
            self.assertIn("# final_bad_triton_commit=", content)
            self.assertIn("# final_bad_llvm=", content)

    def test_csv_contains_column_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            csv_path = builder.generate_csv()
            content = csv_path.read_text()
            self.assertIn("triton_commit,llvm_commit_last_compatible", content)

    def test_csv_data_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.state.add_pair("triton_aaa", "llvm_aaa")
            builder.state.add_pair("triton_bbb", "llvm_bbb")

            csv_path = builder.generate_csv()
            content = csv_path.read_text()
            self.assertIn("triton_aaa,llvm_aaa", content)
            self.assertIn("triton_bbb,llvm_bbb", content)

    def test_csv_phase_set_to_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.generate_csv()
            self.assertEqual(builder.state.phase, CompatBuildPhase.COMPLETED)

    def test_csv_ends_with_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            csv_path = builder.generate_csv()
            content = csv_path.read_text()
            self.assertTrue(content.endswith("\n"))

    def test_csv_empty_pairs_has_no_data_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            csv_path = builder.generate_csv()
            content = csv_path.read_text()
            data_rows = [
                line
                for line in content.splitlines()
                if line and not line.startswith("#") and "triton_commit" not in line
            ]
            self.assertEqual(data_rows, [])

    def test_csv_metadata_values_match_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            state = builder.state
            csv_path = builder.generate_csv()
            content = csv_path.read_text()
            self.assertIn(f"# llvm_bump_commit={state.llvm_bump_commit}", content)
            self.assertIn(f"# old_llvm={state.old_llvm}", content)
            self.assertIn(f"# new_llvm={state.new_llvm}", content)


class CompatBuilderFixIncompatibilityTest(unittest.TestCase):
    def test_raises_waiting_for_fix_without_ai_fixer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp, ai_fixer_factory=None)
            with self.assertRaises(WaitingForFixError) as cm:
                builder.fix_incompatibility("bad_llvm_abc")
            self.assertEqual(cm.exception.incompatible_llvm, "bad_llvm_abc")

    def test_waiting_for_fix_sets_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp, ai_fixer_factory=None)
            try:
                builder.fix_incompatibility("bad_llvm_abc")
            except WaitingForFixError:
                pass
            self.assertEqual(builder.state.phase, CompatBuildPhase.WAITING_FOR_FIX)

    def test_waiting_for_fix_error_carries_state_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp, ai_fixer_factory=None)
            with self.assertRaises(WaitingForFixError) as cm:
                builder.fix_incompatibility("bad_llvm_abc")
            self.assertIsNotNone(cm.exception.state_path)
            self.assertTrue(cm.exception.state_path.exists())

    def test_ai_fixer_success_returns_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ai_fixer = MagicMock()
            ai_fixer.attempt_fix.return_value = "fix_commit_xyz"
            builder = _make_builder(tmp, ai_fixer_factory=lambda wt: ai_fixer)

            result = builder.fix_incompatibility("bad_llvm_abc")

            self.assertEqual(result, "fix_commit_xyz")
            self.assertEqual(builder.state.current_triton, "fix_commit_xyz")

    def test_ai_fixer_success_sets_phase_to_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ai_fixer = MagicMock()
            ai_fixer.attempt_fix.return_value = "fix_commit_xyz"
            builder = _make_builder(tmp, ai_fixer_factory=lambda wt: ai_fixer)
            builder.fix_incompatibility("bad_llvm_abc")
            self.assertEqual(builder.state.phase, CompatBuildPhase.FINDING_INCOMPATIBLE)

    def test_ai_fixer_returns_none_falls_back_to_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ai_fixer = MagicMock()
            ai_fixer.attempt_fix.return_value = None
            builder = _make_builder(tmp, ai_fixer_factory=lambda wt: ai_fixer)
            with self.assertRaises(WaitingForFixError):
                builder.fix_incompatibility("bad_llvm_abc")

    def test_ai_fixer_exception_falls_back_to_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ai_fixer = MagicMock()
            ai_fixer.attempt_fix.side_effect = RuntimeError("AI crashed")
            builder = _make_builder(tmp, ai_fixer_factory=lambda wt: ai_fixer)
            with self.assertRaises(WaitingForFixError):
                builder.fix_incompatibility("bad_llvm_abc")

    def test_ai_fixer_skipped_when_already_attempted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ai_fixer = MagicMock()
            ai_fixer.attempt_fix.return_value = "some_commit"
            builder = _make_builder(tmp, ai_fixer_factory=lambda wt: ai_fixer)
            builder.state.ai_fix_attempted = True

            with self.assertRaises(WaitingForFixError):
                builder.fix_incompatibility("bad_llvm_abc")

            ai_fixer.attempt_fix.assert_not_called()

    def test_ai_fixer_sets_attempted_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ai_fixer = MagicMock()
            ai_fixer.attempt_fix.return_value = None
            builder = _make_builder(tmp, ai_fixer_factory=lambda wt: ai_fixer)
            try:
                builder.fix_incompatibility("bad_llvm_abc")
            except WaitingForFixError:
                pass
            self.assertTrue(builder.state.ai_fix_attempted)


class CompatBuilderParseFirstBadTest(unittest.TestCase):
    def _make_builder_no_state(self) -> CompatBuilder:
        return CompatBuilder(
            triton_dir="/tmp/t",
            llvm_bump_commit="abc",
            output_csv="/tmp/o.csv",
            logger=MagicMock(),
        )

    def test_parses_standard_git_bisect_output(self) -> None:
        builder = self._make_builder_no_state()
        output = (
            "Bisecting: 3 revisions left to test after this (roughly 2 steps)\n"
            "abc123def456789abc12 is the first bad commit\n"
            "commit abc123def456789abc12\n"
            "Author: Test User <test@example.com>\n"
        )
        result = builder._parse_first_bad(output)
        self.assertEqual(result, "abc123def456789abc12")

    def test_returns_none_when_no_first_bad_line(self) -> None:
        builder = self._make_builder_no_state()
        result = builder._parse_first_bad("Bisecting: all commits are good\n")
        self.assertIsNone(result)

    def test_returns_none_for_empty_output(self) -> None:
        builder = self._make_builder_no_state()
        result = builder._parse_first_bad("")
        self.assertIsNone(result)

    def test_returns_none_for_partial_match(self) -> None:
        builder = self._make_builder_no_state()
        # "is the first bad" must have 5 tokens including hash
        result = builder._parse_first_bad("is the first bad commit\n")
        # No hash prefix — line split has only 5 parts, parts[0] is "is"
        # The condition requires parts[1:5] == ["is", "the", "first", "bad"]
        # "is the first bad commit" → parts = ["is", "the", "first", "bad", "commit"]
        # parts[1:5] = ["the", "first", "bad", "commit"] ≠ ["is", "the", "first", "bad"]
        self.assertIsNone(result)


class CompatBuilderExtractBuildErrorTest(unittest.TestCase):
    def _make_builder_no_state(self) -> CompatBuilder:
        return CompatBuilder(
            triton_dir="/tmp/t",
            llvm_bump_commit="abc",
            output_csv="/tmp/o.csv",
            logger=MagicMock(),
        )

    def test_extracts_error_lines(self) -> None:
        builder = self._make_builder_no_state()
        output = "Building...\nerror: undefined reference to 'foo'\nDone\n"
        result = builder._extract_build_error(output)
        self.assertIsNotNone(result)
        self.assertIn("error:", result)

    def test_returns_none_for_empty_output(self) -> None:
        builder = self._make_builder_no_state()
        result = builder._extract_build_error("")
        self.assertIsNone(result)

    def test_falls_back_to_last_lines_when_no_error_keyword(self) -> None:
        builder = self._make_builder_no_state()
        lines = [f"line {i}" for i in range(50)]
        output = "\n".join(lines)
        result = builder._extract_build_error(output)
        self.assertIsNotNone(result)
        self.assertIn("line 49", result)

    def test_returns_full_output_when_short(self) -> None:
        builder = self._make_builder_no_state()
        output = "short output\nonly a few lines"
        result = builder._extract_build_error(output)
        self.assertEqual(result, output)

    def test_limits_error_lines_to_twenty(self) -> None:
        builder = self._make_builder_no_state()
        lines = [f"error: issue {i}" for i in range(30)]
        output = "\n".join(lines)
        result = builder._extract_build_error(output)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLessEqual(len(result.splitlines()), 20)


class CompatBuilderResolveWorktreePathTest(unittest.TestCase):
    def test_uses_explicit_worktree_path(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/triton",
            llvm_bump_commit="abc123def456",
            output_csv="/tmp/o.csv",
            logger=logger,
            worktree_path="/explicit/wt/path",
        )
        result = builder._resolve_worktree_path()
        self.assertEqual(result, "/explicit/wt/path")

    def test_uses_worktree_root_when_no_explicit_path(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/triton",
            llvm_bump_commit="abc123def456abcd",
            output_csv="/tmp/o.csv",
            logger=logger,
            worktree_root="/my/root",
        )
        result = builder._resolve_worktree_path()
        # Should use first 8 chars of llvm_bump_commit
        self.assertIn("abc123de", result)
        self.assertTrue(result.startswith("/my/root"))

    def test_defaults_to_triton_dir_compat_worktrees(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/triton",
            llvm_bump_commit="abc123def456abcd",
            output_csv="/tmp/o.csv",
            logger=logger,
        )
        result = builder._resolve_worktree_path()
        self.assertIn(".compat_worktrees", result)
        self.assertTrue(result.startswith("/tmp/triton"))


class CompatBuilderUnshallowLlvmRepoTest(unittest.TestCase):
    """Test that _ensure_llvm_repo unshallows the LLVM clone."""

    def test_calls_unshallow_when_repo_is_shallow(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/triton",
            llvm_bump_commit="abc",
            output_csv="/tmp/o.csv",
            logger=logger,
        )
        executor = MagicMock()
        builder.executor = executor

        # make dev-install-llvm succeeds
        executor.run_command_streaming.return_value = MagicMock(
            success=True, stdout="ok"
        )

        # git rev-parse --is-shallow-repository returns "true"
        def _run_command_side_effect(
            cmd: list[str], cwd: str = "", **kwargs: object
        ) -> MagicMock:
            key = " ".join(cmd)
            if "--is-shallow-repository" in key:
                return MagicMock(success=True, stdout="true\n")
            if "fetch" in key and "--unshallow" in key:
                return MagicMock(success=True, stdout="", stderr="")
            return MagicMock(success=True, stdout="")

        executor.run_command.side_effect = _run_command_side_effect

        with tempfile.TemporaryDirectory() as tmp:
            llvm_dir = Path(tmp) / "llvm-project"
            # _ensure_llvm_repo checks llvm_dir.exists(); since it doesn't exist
            # it will run make dev-install-llvm first.
            # But llvm_dir still won't exist after that mock, so create it.
            llvm_dir.mkdir()
            builder._ensure_llvm_repo("abc123", tmp)

        unshallow_calls = [
            c
            for c in executor.run_command.call_args_list
            if "--unshallow" in " ".join(c[0][0])
        ]
        self.assertEqual(len(unshallow_calls), 1)

    def test_skips_unshallow_when_not_shallow(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/triton",
            llvm_bump_commit="abc",
            output_csv="/tmp/o.csv",
            logger=logger,
        )
        executor = MagicMock()
        builder.executor = executor

        def _run_command_side_effect(
            cmd: list[str], cwd: str = "", **kwargs: object
        ) -> MagicMock:
            key = " ".join(cmd)
            if "--is-shallow-repository" in key:
                return MagicMock(success=True, stdout="false\n")
            return MagicMock(success=True, stdout="")

        executor.run_command.side_effect = _run_command_side_effect

        with tempfile.TemporaryDirectory() as tmp:
            llvm_dir = Path(tmp) / "llvm-project"
            llvm_dir.mkdir()
            builder._ensure_llvm_repo("abc123", tmp)

        unshallow_calls = [
            c
            for c in executor.run_command.call_args_list
            if "--unshallow" in " ".join(c[0][0])
        ]
        self.assertEqual(len(unshallow_calls), 0)

    def test_raises_when_unshallow_fails(self) -> None:
        logger = MagicMock()
        builder = CompatBuilder(
            triton_dir="/tmp/triton",
            llvm_bump_commit="abc",
            output_csv="/tmp/o.csv",
            logger=logger,
        )
        executor = MagicMock()
        builder.executor = executor

        def _run_command_side_effect(
            cmd: list[str], cwd: str = "", **kwargs: object
        ) -> MagicMock:
            key = " ".join(cmd)
            if "--is-shallow-repository" in key:
                return MagicMock(success=True, stdout="true\n")
            if "fetch" in key and "--unshallow" in key:
                return MagicMock(success=False, stdout="", stderr="network error")
            return MagicMock(success=True, stdout="")

        executor.run_command.side_effect = _run_command_side_effect

        with tempfile.TemporaryDirectory() as tmp:
            llvm_dir = Path(tmp) / "llvm-project"
            llvm_dir.mkdir()
            with self.assertRaises(RuntimeError) as cm:
                builder._ensure_llvm_repo("abc123", tmp)
            self.assertIn("unshallow", str(cm.exception))


class CompatBuilderFindNextIncompatibleErrorTest(unittest.TestCase):
    """Test that find_next_incompatible raises on bisect failure."""

    def test_raises_when_bisect_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.conda_prefix = "/fake/conda/env"
            executor = MagicMock()
            builder.executor = executor

            # Pre-probe must fail so bisect path is reached
            executor.run_command_streaming.return_value = MagicMock(success=False)
            executor.run_command.return_value = MagicMock(
                success=True, stdout="fake_head\n"
            )
            executor.run_git_bisect_sequence.return_value = MagicMock(
                success=False,
                exit_code=128,
                stdout="",
                stderr="fatal: Bad rev input (not a commit): abc123",
                output="fatal: Bad rev input (not a commit): abc123",
            )

            with self.assertRaises(RuntimeError) as cm:
                builder.find_next_incompatible()
            self.assertIn("git bisect failed", str(cm.exception))
            self.assertIn("128", str(cm.exception))

    def test_bisect_failure_with_first_bad_still_returns_commit(self) -> None:
        """If bisect exits non-zero but contains first-bad line, return it."""
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            builder.conda_prefix = "/fake/conda/env"
            executor = MagicMock()
            builder.executor = executor

            # Pre-probe must fail so bisect path is reached
            executor.run_command_streaming.return_value = MagicMock(success=False)
            executor.run_command.return_value = MagicMock(
                success=True, stdout="prev_commit_hash\n"
            )
            executor.run_git_bisect_sequence.return_value = MagicMock(
                success=False,
                exit_code=1,
                stdout="abc123def456789abc12 is the first bad commit\n",
                stderr="",
                output="abc123def456789abc12 is the first bad commit\n",
            )
            executor.run_command.return_value = MagicMock(
                success=True, stdout="prev_commit_hash\n"
            )

            result = builder.find_next_incompatible()
            self.assertEqual(result, "abc123def456789abc12")
