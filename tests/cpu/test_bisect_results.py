# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Regression tests for incomplete bisects and evidence retained after reset."""

import json
import signal
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from tritonparse.bisect import BisectLogger, BisectResult, ShellExecutor
from tritonparse.bisect.base_bisector import BisectError
from tritonparse.bisect.executor import ProcessCleanupError
from tritonparse.bisect.llvm_bisector import LLVMBisectError, LLVMBisector
from tritonparse.bisect.triton_bisector import TritonBisectError, TritonBisector


def candidate_output(*candidates: str) -> str:
    return (
        "There are only 'skip'ped commits left to test.\n"
        "The first bad commit could be any of:\n"
        + "\n".join(candidates)
        + "\nWe cannot bisect more!\n"
    )


class BisectResultParsingTest(unittest.TestCase):
    def test_unique_full_and_abbreviated_hashes(self):
        for length in (7, 12, 40):
            with self.subTest(length=length):
                commit = "a" * length
                result = BisectResult.from_output(f"{commit} is the first bad commit\n")
                self.assertEqual(result.status, "found")
                self.assertEqual(result.culprit, commit)
                self.assertEqual(result.candidates, [])

    def test_skip_candidates_survive_nonzero_git_exit(self):
        candidates = ["a" * 40, "b" * 40, "c" * 40]
        for exit_code in (0, 2):
            with self.subTest(exit_code=exit_code):
                result = BisectResult.from_output(
                    candidate_output(*candidates), exit_code
                )
                self.assertEqual(result.status, "ambiguous")
                self.assertIsNone(result.culprit)
                self.assertEqual(result.candidates, candidates)
                for candidate in candidates:
                    self.assertIn(candidate, result.message)

    def test_single_skip_candidate_is_still_not_a_confirmed_culprit(self):
        result = BisectResult.from_output(candidate_output("a" * 40), 2)
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.culprit)

    def test_ambiguity_takes_precedence_over_an_earlier_first_bad_line(self):
        result = BisectResult.from_output(
            f"{'c' * 40} is the first bad commit\n"
            + candidate_output("a" * 40, "b" * 40)
        )
        self.assertEqual(result.status, "ambiguous")
        self.assertEqual(result.candidates, ["a" * 40, "b" * 40])
        self.assertIsNone(result.culprit)

    def test_failed_command_does_not_promote_a_first_bad_line(self):
        result = BisectResult.from_output(f"{'a' * 40} is the first bad commit\n", 1)
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.culprit)

    def test_conflicting_or_unparseable_output_is_an_error(self):
        for output in (
            "",
            "a build completed",
            f"{'a' * 40} is the first bad commit\n{'b' * 40} is the first bad commit\n",
        ):
            with self.subTest(output=output):
                result = BisectResult.from_output(output)
                self.assertEqual(result.status, "error")
                self.assertIsNone(result.culprit)

    def test_executor_failure_is_distinct_from_an_abort(self):
        result = BisectResult.from_output("OSError: executable not found", -1)
        self.assertEqual(result.status, "error")
        result = BisectResult.from_output(
            "error: bisect run failed: exit code 128 from 'bash test.sh' "
            "is < 0 or >= 128",
            128,
        )
        self.assertEqual(result.status, "aborted")
        self.assertIsNone(result.culprit)

    def test_legacy_parser_raises_with_all_candidates(self):
        bisector = TritonBisector("/unused", "/unused/test.py", "unused", MagicMock())
        with self.assertRaises(BisectError) as error:
            bisector._parse_bisect_result(candidate_output("a" * 40, "b" * 40))
        self.assertEqual(error.exception.result.status, "ambiguous")
        self.assertEqual(error.exception.result.candidates, ["a" * 40, "b" * 40])

    def test_git_crash_signals_are_distinct_from_interrupt_signals(self):
        for signum, expected in (
            (signal.SIGINT, "aborted"),
            (signal.SIGTERM, "aborted"),
            (signal.SIGABRT, "error"),
            (signal.SIGSEGV, "error"),
            (signal.SIGKILL, "error"),
        ):
            with self.subTest(signum=signum):
                result = BisectResult.from_output("Git stopped", -signum)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.exit_code, -signum)
                self.assertIsNone(result.culprit)

    def test_llvm_wrapper_preserves_the_structured_error(self):
        bisector = LLVMBisector("/unused", "/unused/test.py", "unused", MagicMock())
        result = BisectResult.from_output(candidate_output("a" * 40, "b" * 40), 2)
        with (
            patch.object(bisector, "_get_triton_commit", return_value="c" * 40),
            patch.object(bisector, "_ensure_llvm_repo"),
            patch.object(bisector, "_get_next_commit", return_value="d" * 40),
            patch.object(bisector, "_checkout_triton"),
            patch.object(bisector, "_validate_commits"),
            patch.object(
                bisector,
                "_run_bisect",
                side_effect=BisectError(result.message, result=result),
            ),
            self.assertRaises(LLVMBisectError) as error,
        ):
            bisector.run("c" * 40, "d" * 40, "e" * 40)
        self.assertIs(error.exception.result, result)


class GitBisectResultTest(unittest.TestCase):
    """Exercise real Git histories with shell-only test scripts; no compiler/GPU."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.logger = BisectLogger(str(self.root / "logs"))
        for handler in self.logger.logger.handlers:
            self.addCleanup(handler.close)
        self.executor = ShellExecutor(self.logger)
        self._git("init", "-q")
        self._git("config", "user.name", "Tritonparse test")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "commit.gpgsign", "false")
        self.commits = []
        for version in range(5):
            (self.repo / "version").write_text(f"{version}\n")
            self._git("add", "version")
            self._git("commit", "-qm", f"version {version}")
            self.commits.append(self._git("rev-parse", "HEAD"))
        self.script = self.root / "test.sh"

    def _git(self, *args):
        result = self.executor.run_command(
            ["git", *args], cwd=str(self.repo), env={"LC_ALL": "C"}
        )
        self.assertTrue(result.success, result.output)
        return result.stdout.strip()

    def _run(self, body, *, bad=None, output_callback=None):
        self.script.write_text("set -eu\n" + body + "\n")
        self.bisector = TritonBisector(
            str(self.repo), str(self.script), "unused", self.logger
        )
        with patch.object(
            self.bisector, "_get_bisect_script", return_value=self.script
        ):
            return self.bisector.run(
                self.commits[0], bad or self.commits[-1], output_callback
            )

    def _assert_saved_and_reset(self, status):
        result = self.bisector.result
        self.assertEqual(result.status, status)
        saved = json.loads(Path(result.result_file).read_text())
        self.assertEqual(saved, result.to_dict())
        snapshot = Path(result.git_bisect_log).read_text()
        self.assertIn("git bisect start", snapshot)
        self.assertEqual(self._git("rev-parse", "HEAD"), self.commits[-1])
        start = Path(self._git("rev-parse", "--git-path", "BISECT_START"))
        if not start.is_absolute():
            start = self.repo / start
        self.assertFalse(start.exists())
        commands = Path(result.command_log).read_text()
        self.assertLess(
            commands.index("Command: git bisect log"),
            commands.index("Command: git bisect reset"),
        )
        return snapshot, commands

    def test_unique_result_keeps_replay_log_after_reset(self):
        culprit = self._run('read -r version < version\ntest "$version" -lt 2')
        self.assertEqual(culprit, self.commits[2])
        snapshot, _ = self._assert_saved_and_reset("found")
        self.assertIn(self.commits[2], snapshot)

    def test_skipped_boundary_retains_all_candidates_and_reason(self):
        with self.assertRaises(TritonBisectError) as error:
            self._run(
                "read -r version < version\n"
                'if [ "$version" -eq 2 ]; then\n'
                "  printf '%s\\n' 'skip: intentional unavailable revision'\n"
                "  exit 125\n"
                "fi\n"
                'test "$version" -lt 3'
            )
        result = error.exception.result
        self.assertIs(result, self.bisector.result)
        self.assertIsNone(result.culprit)
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(set(result.candidates), set(self.commits[2:4]))
        snapshot, commands = self._assert_saved_and_reset("ambiguous")
        self.assertIn(f"git bisect skip {self.commits[2]}", snapshot)
        self.assertIn("skip: intentional unavailable revision", commands)

    def test_fatal_test_exit_is_aborted(self):
        # Git itself aborts on test exits >= 128, including signal-style exits.
        # Its own exit code need not equal the test's (137 can become 119).
        for test_exit in (128, 137, 139):
            with self.subTest(test_exit=test_exit):
                with self.assertRaises(TritonBisectError) as error:
                    self._run(f"exit {test_exit}")
                self.assertEqual(error.exception.result.status, "aborted")
                self.assertIsNone(error.exception.result.culprit)
                self._assert_saved_and_reset("aborted")

    def test_interruption_stops_the_run_then_saves_and_resets(self):
        def interrupt(line):
            if line == "test started":
                raise KeyboardInterrupt

        with self.assertRaises(TritonBisectError) as error:
            self._run("printf 'test started\\n'\nsleep 30", output_callback=interrupt)
        self.assertEqual(error.exception.result.exit_code, 130)
        self._assert_saved_and_reset("aborted")

    def test_invalid_endpoint_is_error_with_a_saved_log(self):
        with self.assertRaises(TritonBisectError) as error:
            self._run("exit 0", bad="does-not-exist")
        self.assertEqual(error.exception.result.status, "error")
        self.assertIsNone(error.exception.result.culprit)
        self._assert_saved_and_reset("error")

    def test_failed_start_cleans_partial_state_and_allows_retry(self):
        # Git writes BISECT_START before attempting to unlink BISECT_LOG.
        # A directory at that path makes the real start command fail midway.
        blocker = self.repo / ".git" / "BISECT_LOG"
        blocker.mkdir()
        with self.assertRaises(TritonBisectError) as error:
            self._run("exit 0")
        self.assertEqual(error.exception.result.status, "error")
        self.assertFalse((self.repo / ".git" / "BISECT_START").exists())
        blocker.rmdir()
        self.assertEqual(
            self._run('read -r version < version\ntest "$version" -lt 2'),
            self.commits[2],
        )
        self._assert_saved_and_reset("found")

    def test_failed_process_cleanup_preserves_checkout_and_replay_log(self):
        try:
            with (
                patch.object(
                    ShellExecutor,
                    "run_command_streaming",
                    side_effect=ProcessCleanupError(
                        "Process group did not stop after SIGKILL."
                    ),
                ),
                self.assertRaises(TritonBisectError) as error,
            ):
                self._run("exit 0")
            result = error.exception.result
            self.assertEqual(result.status, "error")
            self.assertIn("state was preserved", result.message)
            self.assertIn("SIGKILL", result.message)
            self.assertTrue((self.repo / ".git" / "BISECT_START").exists())
            self.assertNotEqual(self._git("rev-parse", "HEAD"), self.commits[-1])
            self.assertEqual(
                json.loads(Path(result.result_file).read_text()), result.to_dict()
            )
            self.assertIn("git bisect start", Path(result.git_bisect_log).read_text())
            self.assertNotIn(
                "Command: git bisect reset", Path(result.command_log).read_text()
            )
        finally:
            self._git("bisect", "reset")

    def test_partial_result_write_does_not_publish_or_reference_invalid_json(self):
        original_temporary_file = tempfile.NamedTemporaryFile

        @contextmanager
        def fail_result_write(*args, **kwargs):
            with original_temporary_file(*args, **kwargs) as output:
                if "bisect_result" in kwargs.get("suffix", ""):
                    writer = MagicMock(wraps=output)
                    writer.name = output.name

                    def fail_after_partial_write(contents):
                        output.write(contents[:10])
                        raise OSError("disk full during result write")

                    writer.write.side_effect = fail_after_partial_write
                    yield writer
                else:
                    yield output

        with (
            patch(
                "tritonparse.bisect.base_bisector.tempfile.NamedTemporaryFile",
                side_effect=fail_result_write,
            ),
            self.assertRaises(TritonBisectError) as error,
        ):
            self._run('read -r version < version\ntest "$version" -lt 2')
        result = error.exception.result
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.result_file)
        self.assertIsNone(result.culprit)
        self.assertEqual(result.candidates, [self.commits[2]])
        self.assertIn("disk full", result.message)
        self.assertEqual(list(self.logger.log_dir.glob("*bisect_result*")), [])
        self.assertTrue(Path(result.git_bisect_log).is_file())
        self.assertFalse((self.repo / ".git" / "BISECT_START").exists())

    def test_interrupt_with_cleanup_failure_stays_aborted_and_preserves_checkout(self):
        original_stop = ShellExecutor._stop_process_group

        def stop_then_report_failure(process):
            # Really stop the test child; emulate a reported cleanup failure
            # only after the process no longer owns resources.
            original_stop(process)
            raise ProcessCleanupError("simulated cleanup failure")

        def interrupt(line):
            if line == "ready":
                raise KeyboardInterrupt()

        try:
            with (
                patch.object(
                    ShellExecutor,
                    "_stop_process_group",
                    side_effect=stop_then_report_failure,
                ),
                self.assertRaises(TritonBisectError) as error,
            ):
                self._run("printf 'ready\\n'\nsleep 30", output_callback=interrupt)
            result = error.exception.result
            self.assertEqual(result.status, "aborted")
            self.assertEqual(result.exit_code, 130)
            self.assertIn("ready", result.message)
            self.assertIn("simulated cleanup failure", result.message)
            self.assertTrue((self.repo / ".git" / "BISECT_START").exists())
            self.assertEqual(
                json.loads(Path(result.result_file).read_text()), result.to_dict()
            )
            self.assertIn("git bisect start", Path(result.git_bisect_log).read_text())
            commands = Path(result.command_log).read_text()
            self.assertIn("Exit code: 130, Duration:", commands)
            self.assertNotIn("Command: git bisect reset", commands)
        finally:
            self._git("bisect", "reset")

    def test_partial_replay_log_is_not_published_or_referenced(self):
        original_temporary_file = tempfile.NamedTemporaryFile

        @contextmanager
        def fail_log_write(*args, **kwargs):
            with original_temporary_file(*args, **kwargs) as output:
                if "git_bisect.log" in kwargs.get("suffix", ""):
                    writer = MagicMock(wraps=output)
                    writer.name = output.name

                    def fail_after_partial_write(contents):
                        output.write(contents[:10])
                        raise OSError("disk full during log write")

                    writer.write.side_effect = fail_after_partial_write
                    yield writer
                else:
                    yield output

        with (
            patch(
                "tritonparse.bisect.executor.tempfile.NamedTemporaryFile",
                side_effect=fail_log_write,
            ),
            self.assertRaises(TritonBisectError) as error,
        ):
            self._run('read -r version < version\ntest "$version" -lt 2')
        result = error.exception.result
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.git_bisect_log)
        self.assertIsNone(result.culprit)
        self.assertIn("disk full during log write", result.message)
        self.assertEqual(list(self.logger.log_dir.glob("*git_bisect.log*")), [])
        self.assertFalse((self.repo / ".git" / "BISECT_START").exists())
        self.assertEqual(
            json.loads(Path(result.result_file).read_text()), result.to_dict()
        )

    def test_result_rename_failure_does_not_leave_a_result_reference(self):
        with (
            patch.object(Path, "replace", side_effect=OSError("rename failed")),
            self.assertRaises(TritonBisectError) as error,
        ):
            self._run('read -r version < version\ntest "$version" -lt 2')
        self.assertEqual(error.exception.result.status, "error")
        self.assertIsNone(error.exception.result.result_file)
        self.assertEqual(list(self.logger.log_dir.glob("*bisect_result*")), [])
        self.assertIn("rename failed", error.exception.result.message)

    def test_log_capture_failure_cannot_report_success(self):
        with (
            patch.object(
                ShellExecutor, "_save_bisect_log", side_effect=OSError("disk full")
            ),
            self.assertRaises(TritonBisectError) as error,
        ):
            self._run('read -r version < version\ntest "$version" -lt 2')
        result = error.exception.result
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.culprit)
        self.assertIn("disk full", result.message)
        self.assertEqual(self._git("rev-parse", "HEAD"), self.commits[-1])
        self.assertEqual(
            json.loads(Path(result.result_file).read_text())["status"], "error"
        )

    def test_linked_worktree_keeps_its_replay_log(self):
        worktree = self.root / "worktree"
        self._git("worktree", "add", "--detach", str(worktree), self.commits[-1])
        self.repo = worktree
        self.assertTrue((self.repo / ".git").is_file())
        self.assertEqual(
            self._run('read -r version < version\ntest "$version" -lt 2'),
            self.commits[2],
        )
        self._assert_saved_and_reset("found")

    def test_existing_worktree_bisect_is_not_reset(self):
        worktree = self.root / "worktree"
        self._git("worktree", "add", "--detach", str(worktree), self.commits[-1])
        self.repo = worktree
        self._git("bisect", "start", self.commits[-1], self.commits[0])
        previous_head = self._git("rev-parse", "HEAD")
        previous_log = self._git("bisect", "log")
        try:
            with self.assertRaises(TritonBisectError) as error:
                self._run("exit 0")
            self.assertEqual(error.exception.result.status, "error")
            self.assertIn("already in progress", error.exception.result.message)
            self.assertEqual(self._git("rev-parse", "HEAD"), previous_head)
            self.assertEqual(self._git("bisect", "log"), previous_log)
            self.assertIsNone(error.exception.result.git_bisect_log)
        finally:
            self._git("bisect", "reset")

    def test_repeated_attempts_do_not_overwrite_evidence(self):
        body = 'read -r version < version\ntest "$version" -lt 2'
        self._run(body)
        first = self.bisector.result
        first_log = Path(first.git_bisect_log).read_text()
        self._run(body)
        second = self.bisector.result
        self.assertNotEqual(first.result_file, second.result_file)
        self.assertNotEqual(first.git_bisect_log, second.git_bisect_log)
        self.assertTrue(Path(first.result_file).exists())
        self.assertEqual(Path(first.git_bisect_log).read_text(), first_log)


if __name__ == "__main__":
    unittest.main()
