# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for tritonparse.bisect.commit_detector module.

Tests cover:
- LLVMBumpInfo dataclass
- _extract_hash_from_content() pure logic
- CommitDetector against temporary Git histories and failed object reads
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tritonparse.bisect.commit_detector import (
    CommitDetector,
    CommitDetectorError,
    LLVMBumpInfo,
)
from tritonparse.bisect.executor import CommandResult, ShellExecutor


class LLVMBumpInfoTest(unittest.TestCase):
    """Tests for LLVMBumpInfo dataclass."""

    def test_default_fields_are_none(self) -> None:
        info = LLVMBumpInfo(is_llvm_bump=False)
        self.assertFalse(info.is_llvm_bump)
        self.assertIsNone(info.old_hash)
        self.assertIsNone(info.new_hash)
        self.assertIsNone(info.triton_commit)

    def test_all_fields_populated(self) -> None:
        info = LLVMBumpInfo(
            is_llvm_bump=True,
            old_hash="abc1234",
            new_hash="def5678",
            triton_commit="commit123",
        )
        self.assertTrue(info.is_llvm_bump)
        self.assertEqual(info.old_hash, "abc1234")
        self.assertEqual(info.new_hash, "def5678")
        self.assertEqual(info.triton_commit, "commit123")


class ExtractHashTest(unittest.TestCase):
    """Tests for CommitDetector._extract_hash_from_content() pure logic."""

    def setUp(self) -> None:
        mock_logger = MagicMock()
        mock_executor = MagicMock()
        self.detector = CommitDetector(
            triton_dir=Path("/fake/triton"),
            executor=mock_executor,
            logger=mock_logger,
        )

    def test_simple_hash(self) -> None:
        content = "abc1234def5678901234567890123456789012"
        result = self.detector._extract_hash_from_content(content)
        self.assertEqual(result, "abc1234def5678901234567890123456789012")

    def test_hash_with_whitespace(self) -> None:
        content = "  abc1234def5678901234567890123456789012  \n"
        result = self.detector._extract_hash_from_content(content)
        self.assertEqual(result, "abc1234def5678901234567890123456789012")

    def test_hash_with_newlines(self) -> None:
        content = "\n\nabc1234def5678901234567890123456789012\n\n"
        result = self.detector._extract_hash_from_content(content)
        self.assertEqual(result, "abc1234def5678901234567890123456789012")

    def test_hash_with_comment_lines(self) -> None:
        content = "# This is a comment\nabc1234def5678\n# Another comment"
        result = self.detector._extract_hash_from_content(content)
        self.assertEqual(result, "abc1234def5678")

    def test_short_hash_7_chars(self) -> None:
        content = "abc1234"
        result = self.detector._extract_hash_from_content(content)
        self.assertEqual(result, "abc1234")

    def test_full_40_char_hash(self) -> None:
        content = "1234567890abcdef1234567890abcdef12345678"
        result = self.detector._extract_hash_from_content(content)
        self.assertEqual(result, "1234567890abcdef1234567890abcdef12345678")

    def test_invalid_content_raises_error(self) -> None:
        with self.assertRaises(CommitDetectorError):
            self.detector._extract_hash_from_content("not a valid hash!")

    def test_too_short_hash_raises_error(self) -> None:
        with self.assertRaises(CommitDetectorError):
            self.detector._extract_hash_from_content("abc123")

    def test_empty_content_raises_error(self) -> None:
        with self.assertRaises(CommitDetectorError):
            self.detector._extract_hash_from_content("")

    def test_only_comments_raises_error(self) -> None:
        with self.assertRaises(CommitDetectorError):
            self.detector._extract_hash_from_content("# Comment only\n# Another")


class CommitDetectorTest(unittest.TestCase):
    """Exercise descriptor transitions using real, temporary Git histories."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.executor = ShellExecutor(MagicMock())
        self.git("init")
        self.git("config", "user.name", "Tritonparse Test")
        self.git("config", "user.email", "tritonparse-test@example.invalid")
        self.detector = CommitDetector(
            triton_dir=self.repo,
            executor=self.executor,
            logger=MagicMock(),
            artifact_platform="almalinux-x64",
        )
        self.llvm_a = "a" * 40
        self.llvm_b = "b" * 40

    def git(self, *args: str) -> str:
        result = self.executor.run_command(["git", *args], cwd=str(self.repo))
        self.assertTrue(result.success, result.output)
        return result.stdout.strip()

    def commit(self, **files: str | None) -> str:
        for name, content in files.items():
            path = self.repo / "cmake" / name
            path.parent.mkdir(exist_ok=True)
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(content)
        self.git("add", "-A")
        self.git("commit", "--allow-empty", "-m", "descriptor fixture")
        return self.git("rev-parse", "HEAD")

    def modern(self, **values: object) -> str:
        return json.dumps(
            {
                "llvm_hash": self.llvm_a,
                "build_number": 1,
                "sha256sum": {"almalinux-x64": "1" * 64, "macos-x64": "2" * 64},
                **values,
            }
        )

    def test_legacy_comments_are_not_a_source_bump(self) -> None:
        self.commit(**{"llvm-hash.txt": self.llvm_a})
        commit = self.commit(
            **{"llvm-hash.txt": f"# Updated comment\n {self.llvm_a}\n"}
        )
        info = self.detector.detect(commit)
        self.assertFalse(info.is_llvm_bump)
        self.assertIsNone(info.artifact_changed)
        self.assertEqual(info.old_hash, info.new_hash)

    def test_legacy_source_bump(self) -> None:
        self.commit(**{"llvm-hash.txt": self.llvm_a})
        commit = self.commit(**{"llvm-hash.txt": self.llvm_b})
        info = self.detector.detect(commit)
        self.assertTrue(info.is_llvm_bump)
        self.assertEqual((info.old_hash, info.new_hash), (self.llvm_a, self.llvm_b))
        self.assertEqual(info.triton_commit, commit)

    def test_modern_source_bump_and_lookup(self) -> None:
        self.commit(**{"llvm-info.json": self.modern()})
        commit = self.commit(**{"llvm-info.json": self.modern(llvm_hash=self.llvm_b)})
        info = self.detector.detect(commit)
        self.assertTrue(info.is_llvm_bump)
        self.assertEqual(self.detector.get_llvm_hash_at_commit(commit), self.llvm_b)
        self.assertEqual(info.new_descriptor.source_file, "cmake/llvm-info.json")

    def test_format_migration_preserves_source_identity(self) -> None:
        self.commit(**{"llvm-hash.txt": self.llvm_a.upper()})
        commit = self.commit(**{"llvm-hash.txt": None, "llvm-info.json": self.modern()})
        info = self.detector.detect(commit)
        self.assertFalse(info.is_llvm_bump)
        self.assertIsNone(info.artifact_changed)
        self.assertEqual((info.old_hash, info.new_hash), (self.llvm_a, self.llvm_a))

    def test_json_preferred_when_both_formats_exist(self) -> None:
        commit = self.commit(
            **{"llvm-hash.txt": self.llvm_b, "llvm-info.json": self.modern()}
        )
        self.assertEqual(self.detector.get_llvm_hash_at_commit(commit), self.llvm_a)

    def test_json_reformatting_is_not_a_bump(self) -> None:
        self.commit(**{"llvm-info.json": self.modern()})
        commit = self.commit(
            **{
                "llvm-info.json": json.dumps(
                    json.loads(self.modern()), indent=2, sort_keys=True
                )
            }
        )
        info = self.detector.detect(commit)
        self.assertFalse(info.is_llvm_bump)
        self.assertFalse(info.artifact_changed)

    def test_build_number_update_is_artifact_only(self) -> None:
        self.commit(**{"llvm-info.json": self.modern()})
        commit = self.commit(**{"llvm-info.json": self.modern(build_number=2)})
        info = self.detector.detect(commit)
        self.assertFalse(info.is_llvm_bump)
        self.assertTrue(info.artifact_changed)
        self.assertEqual(info.to_dict()["new_descriptor"]["build_number"], 2)

    def test_selected_platform_checksum_changed(self) -> None:
        self.commit(**{"llvm-info.json": self.modern()})
        commit = self.commit(
            **{"llvm-info.json": self.modern(sha256sum={"almalinux-x64": "3" * 64})}
        )
        info = self.detector.detect(commit)
        self.assertFalse(info.is_llvm_bump)
        self.assertTrue(info.artifact_changed)

    def test_other_platform_checksum_does_not_change_local_artifact(self) -> None:
        self.commit(**{"llvm-info.json": self.modern()})
        commit = self.commit(
            **{
                "llvm-info.json": self.modern(
                    sha256sum={"almalinux-x64": "1" * 64, "macos-x64": "3" * 64}
                )
            }
        )
        self.assertFalse(self.detector.detect(commit).artifact_changed)
        self.detector.artifact_platform = None
        self.assertIsNone(self.detector.detect(commit).artifact_changed)

    def test_unchanged_all_platform_checksums(self) -> None:
        self.commit(**{"llvm-info.json": self.modern()})
        commit = self.commit()
        self.detector.artifact_platform = None
        self.assertFalse(self.detector.detect(commit).artifact_changed)

    def test_missing_selected_platform_is_unknown(self) -> None:
        self.commit(**{"llvm-info.json": self.modern()})
        commit = self.commit(**{"llvm-info.json": self.modern(sha256sum={})})
        self.assertIsNone(self.detector.detect(commit).artifact_changed)

    def test_bad_json_does_not_fall_back_to_legacy(self) -> None:
        for content in (
            "{",
            "[]",
            self.modern(llvm_hash="abc1234"),
            self.modern(build_number=True),
            self.modern(build_number=-1),
            self.modern(sha256sum={"almalinux-x64": "invalid"}),
        ):
            with self.subTest(content=content):
                commit = self.commit(
                    **{"llvm-hash.txt": self.llvm_a, "llvm-info.json": content}
                )
                with self.assertRaises(CommitDetectorError):
                    self.detector.read_llvm_descriptor(commit)

    def test_git_read_failure_does_not_fall_back_to_legacy(self) -> None:
        commit = self.commit(
            **{"llvm-hash.txt": self.llvm_a, "llvm-info.json": self.modern()}
        )
        run_command = self.executor.run_command

        def fail_show(command, **kwargs):
            if command[1] == "show":
                return CommandResult(
                    command=" ".join(command),
                    exit_code=128,
                    stdout="",
                    stderr="unreadable object",
                    duration_seconds=0,
                )
            return run_command(command, **kwargs)

        with patch.object(self.executor, "run_command", side_effect=fail_show):
            with self.assertRaisesRegex(CommitDetectorError, "unreadable object"):
                self.detector.read_llvm_descriptor(commit)

    def test_missing_descriptor_and_bad_revision_are_errors(self) -> None:
        commit = self.commit()
        with self.assertRaises(CommitDetectorError):
            self.detector.read_llvm_descriptor(commit)
        with self.assertRaises(CommitDetectorError):
            self.detector.detect("not-a-revision")

    def test_root_has_no_comparable_parent(self) -> None:
        commit = self.commit(**{"llvm-info.json": self.modern()})
        with self.assertRaises(CommitDetectorError):
            self.detector.detect(commit)

    def test_abbreviated_hashes_are_not_assumed_equal(self) -> None:
        self.commit(**{"llvm-hash.txt": self.llvm_a[:7]})
        commit = self.commit(**{"llvm-info.json": self.modern()})
        with self.assertRaisesRegex(CommitDetectorError, "full 40-character"):
            self.detector.detect(commit)

    def test_merge_comparison_uses_explicit_or_first_parent(self) -> None:
        first = self.commit(**{"llvm-info.json": self.modern()})
        second = self.commit(**{"llvm-info.json": self.modern(llvm_hash=self.llvm_b)})
        merge = self.git(
            "commit-tree", f"{first}^{{tree}}", "-p", first, "-p", second, "-m", "merge"
        )
        self.assertFalse(self.detector.detect(merge).is_llvm_bump)
        info = self.detector.detect(merge, parent=second)
        self.assertTrue(info.is_llvm_bump)
        self.assertEqual(info.old_descriptor.revision, second)
