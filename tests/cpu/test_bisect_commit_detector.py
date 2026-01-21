# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for tritonparse.bisect.commit_detector module.

Tests cover:
- LLVMBumpInfo dataclass
- _extract_hash_from_content() pure logic
- CommitDetector with mocked ShellExecutor
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from tritonparse.bisect.commit_detector import (
    CommitDetector,
    CommitDetectorError,
    LLVMBumpInfo,
)


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
    """Tests for CommitDetector with mocked git interactions."""

    def setUp(self) -> None:
        self.mock_logger = MagicMock()
        self.mock_executor = MagicMock()
        self.detector = CommitDetector(
            triton_dir=Path("/fake/triton"),
            executor=self.mock_executor,
            logger=self.mock_logger,
        )

    def _make_result(
        self, success: bool, stdout: str = "", stderr: str = ""
    ) -> MagicMock:
        result = MagicMock()
        result.success = success
        result.stdout = stdout
        result.stderr = stderr
        return result

    def test_detect_non_llvm_bump(self) -> None:
        self.mock_executor.run_command.return_value = self._make_result(
            success=True, stdout="src/file1.py\nsrc/file2.py\n"
        )

        info = self.detector.detect("abc1234567890123456789012345678901234567")

        self.assertFalse(info.is_llvm_bump)
        self.assertEqual(info.triton_commit, "abc1234567890123456789012345678901234567")
        self.assertIsNone(info.old_hash)
        self.assertIsNone(info.new_hash)

    def test_detect_llvm_bump(self) -> None:
        old_hash = "0000000000000000000000000000000000000001"
        new_hash = "0000000000000000000000000000000000000002"

        def side_effect(cmd: list, cwd: str) -> MagicMock:
            if cmd[1] == "diff":
                return self._make_result(
                    success=True, stdout="cmake/llvm-hash.txt\nsrc/other.py\n"
                )
            elif "~1:cmake/llvm-hash.txt" in cmd[2]:
                return self._make_result(success=True, stdout=old_hash)
            else:
                return self._make_result(success=True, stdout=new_hash)

        self.mock_executor.run_command.side_effect = side_effect

        info = self.detector.detect("abc1234567890123456789012345678901234567")

        self.assertTrue(info.is_llvm_bump)
        self.assertEqual(info.old_hash, old_hash)
        self.assertEqual(info.new_hash, new_hash)

    def test_get_llvm_hash_at_commit_success(self) -> None:
        self.mock_executor.run_command.return_value = self._make_result(
            success=True, stdout="abc1234567890123456789012345678901234567\n"
        )

        result = self.detector.get_llvm_hash_at_commit("commit123")

        self.assertEqual(result, "abc1234567890123456789012345678901234567")

    def test_get_llvm_hash_at_commit_failure(self) -> None:
        self.mock_executor.run_command.return_value = self._make_result(
            success=False, stderr="fatal: not a git repository"
        )

        with self.assertRaises(CommitDetectorError):
            self.detector.get_llvm_hash_at_commit("commit123")

    def test_fallback_detection_when_diff_fails(self) -> None:
        hash_at_commit = "aaaaaaa1234567890123456789012345678901"
        hash_at_parent = "bbbbbbb1234567890123456789012345678901"
        call_count = [0]

        def side_effect(cmd: list, cwd: str) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                return self._make_result(success=False, stderr="diff failed")
            elif call_count[0] == 2:
                return self._make_result(success=True, stdout=hash_at_commit)
            else:
                return self._make_result(success=True, stdout=hash_at_parent)

        self.mock_executor.run_command.side_effect = side_effect

        is_bump = self.detector._is_llvm_bump_commit("abc123")

        self.assertTrue(is_bump)
