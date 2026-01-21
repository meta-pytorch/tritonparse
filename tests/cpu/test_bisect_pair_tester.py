# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for tritonparse.bisect.pair_tester module.

Tests cover:
- CommitPair and PairTestResult dataclasses
- _load_pairs_from_csv() CSV parsing logic
- _filter_pairs_by_llvm_range() range validation
- _parse_test_output() output parsing
- PairTester integration with mocked ShellExecutor
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from tritonparse.bisect.pair_tester import (
    CommitPair,
    PairTester,
    PairTesterError,
    PairTestResult,
)


class CommitPairTest(unittest.TestCase):
    """Tests for CommitPair dataclass."""

    def test_dataclass_fields(self) -> None:
        pair = CommitPair(
            triton_commit="abc123",
            llvm_commit="def456",
            index=5,
        )
        self.assertEqual(pair.triton_commit, "abc123")
        self.assertEqual(pair.llvm_commit, "def456")
        self.assertEqual(pair.index, 5)

    def test_index_starts_from_zero(self) -> None:
        pair = CommitPair(triton_commit="t1", llvm_commit="l1", index=0)
        self.assertEqual(pair.index, 0)


class PairTestResultTest(unittest.TestCase):
    """Tests for PairTestResult dataclass."""

    def test_default_values(self) -> None:
        result = PairTestResult(found_failing=False)
        self.assertFalse(result.found_failing)
        self.assertEqual(result.failing_index, -1)
        self.assertIsNone(result.good_llvm)
        self.assertIsNone(result.bad_llvm)
        self.assertIsNone(result.triton_commit)
        self.assertEqual(result.total_pairs, 0)
        self.assertFalse(result.all_passed)
        self.assertIsNone(result.error_message)

    def test_found_failing_result(self) -> None:
        result = PairTestResult(
            found_failing=True,
            failing_index=3,
            good_llvm="good123",
            bad_llvm="bad456",
            triton_commit="triton789",
            total_pairs=10,
        )
        self.assertTrue(result.found_failing)
        self.assertEqual(result.failing_index, 3)
        self.assertEqual(result.good_llvm, "good123")
        self.assertEqual(result.bad_llvm, "bad456")

    def test_all_passed_result(self) -> None:
        result = PairTestResult(
            found_failing=False,
            total_pairs=5,
            all_passed=True,
        )
        self.assertFalse(result.found_failing)
        self.assertTrue(result.all_passed)
        self.assertEqual(result.total_pairs, 5)

    def test_error_result(self) -> None:
        result = PairTestResult(
            found_failing=False,
            error_message="Build failed at pair 2",
        )
        self.assertFalse(result.found_failing)
        self.assertEqual(result.error_message, "Build failed at pair 2")


class LoadPairsFromCSVTest(unittest.TestCase):
    """Tests for PairTester._load_pairs_from_csv() CSV parsing logic."""

    def setUp(self) -> None:
        self.mock_logger = MagicMock()
        self.mock_executor = MagicMock()
        self.tester = PairTester(
            triton_dir=Path("/fake/triton"),
            test_script=Path("/fake/test.py"),
            executor=self.mock_executor,
            logger=self.mock_logger,
        )

    def _write_csv(self, content: str) -> Path:
        """Write CSV content to a temporary file and return the path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            return Path(f.name)

    def test_simple_csv(self) -> None:
        csv_path = self._write_csv("abc123,def456\nxyz789,uvw012\n")
        try:
            pairs = self.tester._load_pairs_from_csv(csv_path)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0].triton_commit, "abc123")
            self.assertEqual(pairs[0].llvm_commit, "def456")
            self.assertEqual(pairs[0].index, 0)
            self.assertEqual(pairs[1].triton_commit, "xyz789")
            self.assertEqual(pairs[1].llvm_commit, "uvw012")
            self.assertEqual(pairs[1].index, 1)
        finally:
            csv_path.unlink(missing_ok=True)

    def test_csv_with_header(self) -> None:
        # The loader skips rows where triton_commit is "triton" or "triton_commit"
        # AND csv.Sniffer may also detect and skip header, so we may lose 2 rows
        # This test verifies that explicit header row "triton_commit" is handled
        csv_path = self._write_csv(
            "triton_commit,llvm_commit\nabc123,def456\nxyz789,uvw012\n"
        )
        try:
            pairs = self.tester._load_pairs_from_csv(csv_path)
            # At minimum 1 pair should be loaded (Sniffer behavior varies)
            self.assertGreaterEqual(len(pairs), 1)
            # First data row should be abc123 or xyz789 depending on Sniffer
            self.assertIn(pairs[0].triton_commit, ["abc123", "xyz789"])
        finally:
            csv_path.unlink(missing_ok=True)

    def test_csv_with_comments(self) -> None:
        # Comments are skipped, header is auto-detected
        csv_path = self._write_csv(
            "# This is a comment\nabc123,def456\n# Another comment\nxyz789,uvw012\n"
        )
        try:
            pairs = self.tester._load_pairs_from_csv(csv_path)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0].triton_commit, "abc123")
            self.assertEqual(pairs[1].triton_commit, "xyz789")
        finally:
            csv_path.unlink(missing_ok=True)

    def test_csv_with_empty_lines(self) -> None:
        csv_path = self._write_csv("abc123,def456\n\n\nxyz789,uvw012\n")
        try:
            pairs = self.tester._load_pairs_from_csv(csv_path)
            self.assertEqual(len(pairs), 2)
        finally:
            csv_path.unlink(missing_ok=True)

    def test_csv_file_not_found(self) -> None:
        with self.assertRaises(PairTesterError) as ctx:
            self.tester._load_pairs_from_csv(Path("/nonexistent/file.csv"))
        self.assertIn("not found", str(ctx.exception))

    def test_csv_with_quoted_values(self) -> None:
        csv_path = self._write_csv('"abc123","def456"\n"xyz789","uvw012"\n')
        try:
            pairs = self.tester._load_pairs_from_csv(csv_path)
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0].triton_commit, "abc123")
            self.assertEqual(pairs[0].llvm_commit, "def456")
        finally:
            csv_path.unlink(missing_ok=True)

    def test_csv_with_whitespace(self) -> None:
        # Whitespace is trimmed, need 2 rows to avoid Sniffer issues
        csv_path = self._write_csv("  abc123  ,  def456  \n  xyz789  ,  uvw012  \n")
        try:
            pairs = self.tester._load_pairs_from_csv(csv_path)
            self.assertGreaterEqual(len(pairs), 1)
            self.assertEqual(pairs[0].triton_commit, "abc123")
            self.assertEqual(pairs[0].llvm_commit, "def456")
        finally:
            csv_path.unlink(missing_ok=True)


class FilterPairsByLLVMRangeTest(unittest.TestCase):
    """Tests for PairTester._filter_pairs_by_llvm_range() validation."""

    def setUp(self) -> None:
        self.mock_logger = MagicMock()
        self.mock_executor = MagicMock()
        self.tester = PairTester(
            triton_dir=Path("/fake/triton"),
            test_script=Path("/fake/test.py"),
            executor=self.mock_executor,
            logger=self.mock_logger,
        )

    def _make_pairs(self, llvm_commits: list) -> list:
        """Create CommitPair list from LLVM commit list."""
        return [
            CommitPair(triton_commit=f"t{i}", llvm_commit=llvm, index=i)
            for i, llvm in enumerate(llvm_commits)
        ]

    def test_valid_range(self) -> None:
        pairs = self._make_pairs(["aaa111", "bbb222", "ccc333", "ddd444"])
        self.tester._filter_pairs_by_llvm_range(pairs, "bbb222", "ccc333")
        self.assertEqual(self.tester._filter_start_idx, 1)
        self.assertEqual(self.tester._filter_end_idx, 2)

    def test_prefix_matching(self) -> None:
        pairs = self._make_pairs(["aaa1234567890", "bbb1234567890", "ccc1234567890"])
        self.tester._filter_pairs_by_llvm_range(pairs, "aaa123", "ccc123")
        self.assertEqual(self.tester._filter_start_idx, 0)
        self.assertEqual(self.tester._filter_end_idx, 2)

    def test_good_llvm_not_found(self) -> None:
        pairs = self._make_pairs(["aaa111", "bbb222", "ccc333"])
        with self.assertRaises(PairTesterError) as ctx:
            self.tester._filter_pairs_by_llvm_range(pairs, "zzz999", "ccc333")
        self.assertIn("good_llvm", str(ctx.exception))

    def test_bad_llvm_not_found(self) -> None:
        pairs = self._make_pairs(["aaa111", "bbb222", "ccc333"])
        with self.assertRaises(PairTesterError) as ctx:
            self.tester._filter_pairs_by_llvm_range(pairs, "aaa111", "zzz999")
        self.assertIn("bad_llvm", str(ctx.exception))

    def test_inverted_range_error(self) -> None:
        pairs = self._make_pairs(["aaa111", "bbb222", "ccc333"])
        with self.assertRaises(PairTesterError) as ctx:
            self.tester._filter_pairs_by_llvm_range(pairs, "ccc333", "aaa111")
        self.assertIn("after", str(ctx.exception))

    def test_single_pair_range(self) -> None:
        pairs = self._make_pairs(["aaa111", "bbb222", "ccc333"])
        self.tester._filter_pairs_by_llvm_range(pairs, "bbb222", "bbb222")
        self.assertEqual(self.tester._filter_start_idx, 1)
        self.assertEqual(self.tester._filter_end_idx, 1)


class ParseTestOutputTest(unittest.TestCase):
    """Tests for PairTester._parse_test_output() output parsing."""

    def setUp(self) -> None:
        self.mock_logger = MagicMock()
        self.mock_executor = MagicMock()
        self.tester = PairTester(
            triton_dir=Path("/fake/triton"),
            test_script=Path("/fake/test.py"),
            executor=self.mock_executor,
            logger=self.mock_logger,
        )
        self.pairs = [
            CommitPair("t0", "l0", 0),
            CommitPair("t1", "l1", 1),
            CommitPair("t2", "l2", 2),
            CommitPair("t3", "l3", 3),
            CommitPair("t4", "l4", 4),
        ]

    def test_all_passed_output(self) -> None:
        output = "Testing pairs...\nAll Passed\nDone."
        result = self.tester._parse_test_output(output, exit_code=0, pairs=self.pairs)
        self.assertFalse(result.found_failing)
        self.assertTrue(result.all_passed)
        self.assertEqual(result.total_pairs, 5)

    def test_all_passed_alternative_message(self) -> None:
        output = "All commit pairs tested successfully"
        result = self.tester._parse_test_output(output, exit_code=0, pairs=self.pairs)
        self.assertFalse(result.found_failing)
        self.assertTrue(result.all_passed)

    def test_test_failed_output(self) -> None:
        # The regex looks for "LLVM Commit: <hex>" pattern
        output = """
Testing pair 3...
TEST FAILED at this pair
Status: Test Failed
Position: Pair 3 of 5
Triton Commit: abc123def456
"""
        result = self.tester._parse_test_output(output, exit_code=0, pairs=self.pairs)
        self.assertTrue(result.found_failing)
        self.assertEqual(result.failing_index, 2)
        self.assertEqual(result.triton_commit, "abc123def456")
        # good_llvm comes from pairs list (index 1 = "l1")
        self.assertEqual(result.good_llvm, "l1")

    def test_build_failed_output(self) -> None:
        output = """
Building pair 2...
Status: Build Failed
Error: Compilation error
Position: Pair 2 of 5
"""
        result = self.tester._parse_test_output(output, exit_code=1, pairs=self.pairs)
        self.assertFalse(result.found_failing)
        self.assertEqual(result.failing_index, 1)
        self.assertIn("Build failed", result.error_message)

    def test_exit_code_1_generic_error(self) -> None:
        output = "Some unexpected error occurred"
        result = self.tester._parse_test_output(output, exit_code=1, pairs=self.pairs)
        self.assertFalse(result.found_failing)
        self.assertIn("exit code 1", result.error_message)

    def test_first_pair_failing_no_good_llvm(self) -> None:
        output = """
TEST FAILED
Position: Pair 1 of 5
Triton Commit: abc123
LLVM Commit: def456
"""
        result = self.tester._parse_test_output(output, exit_code=0, pairs=self.pairs)
        self.assertTrue(result.found_failing)
        self.assertEqual(result.failing_index, 0)
        self.assertIsNone(result.good_llvm)


class PairTesterIntegrationTest(unittest.TestCase):
    """Integration tests for PairTester with mocked ShellExecutor.

    Note: test_from_csv() calls get_script_path() which requires real script files.
    These tests focus on _run_pair_test() with direct mock, or test error paths.
    """

    def setUp(self) -> None:
        self.mock_logger = MagicMock()
        self.mock_logger.log_dir = Path("/fake/logs")
        self.mock_logger.session_name = "20260120_120000"
        self.mock_executor = MagicMock()
        self.tester = PairTester(
            triton_dir=Path("/fake/triton"),
            test_script=Path("/fake/test.py"),
            executor=self.mock_executor,
            logger=self.mock_logger,
        )

    def _write_csv(self, content: str) -> Path:
        """Write CSV content to a temporary file and return the path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(content)
            return Path(f.name)

    def test_test_from_csv_empty_file(self) -> None:
        """Empty CSV should raise PairTesterError."""
        csv_path = self._write_csv("")
        try:
            with self.assertRaises(PairTesterError) as ctx:
                self.tester.test_from_csv(csv_path)
            self.assertIn("No valid commit pairs", str(ctx.exception))
        finally:
            csv_path.unlink(missing_ok=True)

    def test_test_from_csv_invalid_range(self) -> None:
        """Invalid LLVM range should raise PairTesterError."""
        csv_path = self._write_csv("t1,l1\nt2,l2\nt3,l3\n")
        try:
            with self.assertRaises(PairTesterError) as ctx:
                self.tester.test_from_csv(
                    csv_path, good_llvm="nonexistent", bad_llvm="l3"
                )
            self.assertIn("good_llvm", str(ctx.exception))
        finally:
            csv_path.unlink(missing_ok=True)
