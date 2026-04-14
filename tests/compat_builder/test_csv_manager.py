# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for tritonparse.compat_builder.csv_manager."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tritonparse.compat_builder.csv_manager import BumpBlock, CSVManager


def _make_metadata(
    *,
    schema_version: str = "1",
    llvm_bump_commit: str = "bumpabc123def456abcd",
    old_llvm: str = "a" * 40,
    new_llvm: str = "b" * 40,
    final_bad_triton_commit: str = "bumpabc123def456abcd",
    final_bad_llvm: str = "b" * 40,
) -> BumpBlock:
    return BumpBlock(
        schema_version=schema_version,
        llvm_bump_commit=llvm_bump_commit,
        old_llvm=old_llvm,
        new_llvm=new_llvm,
        final_bad_triton_commit=final_bad_triton_commit,
        final_bad_llvm=final_bad_llvm,
    )


def _write(
    path: Path,
    metadata: BumpBlock | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> Path:
    """Write a valid CSV to path and return it."""
    return CSVManager.write_csv(
        output_path=path,
        metadata=metadata if metadata is not None else _make_metadata(),
        pairs=pairs if pairs is not None else [],
    )


class BumpBlockTest(unittest.TestCase):
    def test_fields_stored(self) -> None:
        bb = _make_metadata()
        self.assertEqual(bb.schema_version, "1")
        self.assertEqual(bb.llvm_bump_commit, "bumpabc123def456abcd")
        self.assertEqual(bb.old_llvm, "a" * 40)
        self.assertEqual(bb.new_llvm, "b" * 40)
        self.assertEqual(bb.final_bad_triton_commit, "bumpabc123def456abcd")
        self.assertEqual(bb.final_bad_llvm, "b" * 40)

    def test_equality_same_values(self) -> None:
        self.assertEqual(_make_metadata(), _make_metadata())

    def test_inequality_different_field(self) -> None:
        a = _make_metadata(old_llvm="a" * 40)
        b = _make_metadata(old_llvm="c" * 40)
        self.assertNotEqual(a, b)


class CSVManagerConstructorTest(unittest.TestCase):
    def test_raises_when_file_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            CSVManager(Path("/nonexistent/path/missing.csv"))

    def test_succeeds_when_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "c.csv")
            mgr = CSVManager(path)
            self.assertEqual(mgr._path, path)


class CSVManagerLoadMetadataTest(unittest.TestCase):
    def test_parses_all_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata()
            path = _write(Path(tmp) / "c.csv", metadata=meta)
            loaded = CSVManager(path).load_metadata()
            self.assertEqual(loaded, meta)

    def test_schema_version_is_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "c.csv")
            loaded = CSVManager(path).load_metadata()
            self.assertEqual(loaded.schema_version, "1")

    def test_llvm_bump_commit_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata(llvm_bump_commit="deadbeef1234abcd5678")
            path = _write(Path(tmp) / "c.csv", metadata=meta)
            loaded = CSVManager(path).load_metadata()
            self.assertEqual(loaded.llvm_bump_commit, "deadbeef1234abcd5678")

    def test_missing_key_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            # Write without final_bad_llvm
            path.write_text(
                "# schema_version=1\n"
                "# llvm_bump_commit=abc\n"
                "# old_llvm=old\n"
                "# new_llvm=new\n"
                "# final_bad_triton_commit=abc\n"
                "triton_commit,llvm_commit_last_compatible\n"
            )
            with self.assertRaises(ValueError):
                CSVManager(path).load_metadata()

    def test_all_keys_missing_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            path.write_text("triton_commit,llvm_commit_last_compatible\n")
            with self.assertRaises(ValueError):
                CSVManager(path).load_metadata()

    def test_comment_lines_without_equals_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata()
            path = _write(Path(tmp) / "c.csv", metadata=meta)
            # Prepend a comment-only line with no '='
            original = path.read_text()
            path.write_text("# This is a comment\n" + original)
            loaded = CSVManager(path).load_metadata()
            self.assertEqual(loaded, meta)


class CSVManagerLoadPairsTest(unittest.TestCase):
    def test_empty_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "c.csv", pairs=[])
            self.assertEqual(CSVManager(path).load_pairs(), [])

    def test_single_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "c.csv", pairs=[("triton_aaa", "llvm_aaa")])
            self.assertEqual(
                CSVManager(path).load_pairs(), [("triton_aaa", "llvm_aaa")]
            )

    def test_multiple_pairs_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            expected = [("triton_aaa", "llvm_aaa"), ("triton_bbb", "llvm_bbb")]
            path = _write(Path(tmp) / "c.csv", pairs=expected)
            self.assertEqual(CSVManager(path).load_pairs(), expected)

    def test_column_header_not_in_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "c.csv", pairs=[("triton_aaa", "llvm_aaa")])
            pairs = CSVManager(path).load_pairs()
            for triton, _ in pairs:
                self.assertNotEqual(triton, "triton_commit")

    def test_whitespace_stripped_from_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            path.write_text(
                "# schema_version=1\n"
                "# llvm_bump_commit=abc\n"
                "# old_llvm=old\n"
                "# new_llvm=new\n"
                "# final_bad_triton_commit=abc\n"
                "# final_bad_llvm=new\n"
                "triton_commit,llvm_commit_last_compatible\n"
                " triton_aaa , llvm_aaa \n"
            )
            pairs = CSVManager(path).load_pairs()
            self.assertEqual(pairs, [("triton_aaa", "llvm_aaa")])

    def test_malformed_row_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            path.write_text(
                "# schema_version=1\n"
                "# llvm_bump_commit=abc\n"
                "# old_llvm=old\n"
                "# new_llvm=new\n"
                "# final_bad_triton_commit=abc\n"
                "# final_bad_llvm=new\n"
                "triton_commit,llvm_commit_last_compatible\n"
                "only_one_field\n"
            )
            with self.assertRaises(ValueError):
                CSVManager(path).load_pairs()


class CSVManagerWriteCsvTest(unittest.TestCase):
    def test_write_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            CSVManager.write_csv(path, _make_metadata(), [])
            self.assertTrue(path.exists())

    def test_write_returns_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            result = CSVManager.write_csv(path, _make_metadata(), [])
            self.assertEqual(result, path)

    def test_write_contains_all_metadata_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata()
            path = Path(tmp) / "c.csv"
            CSVManager.write_csv(path, meta, [])
            content = path.read_text()
            self.assertIn(f"# schema_version={meta.schema_version}", content)
            self.assertIn(f"# llvm_bump_commit={meta.llvm_bump_commit}", content)
            self.assertIn(f"# old_llvm={meta.old_llvm}", content)
            self.assertIn(f"# new_llvm={meta.new_llvm}", content)
            self.assertIn(
                f"# final_bad_triton_commit={meta.final_bad_triton_commit}", content
            )
            self.assertIn(f"# final_bad_llvm={meta.final_bad_llvm}", content)

    def test_write_contains_column_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            CSVManager.write_csv(path, _make_metadata(), [])
            content = path.read_text()
            self.assertIn("triton_commit,llvm_commit_last_compatible", content)

    def test_write_contains_data_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            pairs = [("triton_aaa", "llvm_aaa"), ("triton_bbb", "llvm_bbb")]
            CSVManager.write_csv(path, _make_metadata(), pairs)
            content = path.read_text()
            self.assertIn("triton_aaa,llvm_aaa", content)
            self.assertIn("triton_bbb,llvm_bbb", content)

    def test_write_ends_with_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            CSVManager.write_csv(path, _make_metadata(), [])
            self.assertTrue(path.read_text().endswith("\n"))

    def test_write_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subdir" / "nested" / "c.csv"
            CSVManager.write_csv(path, _make_metadata(), [])
            self.assertTrue(path.exists())

    def test_write_empty_pairs_no_data_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            CSVManager.write_csv(path, _make_metadata(), [])
            content = path.read_text()
            data_rows = [
                line
                for line in content.splitlines()
                if line and not line.startswith("#") and "triton_commit" not in line
            ]
            self.assertEqual(data_rows, [])


class CSVManagerRoundTripTest(unittest.TestCase):
    def test_roundtrip_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata()
            path = _write(Path(tmp) / "c.csv", metadata=meta)
            self.assertEqual(CSVManager(path).load_metadata(), meta)

    def test_roundtrip_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [("triton_aaa", "llvm_aaa"), ("triton_bbb", "llvm_bbb")]
            path = _write(Path(tmp) / "c.csv", pairs=pairs)
            self.assertEqual(CSVManager(path).load_pairs(), pairs)

    def test_roundtrip_empty_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "c.csv", pairs=[])
            self.assertEqual(CSVManager(path).load_pairs(), [])

    def test_roundtrip_metadata_with_many_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata(llvm_bump_commit="cafebabe1234abcd5678")
            pairs = [(f"triton_{i:03}", f"llvm_{i:03}") for i in range(10)]
            path = _write(Path(tmp) / "c.csv", metadata=meta, pairs=pairs)
            mgr = CSVManager(path)
            self.assertEqual(mgr.load_metadata(), meta)
            self.assertEqual(mgr.load_pairs(), pairs)


class CSVManagerValidateMonotonicTest(unittest.TestCase):
    def test_empty_pairs_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "c.csv", pairs=[])
            self.assertEqual(CSVManager(path).validate_monotonic_pairs(), [])

    def test_valid_pairs_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [("triton_aaa", "llvm_aaa"), ("triton_bbb", "llvm_bbb")]
            path = _write(Path(tmp) / "c.csv", pairs=pairs)
            self.assertEqual(CSVManager(path).validate_monotonic_pairs(), [])

    def test_empty_triton_hash_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            path.write_text(
                "# schema_version=1\n"
                "# llvm_bump_commit=abc\n"
                "# old_llvm=old\n"
                "# new_llvm=new\n"
                "# final_bad_triton_commit=abc\n"
                "# final_bad_llvm=new\n"
                "triton_commit,llvm_commit_last_compatible\n"
                ",llvm_aaa\n"
            )
            errors = CSVManager(path).validate_monotonic_pairs()
            self.assertGreater(len(errors), 0)
            self.assertTrue(any("empty triton_commit" in e for e in errors))

    def test_empty_llvm_hash_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.csv"
            path.write_text(
                "# schema_version=1\n"
                "# llvm_bump_commit=abc\n"
                "# old_llvm=old\n"
                "# new_llvm=new\n"
                "# final_bad_triton_commit=abc\n"
                "# final_bad_llvm=new\n"
                "triton_commit,llvm_commit_last_compatible\n"
                "triton_aaa,\n"
            )
            errors = CSVManager(path).validate_monotonic_pairs()
            self.assertGreater(len(errors), 0)
            self.assertTrue(
                any("empty llvm_commit_last_compatible" in e for e in errors)
            )

    def test_duplicate_triton_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [("triton_aaa", "llvm_aaa"), ("triton_aaa", "llvm_bbb")]
            path = _write(Path(tmp) / "c.csv", pairs=pairs)
            errors = CSVManager(path).validate_monotonic_pairs()
            self.assertGreater(len(errors), 0)
            self.assertTrue(any("duplicate triton_commit" in e for e in errors))

    def test_multiple_violations_all_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Both rows have empty triton
            path = Path(tmp) / "c.csv"
            path.write_text(
                "# schema_version=1\n"
                "# llvm_bump_commit=abc\n"
                "# old_llvm=old\n"
                "# new_llvm=new\n"
                "# final_bad_triton_commit=abc\n"
                "# final_bad_llvm=new\n"
                "triton_commit,llvm_commit_last_compatible\n"
                ",llvm_aaa\n"
                ",llvm_bbb\n"
            )
            errors = CSVManager(path).validate_monotonic_pairs()
            empty_errors = [e for e in errors if "empty triton_commit" in e]
            self.assertEqual(len(empty_errors), 2)


class CSVManagerValidateTerminalTest(unittest.TestCase):
    def test_consistent_metadata_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata(
                llvm_bump_commit="bumpabc",
                new_llvm="newllvm",
                final_bad_triton_commit="bumpabc",
                final_bad_llvm="newllvm",
            )
            path = _write(Path(tmp) / "c.csv", metadata=meta)
            self.assertEqual(CSVManager(path).validate_terminal_boundary(), [])

    def test_mismatched_final_bad_llvm_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata(
                new_llvm="correct_llvm",
                final_bad_llvm="wrong_llvm",
            )
            path = _write(Path(tmp) / "c.csv", metadata=meta)
            errors = CSVManager(path).validate_terminal_boundary()
            self.assertGreater(len(errors), 0)
            self.assertTrue(any("final_bad_llvm" in e for e in errors))

    def test_mismatched_final_bad_triton_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata(
                llvm_bump_commit="bump_abc",
                final_bad_triton_commit="wrong_triton",
            )
            path = _write(Path(tmp) / "c.csv", metadata=meta)
            errors = CSVManager(path).validate_terminal_boundary()
            self.assertGreater(len(errors), 0)
            self.assertTrue(any("final_bad_triton_commit" in e for e in errors))

    def test_both_mismatched_returns_two_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = _make_metadata(
                llvm_bump_commit="bump_abc",
                new_llvm="new_llvm",
                final_bad_triton_commit="wrong_triton",
                final_bad_llvm="wrong_llvm",
            )
            path = _write(Path(tmp) / "c.csv", metadata=meta)
            errors = CSVManager(path).validate_terminal_boundary()
            self.assertEqual(len(errors), 2)

    def test_default_make_metadata_is_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "c.csv")
            self.assertEqual(CSVManager(path).validate_terminal_boundary(), [])
