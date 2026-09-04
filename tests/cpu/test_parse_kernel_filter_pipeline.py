# Copyright (c) Meta Platforms, Inc. and affiliates.

import gzip
import json
import os
import tempfile
import unittest
from unittest import mock

from tritonparse.parse import utils as utils_module
from tritonparse.parse.common import parse_logs, RankConfig
from tritonparse.parse.utils import oss_run


def _compilation(kernel_hash: str, kernel_name: str, *, frame_id: int = 0) -> dict:
    return {
        "event_type": "compilation",
        "pid": 1000,
        "timestamp": "2026-09-02T00:00:00",
        "stack": [],
        "payload": {
            "metadata": {
                "hash": kernel_hash,
                "name": kernel_name,
                "num_warps": 4,
                "num_stages": 2,
                "num_ctas": 1,
            },
            "pt_info": {"frame_id": frame_id, "frame_compile_id": 0},
            "file_content": {},
            "file_path": {},
        },
    }


def _write_trace(path: str, events: list[dict]) -> None:
    with open(path, "w") as output:
        for event in events:
            output.write(json.dumps(event) + "\n")


def _compilation_hashes(parsed_dir: str) -> set[str]:
    hashes = set()
    for root, _, filenames in os.walk(parsed_dir):
        for filename in filenames:
            if not filename.endswith(".ndjson.gz"):
                continue
            with gzip.open(os.path.join(root, filename), "rt") as source:
                for line in source:
                    event = json.loads(line)
                    if event.get("event_type") == "compilation":
                        hashes.add(event["payload"]["metadata"]["hash"])
    return hashes


class ParseLogsKernelFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.temporary_directory.name, "input")
        os.makedirs(self.input_dir)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_rank(self, rank: int, kernel_hash: str, kernel_name: str) -> None:
        path = os.path.join(
            self.input_dir,
            f"dedicated_log_triton_trace_user_rank_{rank}_pid_{rank + 1000}_.ndjson",
        )
        _write_trace(path, [_compilation(kernel_hash, kernel_name, frame_id=rank)])

    def test_all_ranks_allows_one_rank_without_a_match(self) -> None:
        self._write_rank(0, "shared_hash", "other_kernel")
        self._write_rank(1, "shared_hash", "target_kernel")

        parsed_dir, file_mapping = parse_logs(
            self.input_dir,
            RankConfig(all_ranks=True),
            kernel_patterns=("target*",),
        )

        self.assertEqual(_compilation_hashes(parsed_dir), {"shared_hash"})
        self.assertNotIn("rank_0", file_mapping)
        self.assertIn("rank_1", file_mapping)
        self.assertFalse(os.path.exists(os.path.join(parsed_dir, "rank_0")))

    def test_no_rank_miss_leaves_no_root_artifacts(self) -> None:
        no_rank_path = os.path.join(
            self.input_dir,
            "dedicated_log_triton_trace_user_pid_2000_.ndjson",
        )
        _write_trace(
            no_rank_path,
            [_compilation("other_hash", "other_kernel")],
        )
        self._write_rank(1, "target_hash", "target_kernel")

        parsed_dir, file_mapping = parse_logs(
            self.input_dir,
            RankConfig(all_ranks=True),
            kernel_patterns=("target*",),
        )

        self.assertEqual(set(os.listdir(parsed_dir)), {"log_file_list.json", "rank_1"})
        self.assertNotIn("rank_none", file_mapping)

    def test_all_ranks_reports_no_match_with_names_from_every_rank(self) -> None:
        self._write_rank(0, "hash_a", "kernel_a")
        self._write_rank(1, "hash_b", "kernel_b")

        with self.assertRaisesRegex(
            RuntimeError,
            "No kernels matched.*kernel_a, kernel_b",
        ):
            parse_logs(
                self.input_dir,
                RankConfig(all_ranks=True),
                kernel_patterns=("target*",),
            )


class OssRunKernelFilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self.temporary_directory.name, "input")
        os.makedirs(self.input_dir)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_comma_separated_allowlist_filters_output(self) -> None:
        trace_path = os.path.join(
            self.input_dir,
            "dedicated_log_triton_trace_user_pid_1000_.ndjson",
        )
        _write_trace(
            trace_path,
            [
                _compilation("target_hash", "target_kernel"),
                _compilation("other_hash", "other_kernel"),
                _compilation("excluded_hash", "excluded_kernel"),
            ],
        )
        out_dir = os.path.join(self.temporary_directory.name, "out")

        oss_run(
            source=self.input_dir,
            out=out_dir,
            kernel_allowlist=" target*, , other_kernel ",
        )

        self.assertEqual(
            _compilation_hashes(out_dir),
            {"target_hash", "other_hash"},
        )

    def test_rejects_local_clp_before_overwriting_output(self) -> None:
        clp_dir = os.path.join(self.temporary_directory.name, "clp")
        os.makedirs(clp_dir)
        os.makedirs(os.path.join(clp_dir, "trace.clp"))
        out_dir = os.path.join(self.temporary_directory.name, "out")
        os.makedirs(out_dir)
        sentinel = os.path.join(out_dir, "keep.txt")
        with open(sentinel, "w") as output:
            output.write("keep")

        with self.assertRaisesRegex(RuntimeError, "LOCAL_CLP"):
            oss_run(
                source=clp_dir,
                out=out_dir,
                overwrite=True,
                kernel_allowlist="target*",
            )

        self.assertTrue(os.path.exists(sentinel))

    def test_empty_allowlist_copies_local_clp_unchanged(self) -> None:
        clp_dir = os.path.join(self.temporary_directory.name, "clp")
        os.makedirs(clp_dir)
        archive_dir = os.path.join(clp_dir, "trace.clp")
        os.makedirs(archive_dir)
        metadata_path = os.path.join(archive_dir, "metadata")
        with open(metadata_path, "w") as metadata:
            metadata.write("existing archive")
        out_dir = os.path.join(self.temporary_directory.name, "out")

        oss_run(
            source=clp_dir,
            out=out_dir,
            all_ranks=True,
            kernel_allowlist=" , , ",
        )

        copied_metadata = os.path.join(out_dir, "trace.clp", "metadata")
        with open(copied_metadata) as metadata:
            self.assertEqual(metadata.read(), "existing archive")

    def test_clp_backup_directory_does_not_block_filtering(self) -> None:
        os.makedirs(os.path.join(self.input_dir, "trace.clp.bak"))
        trace_path = os.path.join(
            self.input_dir,
            "dedicated_log_triton_trace_user_pid_1000_.ndjson",
        )
        _write_trace(trace_path, [_compilation("target_hash", "target_kernel")])
        out_dir = os.path.join(self.temporary_directory.name, "out")

        oss_run(
            source=self.input_dir,
            out=out_dir,
            kernel_allowlist="target*",
        )

        self.assertEqual(_compilation_hashes(out_dir), {"target_hash"})

    def test_no_match_preserves_existing_output(self) -> None:
        trace_path = os.path.join(
            self.input_dir,
            "dedicated_log_triton_trace_user_pid_1000_.ndjson",
        )
        _write_trace(trace_path, [_compilation("other_hash", "other_kernel")])
        out_dir = os.path.join(self.temporary_directory.name, "out")
        os.makedirs(out_dir)
        sentinel = os.path.join(out_dir, "keep.txt")
        with open(sentinel, "w") as output:
            output.write("keep")

        with self.assertRaisesRegex(RuntimeError, "No kernels matched"):
            oss_run(
                source=self.input_dir,
                out=out_dir,
                overwrite=True,
                kernel_allowlist="target*",
            )

        self.assertTrue(os.path.exists(sentinel))

    def test_parse_error_preserves_existing_output(self) -> None:
        out_dir = os.path.join(self.temporary_directory.name, "out")
        os.makedirs(out_dir)
        sentinel = os.path.join(out_dir, "keep.txt")
        with open(sentinel, "w") as output:
            output.write("keep")

        with (
            mock.patch.object(
                utils_module,
                "copy_local_to_tmpdir",
                return_value=self.input_dir,
            ),
            mock.patch.object(
                utils_module,
                "parse_logs",
                side_effect=RuntimeError("parse failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "parse failed"),
        ):
            oss_run(
                source=self.input_dir,
                out=out_dir,
                overwrite=True,
                kernel_allowlist="target*",
            )

        self.assertTrue(os.path.exists(sentinel))

    def test_parse_error_does_not_create_missing_output(self) -> None:
        out_dir = os.path.join(self.temporary_directory.name, "missing")

        with (
            mock.patch.object(
                utils_module,
                "copy_local_to_tmpdir",
                return_value=self.input_dir,
            ),
            mock.patch.object(
                utils_module,
                "parse_logs",
                side_effect=RuntimeError("parse failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "parse failed"),
        ):
            oss_run(
                source=self.input_dir,
                out=out_dir,
                kernel_allowlist="target*",
            )

        self.assertFalse(os.path.exists(out_dir))

    def test_existing_output_without_overwrite_fails_before_copy(self) -> None:
        out_dir = os.path.join(self.temporary_directory.name, "out")
        os.makedirs(out_dir)

        with (
            mock.patch.object(utils_module, "copy_local_to_tmpdir") as copy_mock,
            mock.patch.object(utils_module, "parse_logs") as parse_mock,
            self.assertRaisesRegex(RuntimeError, "pass --overwrite"),
        ):
            oss_run(source=self.input_dir, out=out_dir)

        copy_mock.assert_not_called()
        parse_mock.assert_not_called()

    def test_success_replaces_existing_output(self) -> None:
        trace_path = os.path.join(
            self.input_dir,
            "dedicated_log_triton_trace_user_pid_1000_.ndjson",
        )
        _write_trace(trace_path, [_compilation("target_hash", "target_kernel")])
        out_dir = os.path.join(self.temporary_directory.name, "out")
        os.makedirs(out_dir)
        sentinel = os.path.join(out_dir, "remove.txt")
        with open(sentinel, "w") as output:
            output.write("remove")

        oss_run(
            source=self.input_dir,
            out=out_dir,
            overwrite=True,
            kernel_allowlist="target*",
        )

        self.assertFalse(os.path.exists(sentinel))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "log_file_list.json")))
