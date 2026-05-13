# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
End-to-end tests for the Diff 5 user-visible behavior change: by default,
no-rank trace files whose PID matches a ranked file are re-attributed to
that rank, so kernels compiled before `dist.init_process_group` show up
under their owning rank.

Bucket-level Pass 2 unit tests live in `test_legacy_filename_compat.py`
(CollectAndBucketFilesTest); this file focuses on the e2e `parse_logs`
output contents and CLI plumbing.

See ~/ai_discussions/tritonparse/refactor/multiprocess_trace_filename_refactor.md
§7.7, §9.2.1, and §10 Diff 5.
"""

import gzip
import json
import os
import tempfile
import unittest
from unittest import mock

from tritonparse.parse import utils as utils_module
from tritonparse.parse.common import parse_logs, Rank, RankConfig
from tritonparse.parse.source_type import SourceType
from tritonparse.parse.utils import oss_run


def _make_compilation_event(
    kernel_hash: str,
    kernel_name: str,
    *,
    frame_id: int | None = None,
    frame_compile_id: int | None = None,
    pid: int = 12345,
) -> dict:
    pt_info = {}
    if frame_id is not None:
        pt_info["frame_id"] = frame_id
    if frame_compile_id is not None:
        pt_info["frame_compile_id"] = frame_compile_id
    return {
        "event_type": "compilation",
        "pid": pid,
        "timestamp": "2026-05-06T00:00:00",
        "stack": [],
        "payload": {
            "metadata": {"hash": kernel_hash, "name": kernel_name},
            "pt_info": pt_info,
            "file_content": {},
            "file_path": {},
        },
    }


def _make_launch_event(
    kernel_hash: str,
    kernel_name: str,
    *,
    pid: int = 12345,
) -> dict:
    return {
        "event_type": "launch",
        "pid": pid,
        "timestamp": "2026-05-06T00:00:01",
        "stack": [],
        "name": kernel_name,
        "compilation_metadata": {
            "hash": kernel_hash,
            "name": kernel_name,
            "num_warps": 4,
            "num_stages": 2,
            "num_ctas": 1,
        },
    }


def _write_trace_file(events: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _read_jsonl_gz(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def _hashes_under(parsed_dir: str, subdir: str) -> set[str]:
    """Return compilation hashes under parsed_dir/subdir/* (gz, non-mapped)."""
    target = os.path.join(parsed_dir, subdir)
    if not os.path.isdir(target):
        return set()
    hashes: set[str] = set()
    for fname in os.listdir(target):
        if not fname.endswith(".ndjson.gz") or "mapped" in fname.lower():
            continue
        for ev in _read_jsonl_gz(os.path.join(target, fname)):
            if ev.get("event_type") == "compilation":
                hashes.add(ev["payload"]["metadata"]["hash"])
    return hashes


class PreInitAttributionE2ETest(unittest.TestCase):
    """E2e tests for `parse_logs` with the new default `enable_pre_init_attribution=True`."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="tritonparse_pre_init_")
        self.input_dir = os.path.join(self.tmpdir, "input")
        os.makedirs(self.input_dir)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_pid_pair(self, pid: int) -> None:
        """Pre-init no-rank file (kernel A) + post-init rank_0 file (kernel B), same PID."""
        no_rank_path = os.path.join(
            self.input_dir,
            f"dedicated_log_triton_trace_user_pid_{pid}_.ndjson",
        )
        rank0_path = os.path.join(
            self.input_dir,
            f"dedicated_log_triton_trace_user_rank_0_pid_{pid}_.ndjson",
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_pre_init",
                    "kernel_pre_init",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=pid,
                ),
                _make_launch_event("hash_pre_init", "kernel_pre_init", pid=pid),
            ],
            no_rank_path,
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_post_init",
                    "kernel_post_init",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=pid,
                ),
                _make_launch_event("hash_post_init", "kernel_post_init", pid=pid),
            ],
            rank0_path,
        )

    def test_pre_init_kernels_merged_into_rank_0_by_default(self) -> None:
        """Default ON: pre-init kernel appears in rank_0 output."""
        self._write_pid_pair(pid=1001)

        parsed_dir, file_mapping = parse_logs(self.input_dir, RankConfig(rank=Rank(0)))

        self.assertIn("rank_0", file_mapping)
        self.assertEqual(
            _hashes_under(parsed_dir, "rank_0"),
            {"hash_pre_init", "hash_post_init"},
        )
        # NO_RANK bucket is filtered out under --rank N (no rank_none output).
        self.assertNotIn("rank_none", file_mapping)

    def test_no_pre_init_attribution_kwarg_keeps_rank_0_clean(self) -> None:
        """Explicit kwarg disables Pass 2: rank_0 only has post-init kernel."""
        self._write_pid_pair(pid=1001)

        parsed_dir, file_mapping = parse_logs(
            self.input_dir,
            RankConfig(rank=Rank(0)),
            enable_pre_init_attribution=False,
        )

        self.assertIn("rank_0", file_mapping)
        self.assertEqual(_hashes_under(parsed_dir, "rank_0"), {"hash_post_init"})

    def test_all_ranks_default_attributes_pid_matched_no_rank(self) -> None:
        """--all-ranks + default ON: PID-matched no-rank file is absorbed into rank_0."""
        self._write_pid_pair(pid=1001)

        parsed_dir, file_mapping = parse_logs(
            self.input_dir, RankConfig(all_ranks=True)
        )

        self.assertEqual(
            _hashes_under(parsed_dir, "rank_0"),
            {"hash_pre_init", "hash_post_init"},
        )
        # No-rank bucket is empty after attribution → no rank_none entry.
        self.assertNotIn("rank_none", file_mapping)

    def test_rank_none_keeps_no_rank_events_visible(self) -> None:
        """--rank none always shows the unattributed view, regardless of default."""
        self._write_pid_pair(pid=1001)

        parsed_dir, file_mapping = parse_logs(
            self.input_dir, RankConfig(rank=Rank(Rank.NO_RANK))
        )

        self.assertIn("rank_none", file_mapping)
        self.assertEqual(_hashes_under(parsed_dir, ""), {"hash_pre_init"})
        # rank_0 file is filtered out under --rank none.
        self.assertNotIn("rank_0", file_mapping)

    def test_legacy_no_pid_no_rank_remains_no_rank_under_default(self) -> None:
        """Legacy no-PID no-rank file isn't auto-attributed (no PID to match)."""
        legacy_path = os.path.join(
            self.input_dir, "dedicated_log_triton_trace_user_.ndjson"
        )
        rank0_path = os.path.join(
            self.input_dir,
            "dedicated_log_triton_trace_user_rank_0_pid_1001_.ndjson",
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_legacy",
                    "kernel_legacy",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=999,
                ),
                _make_launch_event("hash_legacy", "kernel_legacy", pid=999),
            ],
            legacy_path,
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_new",
                    "kernel_new",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=1001,
                ),
                _make_launch_event("hash_new", "kernel_new", pid=1001),
            ],
            rank0_path,
        )

        parsed_dir, file_mapping = parse_logs(
            self.input_dir, RankConfig(all_ranks=True)
        )

        # Legacy file stays in NO_RANK bucket; new file goes to rank_0.
        self.assertEqual(_hashes_under(parsed_dir, "rank_0"), {"hash_new"})
        self.assertEqual(_hashes_under(parsed_dir, ""), {"hash_legacy"})

    def test_single_gpu_only_no_rank_files_stays_no_rank(self) -> None:
        """No ranked files exist → pid_to_rank empty → no attribution can happen."""
        no_rank_path = os.path.join(
            self.input_dir, "dedicated_log_triton_trace_user_pid_1001_.ndjson"
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_solo",
                    "kernel_solo",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=1001,
                ),
                _make_launch_event("hash_solo", "kernel_solo", pid=1001),
            ],
            no_rank_path,
        )

        parsed_dir, file_mapping = parse_logs(
            self.input_dir, RankConfig(all_ranks=True)
        )

        self.assertIn("rank_none", file_mapping)
        self.assertEqual(_hashes_under(parsed_dir, ""), {"hash_solo"})
        self.assertNotIn("rank_0", file_mapping)

    def _write_pid_pair_with_host(
        self,
        pid: int,
        host: str,
        pre_hash: str,
        post_hash: str,
        rank: int = 0,
    ) -> None:
        """Pre-init no-rank file + post-init ranked file, both with host suffix."""
        no_rank_path = os.path.join(
            self.input_dir,
            f"dedicated_log_triton_trace_user_pid_{pid}_host_{host}_.ndjson",
        )
        ranked_path = os.path.join(
            self.input_dir,
            f"dedicated_log_triton_trace_user_rank_{rank}_pid_{pid}_host_{host}_.ndjson",
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    pre_hash, "k_pre", frame_id=0, frame_compile_id=0, pid=pid
                ),
                _make_launch_event(pre_hash, "k_pre", pid=pid),
            ],
            no_rank_path,
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    post_hash, "k_post", frame_id=0, frame_compile_id=0, pid=pid
                ),
                _make_launch_event(post_hash, "k_post", pid=pid),
            ],
            ranked_path,
        )

    def test_two_hosts_same_pid_attributed_to_correct_rank(self) -> None:
        """Multi-host fix: same PID on two hosts → each no-rank file → its host's rank.

        This is the multi-host correctness scenario. Without (host, pid)
        tuple keys in the attribution map, host_b's no-rank file would
        be silently mis-attributed to host_a's rank.
        """
        self._write_pid_pair_with_host(
            pid=12345,
            host="hosta",
            pre_hash="ha_pre",
            post_hash="ha_post",
            rank=0,
        )
        self._write_pid_pair_with_host(
            pid=12345,
            host="hostb",
            pre_hash="hb_pre",
            post_hash="hb_post",
            rank=4,
        )

        parsed_dir, file_mapping = parse_logs(
            self.input_dir, RankConfig(all_ranks=True)
        )

        self.assertEqual(_hashes_under(parsed_dir, "rank_0"), {"ha_pre", "ha_post"})
        self.assertEqual(_hashes_under(parsed_dir, "rank_4"), {"hb_pre", "hb_post"})
        self.assertNotIn("rank_none", file_mapping)

    def test_legacy_no_host_files_dont_match_new_host_files_e2e(self) -> None:
        """Legacy no-host file + new host file with same PID — namespaces stay separate."""
        legacy_pid = 999
        legacy_no_rank = os.path.join(
            self.input_dir,
            f"dedicated_log_triton_trace_user_pid_{legacy_pid}_.ndjson",
        )
        legacy_ranked = os.path.join(
            self.input_dir,
            f"dedicated_log_triton_trace_user_rank_0_pid_{legacy_pid}_.ndjson",
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "legacy_pre",
                    "k_lp",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=legacy_pid,
                ),
                _make_launch_event("legacy_pre", "k_lp", pid=legacy_pid),
            ],
            legacy_no_rank,
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "legacy_post",
                    "k_lpost",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=legacy_pid,
                ),
                _make_launch_event("legacy_post", "k_lpost", pid=legacy_pid),
            ],
            legacy_ranked,
        )
        # New host file with the same PID — must NOT attribute to rank_0.
        new_no_rank = os.path.join(
            self.input_dir,
            f"dedicated_log_triton_trace_user_pid_{legacy_pid}_host_devgpu001_.ndjson",
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "new_pre",
                    "k_np",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=legacy_pid,
                ),
                _make_launch_event("new_pre", "k_np", pid=legacy_pid),
            ],
            new_no_rank,
        )

        parsed_dir, _ = parse_logs(self.input_dir, RankConfig(all_ranks=True))

        # Legacy (None, 999) attribution: pre + post in rank_0.
        # New (devgpu001, 999) has no matching ranked file → NO_RANK bucket.
        self.assertEqual(
            _hashes_under(parsed_dir, "rank_0"), {"legacy_pre", "legacy_post"}
        )
        self.assertEqual(_hashes_under(parsed_dir, ""), {"new_pre"})


class CliPlumbingTest(unittest.TestCase):
    """Verify --no-pre-init-attribution flows through oss_run → parse_logs.

    We patch on `utils_module` (not `common_module`) because oss_run uses
    the `parse_logs` reference imported via `from .common import parse_logs`
    into `tritonparse.parse.utils`'s namespace.
    """

    def _run_oss_with_mocks(self, **oss_kwargs) -> mock.Mock:
        """Invoke oss_run with parse_logs / IO mocked out and return the parse_logs mock."""
        fake_source = mock.Mock()
        fake_source.type = SourceType.LOCAL
        fake_source.value = "/tmp/in"
        with (
            mock.patch.object(
                utils_module,
                "parse_logs",
                return_value=("/tmp/dummy", {}),
            ) as parse_mock,
            mock.patch.object(
                utils_module, "copy_local_to_tmpdir", return_value="/tmp/in"
            ),
            mock.patch.object(utils_module, "print_parsed_files_summary"),
            mock.patch.object(utils_module, "Source", return_value=fake_source),
        ):
            oss_run(source="/tmp/in", **oss_kwargs)
        return parse_mock

    def test_oss_run_no_pre_init_attribution_true_disables(self) -> None:
        """oss_run(no_pre_init_attribution=True) → parse_logs(enable_pre_init_attribution=False)."""
        parse_mock = self._run_oss_with_mocks(no_pre_init_attribution=True)
        parse_mock.assert_called_once()
        self.assertEqual(
            parse_mock.call_args.kwargs["enable_pre_init_attribution"], False
        )

    def test_oss_run_default_enables_attribution(self) -> None:
        """oss_run() default → parse_logs(enable_pre_init_attribution=True)."""
        parse_mock = self._run_oss_with_mocks()
        self.assertEqual(
            parse_mock.call_args.kwargs["enable_pre_init_attribution"], True
        )
