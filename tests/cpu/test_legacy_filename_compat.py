# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for the legacy filename compatibility paths in parse_logs.

Covers:
- Legacy `dedicated_log_triton_trace_{user}_rank_{N}_.ndjson` (no PID suffix)
  parses through the new parse_single_rank path.
- Legacy no-rank `dedicated_log_triton_trace_{user}_.ndjson` parses correctly.
- Mixed old (no-PID) + new (PID-tagged) files in the same rank land in one
  bucket and merge into one output directory (no per-file subdir).
- _collect_and_bucket_files Pass 2 (pre-init re-attribution) is gated by
  enable_pre_init_attribution; with the gate off, behavior matches the
  pre-attribution version.
- When Pass 2 is enabled, legacy no-PID no-rank files still stay in the
  NO_RANK bucket (cannot be re-attributed without a PID).
"""

import gzip
import json
import os
import tempfile
import unittest

from tritonparse.parse.common import (
    _collect_and_bucket_files,
    parse_logs,
    Rank,
    RankConfig,
)


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


def _collect_compilation_hashes(parsed_dir: str) -> set[str]:
    """Walk parsed_dir, decompress every non-mapped .ndjson.gz, return hash set."""
    hashes: set[str] = set()
    for root, _, files in os.walk(parsed_dir):
        for fname in files:
            if not fname.endswith(".ndjson.gz") or "mapped" in fname.lower():
                continue
            for ev in _read_jsonl_gz(os.path.join(root, fname)):
                if ev.get("event_type") == "compilation":
                    hashes.add(ev["payload"]["metadata"]["hash"])
    return hashes


class CollectAndBucketFilesTest(unittest.TestCase):
    """Direct tests of _collect_and_bucket_files — no parsing, just bucketing."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="tritonparse_bucket_")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _touch(self, name: str) -> str:
        path = os.path.join(self.tmpdir, name)
        open(path, "w").close()
        return path

    def test_legacy_no_pid_files_with_rank_only(self) -> None:
        """Legacy `..._rank_N_.ndjson` (no PID) lands in the rank N bucket."""
        a = self._touch("dedicated_log_triton_trace_user_rank_0_.ndjson")
        b = self._touch("dedicated_log_triton_trace_user_rank_1_.ndjson")

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(all_ranks=True),
            enable_pre_init_attribution=False,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {0, 1})
        self.assertEqual(keyed[0], [a])
        self.assertEqual(keyed[1], [b])

    def test_legacy_no_pid_no_rank_file_lands_in_no_rank_bucket(self) -> None:
        """Legacy `..._user_.ndjson` (no rank, no PID) → NO_RANK with --all-ranks."""
        a = self._touch("dedicated_log_triton_trace_user_.ndjson")

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(all_ranks=True),
            enable_pre_init_attribution=False,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {Rank.NO_RANK})
        self.assertEqual(keyed[Rank.NO_RANK], [a])

    def test_mixed_legacy_and_pid_files_same_rank(self) -> None:
        """Old (no-PID) + new (PID-tagged) same-rank files share one bucket.

        Sort order: legacy no-PID file first (key -1), then PID-tagged files
        in ascending PID order.
        """
        legacy = self._touch("dedicated_log_triton_trace_user_rank_0_.ndjson")
        pid_a = self._touch("dedicated_log_triton_trace_user_rank_0_pid_1001_.ndjson")
        pid_b = self._touch("dedicated_log_triton_trace_user_rank_0_pid_1002_.ndjson")

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(rank=Rank(0)),
            enable_pre_init_attribution=False,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {0})
        self.assertEqual(keyed[0], [legacy, pid_a, pid_b])

    def test_pass2_disabled_by_default_pid_no_rank_stays_in_no_rank(self) -> None:
        """Default: PID-tagged no-rank file stays NO_RANK even when matching ranked file exists."""
        ranked = self._touch("dedicated_log_triton_trace_user_rank_0_pid_1001_.ndjson")
        no_rank = self._touch("dedicated_log_triton_trace_user_pid_1001_.ndjson")

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(all_ranks=True),
            enable_pre_init_attribution=False,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {0, Rank.NO_RANK})
        self.assertEqual(keyed[0], [ranked])
        self.assertEqual(keyed[Rank.NO_RANK], [no_rank])

    def test_pass2_enabled_reattributes_pid_no_rank_to_matching_rank(self) -> None:
        """Pass 2 on: PID-matched no-rank file is moved into the ranked bucket."""
        ranked = self._touch("dedicated_log_triton_trace_user_rank_0_pid_1001_.ndjson")
        no_rank = self._touch("dedicated_log_triton_trace_user_pid_1001_.ndjson")

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(all_ranks=True),
            enable_pre_init_attribution=True,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {0})
        # Sort order: no-rank file (no PID-suffix-after-rank wouldn't matter
        # here — both extract PID 1001, but `no_rank` has shorter basename
        # and sorts first by tuple key; pid_a path sorts second).
        self.assertCountEqual(keyed[0], [ranked, no_rank])

    def test_pass2_reattributes_rank_none_schema_no_rank(self) -> None:
        """New `_rank_none_` no-rank schema re-attributes via (host, pid) like the legacy no-token form.

        Verifies that the writer's `_rank_none_` token (always-present
        rank field) is parsed as rank=None by `parse_trace_filename_metadata`
        and then re-attributed by Pass 2 the same way a legacy no-rank
        file is. Without this, switching the writer to emit `_rank_none_`
        would silently drop pre-init kernels from their owning rank.
        """
        ranked = self._touch(
            "dedicated_log_triton_trace_user_rank_0_pid_1001_host_h_.ndjson"
        )
        no_rank_new = self._touch(
            "dedicated_log_triton_trace_user_rank_none_pid_1001_host_h_.ndjson"
        )

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(all_ranks=True),
            enable_pre_init_attribution=True,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {0})
        self.assertCountEqual(keyed[0], [ranked, no_rank_new])

    def test_pass2_legacy_no_pid_no_rank_stays_no_rank(self) -> None:
        """Pass 2 on: legacy no-PID no-rank file stays NO_RANK (no PID to match)."""
        ranked = self._touch("dedicated_log_triton_trace_user_rank_0_pid_1001_.ndjson")
        legacy_no_rank = self._touch("dedicated_log_triton_trace_user_.ndjson")

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(all_ranks=True),
            enable_pre_init_attribution=True,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {0, Rank.NO_RANK})
        self.assertEqual(keyed[0], [ranked])
        self.assertEqual(keyed[Rank.NO_RANK], [legacy_no_rank])

    def test_rank_none_disables_pass2_even_when_enabled(self) -> None:
        """`--rank none` keeps no-rank files unattributed even with Pass 2 on."""
        self._touch("dedicated_log_triton_trace_user_rank_0_pid_1001_.ndjson")
        no_rank = self._touch("dedicated_log_triton_trace_user_pid_1001_.ndjson")

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(rank=Rank(Rank.NO_RANK)),
            enable_pre_init_attribution=True,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {Rank.NO_RANK})
        self.assertEqual(keyed[Rank.NO_RANK], [no_rank])

    def test_parse_metadata_extracts_host_from_new_filename(self) -> None:
        """parse_trace_filename_metadata pulls the hostname from a host-suffixed name."""
        from tritonparse.parse.common import parse_trace_filename_metadata

        self.assertEqual(
            parse_trace_filename_metadata(
                "dedicated_log_triton_trace_user_rank_0_pid_12345_host_devgpu001_.ndjson"
            ).host,
            "devgpu001",
        )
        # No-rank variant
        self.assertEqual(
            parse_trace_filename_metadata(
                "dedicated_log_triton_trace_user_pid_12345_host_devgpu001_.ndjson"
            ).host,
            "devgpu001",
        )

    def test_parse_metadata_returns_none_for_legacy_no_host(self) -> None:
        """Legacy filenames (no host suffix) return host=None."""
        from tritonparse.parse.common import parse_trace_filename_metadata

        self.assertIsNone(
            parse_trace_filename_metadata(
                "dedicated_log_triton_trace_user_rank_0_pid_12345_.ndjson"
            ).host
        )
        self.assertIsNone(
            parse_trace_filename_metadata(
                "dedicated_log_triton_trace_user_.ndjson"
            ).host
        )

    def test_two_hosts_same_pid_different_ranks_attributed_correctly(self) -> None:
        """(host, pid) tuple key — same PID across hosts maps to distinct ranks.

        This is the core multi-host correctness scenario: without the host
        component in the map key, the second host's no-rank file would be
        silently mis-attributed to the first host's rank.
        """
        a_ranked = self._touch(
            "dedicated_log_triton_trace_user_rank_0_pid_12345_host_hosta_.ndjson"
        )
        b_ranked = self._touch(
            "dedicated_log_triton_trace_user_rank_4_pid_12345_host_hostb_.ndjson"
        )
        a_no_rank = self._touch(
            "dedicated_log_triton_trace_user_pid_12345_host_hosta_.ndjson"
        )
        b_no_rank = self._touch(
            "dedicated_log_triton_trace_user_pid_12345_host_hostb_.ndjson"
        )

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(all_ranks=True),
            enable_pre_init_attribution=True,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(set(keyed), {0, 4})
        self.assertCountEqual(keyed[0], [a_ranked, a_no_rank])
        self.assertCountEqual(keyed[4], [b_ranked, b_no_rank])
        self.assertNotIn(Rank.NO_RANK, keyed)

    def test_legacy_no_host_doesnt_match_new_host_under_pass2(self) -> None:
        """(None, pid) and (host, pid) are distinct keys — no false attribution."""
        legacy_ranked = self._touch(
            "dedicated_log_triton_trace_user_rank_0_pid_12345_.ndjson"  # no host
        )
        new_no_rank = self._touch(
            "dedicated_log_triton_trace_user_pid_12345_host_devgpu001_.ndjson"
        )

        buckets = _collect_and_bucket_files(
            self.tmpdir,
            RankConfig(all_ranks=True),
            enable_pre_init_attribution=True,
        )

        keyed = {r.value: files for r, files in buckets.items()}
        self.assertEqual(keyed[0], [legacy_ranked])
        self.assertEqual(keyed[Rank.NO_RANK], [new_no_rank])


class LegacyParseLogsTest(unittest.TestCase):
    """End-to-end parse_logs runs that exercise legacy + mixed inputs."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="tritonparse_legacy_compat_")
        self.input_dir = os.path.join(self.tmpdir, "input")
        os.makedirs(self.input_dir)

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parse_legacy_filename_with_rank_only(self) -> None:
        """Legacy `..._rank_0_.ndjson` (no PID) parses through parse_single_rank."""
        path = os.path.join(
            self.input_dir, "dedicated_log_triton_trace_user_rank_0_.ndjson"
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_legacy", "kernel_legacy", frame_id=0, frame_compile_id=0
                ),
                _make_launch_event("hash_legacy", "kernel_legacy"),
            ],
            path,
        )

        parsed_dir, file_mapping = parse_logs(self.input_dir, RankConfig(rank=Rank(0)))

        self.assertIn("rank_0", file_mapping)
        self.assertEqual(_collect_compilation_hashes(parsed_dir), {"hash_legacy"})

    def test_parse_rank_none_schema_no_rank_filename(self) -> None:
        """New `..._rank_none_pid_M_host_H_.ndjson` parses as no-rank under --all-ranks.

        End-to-end check that the writer's `_rank_none_` token flows all
        the way through parse_logs into a rank_none output bucket.
        """
        path = os.path.join(
            self.input_dir,
            "dedicated_log_triton_trace_user_rank_none_pid_1001_host_h_.ndjson",
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_norank_new",
                    "kernel_norank_new",
                    frame_id=0,
                    frame_compile_id=0,
                    pid=1001,
                ),
                _make_launch_event("hash_norank_new", "kernel_norank_new", pid=1001),
            ],
            path,
        )

        parsed_dir, file_mapping = parse_logs(
            self.input_dir, RankConfig(all_ranks=True)
        )

        self.assertIn("rank_none", file_mapping)
        self.assertEqual(_collect_compilation_hashes(parsed_dir), {"hash_norank_new"})

    def test_parse_legacy_no_rank_filename(self) -> None:
        """Legacy `..._user_.ndjson` (no rank, no PID) parses under --all-ranks."""
        path = os.path.join(self.input_dir, "dedicated_log_triton_trace_user_.ndjson")
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_norank", "kernel_norank", frame_id=0, frame_compile_id=0
                ),
                _make_launch_event("hash_norank", "kernel_norank"),
            ],
            path,
        )

        parsed_dir, file_mapping = parse_logs(
            self.input_dir, RankConfig(all_ranks=True)
        )

        self.assertIn("rank_none", file_mapping)
        self.assertEqual(_collect_compilation_hashes(parsed_dir), {"hash_norank"})

    def test_parse_mixed_legacy_and_pid_files_merged(self) -> None:
        """Mixed legacy + PID files for one rank merge into one output dir.

        Regression for the deleted `use_filenames=True` workaround: previously,
        multiple files for the same rank would be parsed into per-file
        subdirectories. Now they merge into one rank_0/ output by kernel hash.
        """
        legacy = os.path.join(
            self.input_dir, "dedicated_log_triton_trace_user_rank_0_.ndjson"
        )
        new_pid = os.path.join(
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
            legacy,
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
            new_pid,
        )

        parsed_dir, _ = parse_logs(self.input_dir, RankConfig(rank=Rank(0)))

        # Both kernels merged into a single rank_0 output, NOT scattered into
        # per-file subdirectories.
        rank0_dir = os.path.join(parsed_dir, "rank_0")
        self.assertTrue(os.path.isdir(rank0_dir))
        outputs = sorted(os.listdir(rank0_dir))
        # Exactly one frame output gz (plus optional mapped file).
        frame_outputs = [
            o for o in outputs if "mapped" not in o.lower() and o.endswith(".gz")
        ]
        self.assertEqual(frame_outputs, ["f0_fc0_a0_cai-.ndjson.gz"])
        self.assertEqual(
            _collect_compilation_hashes(parsed_dir), {"hash_legacy", "hash_new"}
        )
