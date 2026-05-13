# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for parse_single_rank — the multi-input batch entry point that merges
events across multiple trace files (typically PID files for one rank).

See ~/ai_discussions/tritonparse/refactor/multiprocess_trace_filename_refactor.md
§7.2 (rationale) and §9.2 (test spec).
"""

import json
import os
import tempfile
import unittest

from tritonparse.parse.trace_processor import parse_single_rank


def _make_compilation_event(
    kernel_hash: str,
    kernel_name: str,
    *,
    frame_id: int | None = None,
    frame_compile_id: int | None = None,
    pid: int = 12345,
) -> dict:
    """Build a minimal compilation event suitable for parse_single_rank.

    Includes the bare-minimum payload structure that _determine_output_fname
    and parse_single_trace_content expect (no IR, no source mappings).
    """
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
    """Build a minimal launch event suitable for parse_single_rank."""
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


def _read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _events_by_type(events: list[dict], event_type: str) -> list[dict]:
    return [e for e in events if e.get("event_type") == event_type]


class MultiPidParseTest(unittest.TestCase):
    """parse_single_rank merges across multiple PID input files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tritonparse_multi_pid_")
        self.input_dir = os.path.join(self.tmpdir, "input")
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_multiple_pid_files_same_rank_same_frame(self):
        """Same frame_id across PIDs → one output file containing BOTH kernels.

        This is the §3.5 finding A scenario: subprocess compile workers serve
        the same inductor frame, producing the same `f0_fc0_..` output filename
        per PID. The pre-refactor parse_single_file would overwrite. After the
        parse_single_rank refactor, both kernels are merged into one file.
        """
        pid_a, pid_b = 1001, 1002

        path_a = os.path.join(
            self.input_dir, f"dedicated_log_triton_trace_user_pid_{pid_a}_.ndjson"
        )
        path_b = os.path.join(
            self.input_dir, f"dedicated_log_triton_trace_user_pid_{pid_b}_.ndjson"
        )

        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_A", "kernel_A", frame_id=0, frame_compile_id=0, pid=pid_a
                ),
                _make_launch_event("hash_A", "kernel_A", pid=pid_a),
            ],
            path_a,
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_B", "kernel_B", frame_id=0, frame_compile_id=0, pid=pid_b
                ),
                _make_launch_event("hash_B", "kernel_B", pid=pid_b),
            ],
            path_b,
        )

        parse_single_rank([path_a, path_b], self.output_dir)

        outputs = sorted(os.listdir(self.output_dir))
        # One merged output file (no per-PID subdirectories)
        self.assertEqual(outputs, ["f0_fc0_a0_cai-.ndjson"])

        events = _read_jsonl(os.path.join(self.output_dir, "f0_fc0_a0_cai-.ndjson"))
        compilation_hashes = {
            e["payload"]["metadata"]["hash"]
            for e in _events_by_type(events, "compilation")
        }
        self.assertEqual(compilation_hashes, {"hash_A", "hash_B"})

    def test_multiple_pid_files_disjoint_frames(self):
        """Different frame_ids across PIDs → two separate output files."""
        pid_a, pid_b = 1001, 1002

        path_a = os.path.join(
            self.input_dir, f"dedicated_log_triton_trace_user_pid_{pid_a}_.ndjson"
        )
        path_b = os.path.join(
            self.input_dir, f"dedicated_log_triton_trace_user_pid_{pid_b}_.ndjson"
        )

        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_A", "kernel_A", frame_id=0, frame_compile_id=0, pid=pid_a
                ),
                _make_launch_event("hash_A", "kernel_A", pid=pid_a),
            ],
            path_a,
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_B", "kernel_B", frame_id=1, frame_compile_id=0, pid=pid_b
                ),
                _make_launch_event("hash_B", "kernel_B", pid=pid_b),
            ],
            path_b,
        )

        parse_single_rank([path_a, path_b], self.output_dir)

        outputs = sorted(os.listdir(self.output_dir))
        self.assertEqual(outputs, ["f0_fc0_a0_cai-.ndjson", "f1_fc0_a0_cai-.ndjson"])

        f0_events = _read_jsonl(os.path.join(self.output_dir, "f0_fc0_a0_cai-.ndjson"))
        f1_events = _read_jsonl(os.path.join(self.output_dir, "f1_fc0_a0_cai-.ndjson"))
        f0_hashes = {
            e["payload"]["metadata"]["hash"]
            for e in _events_by_type(f0_events, "compilation")
        }
        f1_hashes = {
            e["payload"]["metadata"]["hash"]
            for e in _events_by_type(f1_events, "compilation")
        }
        self.assertEqual(f0_hashes, {"hash_A"})
        self.assertEqual(f1_hashes, {"hash_B"})

    def test_same_kernel_hash_across_pids_dedup(self):
        """Same kernel_hash in 2 PID files: first compilation wins, all launches kept.

        Mirrors the cross-PID Triton-cache-hit scenario: two workers
        independently compile the same kernel, each emitting their own
        compilation + launch records.
        """
        pid_a, pid_b = 1001, 1002

        path_a = os.path.join(
            self.input_dir, f"dedicated_log_triton_trace_user_pid_{pid_a}_.ndjson"
        )
        path_b = os.path.join(
            self.input_dir, f"dedicated_log_triton_trace_user_pid_{pid_b}_.ndjson"
        )

        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_X", "kernel_X", frame_id=0, frame_compile_id=0, pid=pid_a
                ),
                _make_launch_event("hash_X", "kernel_X", pid=pid_a),
                _make_launch_event("hash_X", "kernel_X", pid=pid_a),
            ],
            path_a,
        )
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_X", "kernel_X", frame_id=0, frame_compile_id=0, pid=pid_b
                ),
                _make_launch_event("hash_X", "kernel_X", pid=pid_b),
                _make_launch_event("hash_X", "kernel_X", pid=pid_b),
            ],
            path_b,
        )

        parse_single_rank([path_a, path_b], self.output_dir)

        outputs = sorted(os.listdir(self.output_dir))
        self.assertEqual(outputs, ["f0_fc0_a0_cai-.ndjson"])

        events = _read_jsonl(os.path.join(self.output_dir, "f0_fc0_a0_cai-.ndjson"))
        compilations = _events_by_type(events, "compilation")
        launches = _events_by_type(events, "launch")

        # Compilation deduped: only the first PID's compilation kept.
        self.assertEqual(len(compilations), 1)
        self.assertEqual(compilations[0]["pid"], pid_a)

        # Launches accumulated: 2 from each PID = 4 total.
        self.assertEqual(len(launches), 4)
        launch_pids = sorted(e["pid"] for e in launches)
        self.assertEqual(launch_pids, [pid_a, pid_a, pid_b, pid_b])

    def test_single_file_via_parse_single_rank_matches_parse_single_file(self):
        """parse_single_rank with a 1-element list = parse_single_file behavior."""
        from tritonparse.parse.trace_processor import parse_single_file

        path = os.path.join(self.input_dir, "log_pid_42_.ndjson")
        _write_trace_file(
            [
                _make_compilation_event(
                    "hash_A", "kernel_A", frame_id=0, frame_compile_id=0, pid=42
                ),
                _make_launch_event("hash_A", "kernel_A", pid=42),
            ],
            path,
        )

        out_rank = os.path.join(self.tmpdir, "out_rank")
        out_file = os.path.join(self.tmpdir, "out_file")
        os.makedirs(out_rank)
        os.makedirs(out_file)

        parse_single_rank([path], out_rank)
        parse_single_file(path, out_file)

        rank_files = sorted(os.listdir(out_rank))
        file_files = sorted(os.listdir(out_file))
        self.assertEqual(rank_files, file_files)

        for fname in rank_files:
            rank_events = _read_jsonl(os.path.join(out_rank, fname))
            file_events = _read_jsonl(os.path.join(out_file, fname))
            self.assertEqual(rank_events, file_events)
