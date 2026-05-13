# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for parse_single_rank — the multi-input batch entry point that merges
events across multiple trace files (typically PID files for one rank).

When subprocess compile workers serve the same inductor frame, their per-PID
output filenames collide; cross-PID merge by kernel_hash produces one
correct output per frame.
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


def _autotune_stack(user_call_id: str = "0") -> list[dict]:
    """Build a stack that `get_autotune_session_id` recognizes as autotune.

    The user-level frame must be stable across PIDs so the session_id hash
    matches; vary `user_call_id` only across distinct call sites.
    """
    return [
        {
            "filename": f"/user/code_{user_call_id}.py",
            "name": "user_func",
            "line": 10,
        },
        # Autotuner boundary frame — `get_autotune_session_id` and
        # `_is_autotune_benchmark_launch` both recognize this filename
        # + name combination (see sourcemap_utils.py).
        {
            "filename": "triton/runtime/autotuner.py",
            "name": "_bench",
            "line": 100,
        },
    ]


def _make_autotune_compilation_event(
    kernel_hash: str,
    kernel_name: str,
    *,
    user_call_id: str = "0",
    num_warps: int = 4,
    num_stages: int = 2,
    pid: int = 12345,
) -> dict:
    """Compilation event whose stack triggers autotune session attribution.

    `num_warps` / `num_stages` end up in `compilation_analysis.configs`,
    so vary them across hashes to make the per-config view distinguishable.
    """
    return {
        "event_type": "compilation",
        "pid": pid,
        "timestamp": "2026-05-06T00:00:00",
        "stack": _autotune_stack(user_call_id),
        "payload": {
            "metadata": {
                "hash": kernel_hash,
                "name": kernel_name,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "num_ctas": 1,
            },
            "pt_info": {"frame_id": 0, "frame_compile_id": 0},
            "file_content": {},
            "file_path": {},
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


class DedupCompilationsByHashUnitTest(unittest.TestCase):
    """Unit tests for `_dedup_compilations_by_hash`.

    The helper is the foundation for cross-PID autotune analysis
    correctness — N PIDs hitting the same Triton cache must collapse
    to 1 logical config, not be counted as N.
    """

    def test_empty_input(self) -> None:
        from tritonparse.parse.event_diff import _dedup_compilations_by_hash

        self.assertEqual(_dedup_compilations_by_hash([]), [])

    def test_all_unique_passes_through(self) -> None:
        from tritonparse.parse.event_diff import _dedup_compilations_by_hash

        events = [_make_compilation_event(f"hash_{i}", f"k_{i}") for i in range(3)]
        result = _dedup_compilations_by_hash(events)
        self.assertEqual(
            [e["payload"]["metadata"]["hash"] for e in result],
            ["hash_0", "hash_1", "hash_2"],
        )

    def test_duplicates_collapsed_first_wins(self) -> None:
        """[A, B, A, C] → [A, B, C]; the second A (different PID) is dropped."""
        from tritonparse.parse.event_diff import _dedup_compilations_by_hash

        a1 = _make_compilation_event("hash_A", "k_A", pid=1)
        b = _make_compilation_event("hash_B", "k_B", pid=1)
        a2 = _make_compilation_event("hash_A", "k_A", pid=2)
        c = _make_compilation_event("hash_C", "k_C", pid=2)
        result = _dedup_compilations_by_hash([a1, b, a2, c])
        self.assertEqual(
            [e["payload"]["metadata"]["hash"] for e in result],
            ["hash_A", "hash_B", "hash_C"],
        )
        # First-seen wins: pid=1's hash_A is kept, pid=2's is dropped.
        self.assertEqual(result[0]["pid"], 1)

    def test_event_without_hash_kept_defensively(self) -> None:
        """Malformed events without a hash should NOT be silently dropped."""
        from tritonparse.parse.event_diff import _dedup_compilations_by_hash

        broken = {
            "event_type": "compilation",
            "pid": 1,
            "payload": {"metadata": {}, "pt_info": {}},
        }
        good = _make_compilation_event("hash_A", "k_A")
        result = _dedup_compilations_by_hash([broken, good])
        self.assertEqual(len(result), 2)


class CrossPidAutotuneDedupE2ETest(unittest.TestCase):
    """End-to-end: parse_single_rank + cross-PID autotune session merge.

    Verifies the analysis-side fix in `_generate_autotune_analysis_events`:
    duplicate compilation events from different PIDs that share the same
    autotune session_id and the same kernel hash must NOT be mis-counted
    as multiple distinct benchmark configs.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="tritonparse_autotune_dedup_")
        self.input_dir = os.path.join(self.tmpdir, "input")
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_pid(self, pid: int, hashes: list[str]) -> str:
        """Write a per-PID trace file containing one autotune compilation per hash.

        All compilations share the same `_autotune_stack(user_call_id="0")`
        so they collapse to a single autotune session_id.
        """
        path = os.path.join(
            self.input_dir, f"dedicated_log_triton_trace_user_pid_{pid}_.ndjson"
        )
        events = []
        for i, h in enumerate(hashes):
            events.append(
                _make_autotune_compilation_event(
                    h,
                    f"k_{h}",
                    user_call_id="0",
                    # Vary num_warps so per-config views are distinguishable
                    # in the positive test.
                    num_warps=4 + i,
                    pid=pid,
                )
            )
        _write_trace_file(events, path)
        return path

    def _autotune_analysis_events(self) -> list[dict]:
        """Collect autotune_analysis events across all output files."""
        out: list[dict] = []
        for fname in os.listdir(self.output_dir):
            if not fname.endswith(".ndjson"):
                continue
            for ev in _read_jsonl(os.path.join(self.output_dir, fname)):
                if ev.get("event_type") == "autotune_analysis":
                    out.append(ev)
        return out

    def test_two_pids_same_session_same_hash_no_autotune_analysis(self) -> None:
        """2 PIDs + same session_id + same single hash → NOT a benchmark session.

        Without dedup, `len(compilation_events)` is 2 (one per PID),
        which would falsely satisfy the `>= 2` benchmark threshold and
        emit an autotune_analysis event.
        """
        path_a = self._write_pid(pid=1001, hashes=["hash_X"])
        path_b = self._write_pid(pid=1002, hashes=["hash_X"])

        parse_single_rank([path_a, path_b], self.output_dir)

        analyses = self._autotune_analysis_events()
        self.assertEqual(
            analyses,
            [],
            f"Expected no autotune_analysis (single hash collapses to 1 config); "
            f"got {len(analyses)}: {analyses}",
        )

    def test_two_pids_same_session_two_hashes_one_analysis_with_two_configs(
        self,
    ) -> None:
        """Same session + 2 distinct hashes across 2 PIDs → 1 analysis, 2 configs.

        Without dedup, this would emit `compilation_analysis.configs` with
        4 entries (2 hashes × 2 PIDs). With dedup, exactly 2 entries.
        """
        path_a = self._write_pid(pid=1001, hashes=["hash_A", "hash_B"])
        path_b = self._write_pid(pid=1002, hashes=["hash_A", "hash_B"])

        parse_single_rank([path_a, path_b], self.output_dir)

        analyses = self._autotune_analysis_events()
        self.assertEqual(
            len(analyses), 1, f"Expected 1 autotune_analysis; got {analyses}"
        )

        ca = analyses[0].get("compilation_analysis", {})
        configs = ca.get("configs", [])
        config_hashes = sorted(c.get("compilation_hash") for c in configs)
        self.assertEqual(
            config_hashes,
            ["hash_A", "hash_B"],
            f"Expected exactly 2 deduped configs; got {len(configs)}: {configs}",
        )
        # compilation_hashes (the parallel list) should also be deduped.
        self.assertEqual(sorted(ca.get("compilation_hashes", [])), ["hash_A", "hash_B"])
