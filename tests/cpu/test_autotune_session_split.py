# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Autotune sessions must split by key, not just by call site.

Two ``Autotuner.run()`` invocations at the same call site with different keys
(e.g. ``M=N=16`` then ``M=N=32`` in a loop) share one call-site session id.
The parser must still emit one ``autotune_analysis`` event per key round, so
per-config arguments stay single-valued in each session's varies table.
"""

import json
import os
import tempfile
import unittest

from tritonparse.parse.event_diff import (
    _config_value_matches,
    _extract_scalar_value,
    _parse_best_config,
)
from tritonparse.parse.trace_processor import parse_single_rank


def _stack(boundary: str) -> list[dict]:
    # Identical user frame for every event: both key rounds run at the same
    # call site, so they share one coarse (call-site) session id.
    return [
        {"filename": "/user/loop.py", "name": "run_kernel", "line": 42},
        {"filename": "triton/runtime/autotuner.py", "name": boundary, "line": 100},
    ]


def _compilation(kernel_hash: str) -> dict:
    return {
        "event_type": "compilation",
        "pid": 1000,
        "timestamp": "2026-09-02T00:00:00",
        "stack": _stack("_bench"),
        "payload": {
            "metadata": {
                "hash": kernel_hash,
                "name": "matmul_kernel",
                "num_warps": 1,
                "num_stages": 1,
                "num_ctas": 1,
            },
            "pt_info": {"frame_id": 0, "frame_compile_id": 0},
            "file_content": {},
            "file_path": {},
        },
    }


def _extracted_args(m: int, block_m: int, data_ptr: str) -> dict:
    return {
        "a": {
            "type": "tensor",
            "shape": [m, 16],
            "dtype": "torch.float16",
            "stride": [16, 1],
            "data_ptr": data_ptr,
        },
        "M": {"type": "int", "value": m},
        "N": {"type": "int", "value": m},
        "BLOCK_SIZE_M": {"type": "int", "value": block_m},
    }


def _launch(
    kernel_hash: str,
    m: int,
    block_m: int,
    boundary: str,
    data_ptr: str,
) -> dict:
    return {
        "event_type": "launch",
        "pid": 1000,
        "timestamp": "2026-09-02T00:00:01",
        "name": "matmul_kernel",
        "stack": _stack(boundary),
        "compilation_metadata": {
            "hash": kernel_hash,
            "name": "matmul_kernel",
            "num_warps": 1,
            "num_stages": 1,
            "num_ctas": 1,
        },
        "extracted_args": _extracted_args(m, block_m, data_ptr),
    }


def _autotune_event(m: int, block_m: int) -> dict:
    return {
        "event_type": "autotune",
        "kernel_name": "matmul_kernel",
        "stack": _stack("run"),
        "best_config": (
            f"BLOCK_SIZE_M: {block_m}, BLOCK_SIZE_N: 16, "
            "num_warps: 1, num_stages: 1, num_ctas: 1"
        ),
        "configs_timings": {"cfg": 1.0},
        "duration": 0.5,
        "cache_hit": False,
        "cache_key": "cache-key",
        "autotune_key": f"({m}, {m}, 'torch.float16')",
    }


def _write_trace(path: str, events: list[dict]) -> None:
    with open(path, "w") as output:
        for event in events:
            output.write(json.dumps(event) + "\n")


def _read_output_events(output_dir: str) -> list[dict]:
    events = []
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".ndjson"):
            with open(os.path.join(output_dir, filename), "r") as source:
                events.extend(json.loads(line) for line in source if line.strip())
    return events


def _two_key_trace() -> list[dict]:
    """Two key rounds (M=16, M=32) at one call site, two configs each."""
    return [
        _compilation("hash_a"),
        _compilation("hash_b"),
        # Round 1: M = N = 16, winner is hash_a (BLOCK_SIZE_M 16).
        _launch("hash_a", 16, 16, "_bench", "0x1000"),
        _launch("hash_b", 16, 32, "_bench", "0x2000"),
        _launch("hash_a", 16, 16, "run", "0x3000"),
        _autotune_event(16, 16),
        # Round 2: M = N = 32, winner is hash_b (BLOCK_SIZE_M 32).
        _launch("hash_a", 32, 16, "_bench", "0x4000"),
        _launch("hash_b", 32, 32, "_bench", "0x5000"),
        _launch("hash_b", 32, 32, "run", "0x6000"),
        _autotune_event(32, 32),
    ]


class AutotuneSessionSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(
            self.temporary_directory.name,
            "dedicated_log_triton_trace_user_pid_1000_.ndjson",
        )
        self.output_dir = os.path.join(self.temporary_directory.name, "output")
        os.makedirs(self.output_dir)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_two_keys_share_call_site_but_split_into_two_sessions(self) -> None:
        _write_trace(self.input_path, _two_key_trace())
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 2)
        by_winner = {a["winner_compilation_hash"]: a for a in analyses}
        self.assertEqual(set(by_winner), {"hash_a", "hash_b"})

        # Each key round keeps single-valued args: M/N are constant
        # within the round (scalars in the sames section), only the true
        # config param varies, and no cell is a multi-valued
        # {unique_count, values} distribution (which the viewer renders
        # as an ellipsis).
        for winner_hash, analysis in by_winner.items():
            configs = analysis["autotune_args_summary"]["autotune_configs"]
            varies = configs["varies"]
            sames = configs["sames"]
            expected_m = 16 if winner_hash == "hash_a" else 32
            self.assertEqual(sames["M"], {"type": "int", "value": expected_m})
            self.assertEqual(sames["N"], {"type": "int", "value": expected_m})
            self.assertEqual(set(varies), {"BLOCK_SIZE_M"})
            self.assertEqual(
                varies["BLOCK_SIZE_M"]["hash_a"], {"type": "int", "value": 16}
            )
            self.assertEqual(
                varies["BLOCK_SIZE_M"]["hash_b"], {"type": "int", "value": 32}
            )
            self.assertNotIn("unique_count", json.dumps(configs))

        # Winners and occurrence ids stay with their own key round.
        self.assertEqual(by_winner["hash_a"]["launch_occurrence_ids"]["winner"], [4])
        self.assertEqual(by_winner["hash_b"]["launch_occurrence_ids"]["winner"], [7])
        self.assertEqual(
            sorted(by_winner["hash_a"]["launch_occurrence_ids"]["benchmark"]),
            [2, 3],
        )
        self.assertEqual(
            sorted(by_winner["hash_b"]["launch_occurrence_ids"]["benchmark"]),
            [5, 6],
        )

        # Each session keeps its own autotune result, matched by best config.
        self.assertIn(
            "BLOCK_SIZE_M: 16", by_winner["hash_a"]["autotune_result"]["best_config"]
        )
        self.assertIn(
            "BLOCK_SIZE_M: 32", by_winner["hash_b"]["autotune_result"]["best_config"]
        )

        # The two sub-sessions are distinguishable by their full key
        # signatures (never truncated: truncation could collide and drop
        # a key round).
        session_ids = {a["session_id"] for a in analyses}
        self.assertEqual(len(session_ids), 2)
        for session_id in session_ids:
            coarse, _, sig = session_id.partition(":")
            self.assertTrue(coarse)
            self.assertEqual(len(sig), 16)

    def test_two_callbacks_without_args_share_one_partition(self) -> None:
        # Degraded but valid trace shape: two key rounds (distinct
        # listener callbacks) whose launches carry no extracted_args.
        # The launches are indistinguishable, so no key attribution is
        # possible: expect one analysis with the latest result attached.
        events = _two_key_trace()
        for event in events:
            if event.get("event_type") == "launch":
                del event["extracted_args"]
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 1)
        self.assertNotIn(":", analyses[0]["session_id"])
        self.assertEqual(analyses[0]["winner_compilation_hash"], "hash_b")
        self.assertIn("BLOCK_SIZE_M: 32", analyses[0]["autotune_result"]["best_config"])

    def test_split_preserves_compilation_without_traced_launches(self) -> None:
        # A candidate whose benchmark launches were never traced cannot be
        # attributed to a key round, but the split must not silently drop
        # it: every partition keeps it alongside its referenced configs.
        events = _two_key_trace()
        events.insert(2, _compilation("hash_c"))
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 2)
        for analysis in analyses:
            self.assertEqual(
                analysis["compilation_analysis"]["compilation_hashes"],
                ["hash_a", "hash_b", "hash_c"],
            )
            # hash_c has no launches, so it contributes no per-config args.
            varies = analysis["autotune_args_summary"]["autotune_configs"]["varies"]
            self.assertNotIn("hash_c", varies["BLOCK_SIZE_M"])
        by_winner = {a["winner_compilation_hash"]: a for a in analyses}
        self.assertEqual(set(by_winner), {"hash_a", "hash_b"})

    def test_note_only_launches_are_ignored_for_splitting(self) -> None:
        # Launches captured inside CUDA graphs record a _note marker
        # instead of real args. They hold no key information and must
        # neither split their own round nor pool unrelated rounds; the
        # same rounds' warmup launches outside capture still split
        # normally, and note occurrences stay out of the partitions.
        events = _two_key_trace()
        note_a = _launch("hash_a", 16, 16, "_bench", "0x1000")
        note_a["extracted_args"] = {"_note": "argument extraction skipped"}
        note_b = _launch("hash_b", 32, 32, "_bench", "0x5000")
        note_b["extracted_args"] = {"_note": "argument extraction skipped"}
        events.insert(3, note_a)
        events.insert(8, note_b)
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 2)
        by_winner = {a["winner_compilation_hash"]: a for a in analyses}
        self.assertEqual(set(by_winner), {"hash_a", "hash_b"})
        # Warmups only; note occurrences (3 and 7) are excluded.
        self.assertEqual(
            sorted(by_winner["hash_a"]["launch_occurrence_ids"]["benchmark"]),
            [2, 4],
        )
        self.assertEqual(
            sorted(by_winner["hash_b"]["launch_occurrence_ids"]["benchmark"]),
            [6, 8],
        )
        self.assertNotIn(
            "unique_count",
            json.dumps(
                by_winner["hash_a"]["autotune_args_summary"]["autotune_configs"]
            ),
        )

    def test_single_partition_keeps_note_occurrences(self) -> None:
        # One key round mixed with graph-captured launches: no
        # attribution choice exists, so note occurrences join the
        # emission (benchmark ids [2, 3, 4], winner ids [5, 6]) while
        # their _note payload stays out of the varies table.
        events = _two_key_trace()[:6]
        note_bench = _launch("hash_a", 16, 16, "_bench", "0x1000")
        note_bench["extracted_args"] = {"_note": "captured"}
        note_winner = _launch("hash_a", 16, 16, "run", "0x3000")
        note_winner["extracted_args"] = {"_note": "captured"}
        events.insert(2, note_bench)
        events.insert(6, note_winner)
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 1)
        self.assertNotIn(":", analyses[0]["session_id"])
        self.assertEqual(
            sorted(analyses[0]["launch_occurrence_ids"]["benchmark"]),
            [2, 3, 4],
        )
        self.assertEqual(sorted(analyses[0]["launch_occurrence_ids"]["winner"]), [5, 6])
        self.assertFalse(analyses[0]["cache_usage"])
        self.assertNotIn(
            "_note",
            json.dumps(analyses[0]["autotune_args_summary"]["autotune_configs"]),
        )

    def test_all_note_session_falls_back_to_coarse_occurrences(self) -> None:
        # Every launch captured inside CUDA graphs: no key derivation is
        # possible, but the emission must keep the real launches via the
        # coarse occurrence ids instead of reporting cache_usage.
        events = _two_key_trace()[:6]
        for event in events:
            if event.get("event_type") == "launch":
                event["extracted_args"] = {"_note": "captured"}
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 1)
        self.assertEqual(
            sorted(analyses[0]["launch_occurrence_ids"]["benchmark"]), [2, 3]
        )
        self.assertEqual(sorted(analyses[0]["launch_occurrence_ids"]["winner"]), [4])
        self.assertFalse(analyses[0]["cache_usage"])
        self.assertEqual(analyses[0]["winner_compilation_hash"], "hash_a")
        self.assertIn("autotune_result", analyses[0])

    def test_optional_scalar_config_stays_whole(self) -> None:
        # An optional config arg recorded as None under one compilation
        # ({"type": "NoneType", "repr": "None"}) and as an int under the
        # other ({"type": "int", "value": 16}) must still be recognized
        # as one config param: a nested path missing under a hash counts
        # as a distinct value, so the whole wrapper is masked instead of
        # splitting the round by config (each half would then fall below
        # the emission threshold and drop the entire session).
        events = _two_key_trace()[:6]
        for event in events:
            if event.get("event_type") == "launch":
                if event["compilation_metadata"]["hash"] == "hash_a":
                    event["extracted_args"]["OPTIONAL_TILE"] = {
                        "type": "NoneType",
                        "repr": "None",
                    }
                else:
                    event["extracted_args"]["OPTIONAL_TILE"] = {
                        "type": "int",
                        "value": 16,
                    }
            elif event.get("event_type") == "autotune":
                event["best_config"] += ", OPTIONAL_TILE: None"
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 1)
        self.assertNotIn(":", analyses[0]["session_id"])
        self.assertEqual(analyses[0]["winner_compilation_hash"], "hash_a")
        self.assertIn(
            "OPTIONAL_TILE: None",
            analyses[0]["autotune_result"]["best_config"],
        )

    def test_all_note_session_with_external_compilations(self) -> None:
        # Kernels compiled outside autotuning (no autotuner frames on the
        # compilation stack) leave the session with no compilation events.
        # The all-note fallback preserves the coarse launch groups, so
        # output-file lookup resolves through the launches' compilation
        # hashes and the session still emits instead of hitting the
        # neither-compilations-nor-launches guard.
        events = _two_key_trace()[:6]
        for event in events:
            if event.get("event_type") == "compilation":
                event["stack"] = [
                    {"filename": "/user/loop.py", "name": "run_kernel", "line": 1}
                ]
            elif event.get("event_type") == "launch":
                event["extracted_args"] = {"_note": "captured"}
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 1)
        self.assertEqual(
            sorted(analyses[0]["launch_occurrence_ids"]["benchmark"]), [2, 3]
        )
        self.assertEqual(sorted(analyses[0]["launch_occurrence_ids"]["winner"]), [4])
        self.assertFalse(analyses[0]["cache_usage"])
        self.assertEqual(analyses[0]["winner_compilation_hash"], "hash_a")
        self.assertIn("autotune_result", analyses[0])

    def test_mixed_descriptor_masks_only_config_paths(self) -> None:
        # A descriptor mixing key fields (shape, one per round) with
        # config fields (block_shape, one per compilation) must split on
        # the key fields only. Masking the whole argument would merge the
        # rounds; keeping it whole would over-split on block_shape (four
        # partitions); per-path masking yields exactly the two rounds.
        def _descriptor(shape, block_shape, data_ptr):
            return {
                "type": "TensorDescriptor",
                "base": {
                    "type": "tensor",
                    "shape": [16, 16],
                    "dtype": "torch.float16",
                    "stride": [16, 1],
                    "data_ptr": data_ptr,
                },
                "shape": shape,
                "strides": [16, 1],
                "block_shape": block_shape,
                "padding": "zero",
            }

        def _desc_launch(kernel_hash, shape, block_shape, boundary, data_ptr):
            launch = _launch(
                kernel_hash,
                16,
                16 if kernel_hash == "hash_a" else 32,
                boundary,
                data_ptr,
            )
            launch["extracted_args"]["desc"] = _descriptor(shape, block_shape, data_ptr)
            return launch

        events = [
            _compilation("hash_a"),
            _compilation("hash_b"),
            # Round 1: shape [16, 16], winner hash_a.
            _desc_launch("hash_a", [16, 16], [16, 16], "_bench", "0x1000"),
            _desc_launch("hash_b", [16, 16], [32, 16], "_bench", "0x2000"),
            _desc_launch("hash_a", [16, 16], [16, 16], "run", "0x3000"),
            _autotune_event(16, 16),
            # Round 2: shape [32, 32], winner hash_b.
            _desc_launch("hash_a", [32, 32], [16, 16], "_bench", "0x4000"),
            _desc_launch("hash_b", [32, 32], [32, 16], "_bench", "0x5000"),
            _desc_launch("hash_b", [32, 32], [32, 16], "run", "0x6000"),
            _autotune_event(16, 32),
        ]
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 2)
        by_winner = {a["winner_compilation_hash"]: a for a in analyses}
        self.assertEqual(set(by_winner), {"hash_a", "hash_b"})
        self.assertIn(
            "BLOCK_SIZE_M: 16", by_winner["hash_a"]["autotune_result"]["best_config"]
        )
        self.assertIn(
            "BLOCK_SIZE_M: 32", by_winner["hash_b"]["autotune_result"]["best_config"]
        )
        for analysis in analyses:
            _, _, sig = analysis["session_id"].partition(":")
            self.assertEqual(len(sig), 16)

    def test_descriptor_args_stable_across_reallocation(self) -> None:
        # TMA descriptors nest backing-tensor data_ptrs. Reallocating the
        # backing tensor must not split a round, and config-shaped fields
        # (block_shape rewritten per config) must still be recognized as
        # config params so the round stays whole.
        def _descriptor(block_shape, data_ptr):
            return {
                "type": "TensorDescriptor",
                "base": {
                    "type": "tensor",
                    "shape": [16, 16],
                    "dtype": "torch.float16",
                    "stride": [16, 1],
                    "data_ptr": data_ptr,
                },
                "shape": [16, 16],
                "strides": [16, 1],
                "block_shape": block_shape,
                "padding": "zero",
            }

        def _desc_launch(kernel_hash, block_shape, data_ptr, boundary):
            launch = _launch(kernel_hash, 16, 16, boundary, data_ptr)
            launch["extracted_args"]["desc"] = _descriptor(block_shape, data_ptr)
            return launch

        events = [
            _compilation("hash_a"),
            _compilation("hash_b"),
            _desc_launch("hash_a", [16, 16], "0x1000", "_bench"),
            _desc_launch("hash_b", [32, 16], "0x1000", "_bench"),
            # Winner rerun after the backing tensor was reallocated.
            _desc_launch("hash_a", [16, 16], "0x9999", "run"),
            _autotune_event(16, 16),
        ]
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 1)
        self.assertNotIn(":", analyses[0]["session_id"])
        self.assertEqual(analyses[0]["winner_compilation_hash"], "hash_a")
        self.assertIn("autotune_result", analyses[0])

    def test_warp_specialization_matches_on_warps_base(self) -> None:
        # Under warp specialization the launch metadata carries the
        # expanded num_warps while best_config prints the requested one;
        # matching must use the compilation payload's num_warps_base.
        events = _two_key_trace()
        for event in events:
            if event.get("event_type") == "launch":
                event["compilation_metadata"]["num_warps"] = 8
            elif event.get("event_type") == "compilation":
                event["payload"]["metadata"]["num_warps"] = 8
                event["payload"]["metadata"]["num_warps_base"] = 4
            elif event.get("event_type") == "autotune":
                event["best_config"] = event["best_config"].replace(
                    "num_warps: 1", "num_warps: 4"
                )
        _write_trace(self.input_path, events)
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 2)
        by_winner = {a["winner_compilation_hash"]: a for a in analyses}
        self.assertIn(
            "BLOCK_SIZE_M: 16", by_winner["hash_a"]["autotune_result"]["best_config"]
        )
        self.assertIn(
            "BLOCK_SIZE_M: 32", by_winner["hash_b"]["autotune_result"]["best_config"]
        )

    def test_single_key_session_keeps_plain_session_id(self) -> None:
        _write_trace(self.input_path, _two_key_trace()[:6])
        parse_single_rank([self.input_path], self.output_dir)

        analyses = [
            e
            for e in _read_output_events(self.output_dir)
            if e.get("event_type") == "autotune_analysis"
        ]
        self.assertEqual(len(analyses), 1)
        # No split happened, so the session id keeps its legacy plain form.
        self.assertNotIn(":", analyses[0]["session_id"])
        self.assertEqual(analyses[0]["winner_compilation_hash"], "hash_a")
        self.assertIn("autotune_result", analyses[0])


class BestConfigParsingTest(unittest.TestCase):
    def test_tuple_value_with_commas_survives(self) -> None:
        parsed = _parse_best_config("BLOCK: (16, 32), BLOCK_SIZE_M: 16, num_warps: 1")
        self.assertEqual(
            parsed,
            {"BLOCK": "(16, 32)", "BLOCK_SIZE_M": "16", "num_warps": "1"},
        )

    def test_non_string_config_parses_to_empty(self) -> None:
        self.assertEqual(_parse_best_config(None), {})
        self.assertEqual(_parse_best_config({"BLOCK_SIZE_M": 16}), {})

    def test_config_value_matching(self) -> None:
        # Exact ints, including beyond float precision and hex spellings.
        self.assertTrue(_config_value_matches("16", 16))
        self.assertTrue(_config_value_matches("0x10", 16))
        self.assertTrue(_config_value_matches(str(2**60 + 1), 2**60 + 1))
        self.assertFalse(_config_value_matches("17", 16))
        # None spellings.
        self.assertTrue(_config_value_matches("None", None))
        self.assertFalse(_config_value_matches("16", None))
        # Bools only accept true/false/1/0 spellings, never other numbers.
        self.assertTrue(_config_value_matches("True", True))
        self.assertTrue(_config_value_matches("1", True))
        self.assertTrue(_config_value_matches("False", False))
        self.assertTrue(_config_value_matches("0", False))
        self.assertFalse(_config_value_matches("True", 1))
        self.assertFalse(_config_value_matches("2", True))
        self.assertFalse(_config_value_matches("-1", True))
        self.assertFalse(_config_value_matches("yes", True))
        self.assertFalse(_config_value_matches("True", False))
        # Floats and plain strings fall through.
        self.assertTrue(_config_value_matches("1.5", 1.5))
        self.assertTrue(_config_value_matches("relu", "relu"))
        self.assertFalse(_config_value_matches("relu", "none"))
        # Sequences: best_config prints tuples while JSON round-trips
        # turn them into lists.
        self.assertTrue(_config_value_matches("(2, 1, 1)", [2, 1, 1]))
        self.assertTrue(_config_value_matches("(2, 1, 1)", (2, 1, 1)))
        self.assertFalse(_config_value_matches("(2, 1)", [2, 1, 1]))
        self.assertFalse(_config_value_matches("None", [1]))
        # "repr" payloads unwrap for values without structured form.
        self.assertEqual(
            _extract_scalar_value({"type": "NoneType", "repr": "None"}), "None"
        )
        self.assertTrue(_config_value_matches("None", "None"))
        self.assertTrue(_config_value_matches("(16, 32)", "(16, 32)"))
