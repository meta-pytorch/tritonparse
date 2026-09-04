# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import tempfile
import unittest

from tritonparse.parse.trace_processor import parse_single_file, parse_single_rank


def _autotune_stack(session: str, boundary: str = "_bench") -> list[dict]:
    return [
        {
            "filename": f"/user/{session}.py",
            "name": "run_kernel",
            "line": 10,
        },
        {
            "filename": "triton/runtime/autotuner.py",
            "name": boundary,
            "line": 100,
        },
    ]


def _compilation(
    kernel_hash: str,
    kernel_name: object = None,
    *,
    frame_id: int = 0,
    stack: list[dict] | None = None,
    pid: int = 1000,
) -> dict:
    metadata = {
        "hash": kernel_hash,
        "num_warps": 4,
        "num_stages": 2,
        "num_ctas": 1,
    }
    if kernel_name is not None:
        metadata["name"] = kernel_name
    return {
        "event_type": "compilation",
        "pid": pid,
        "timestamp": "2026-09-02T00:00:00",
        "stack": stack or [],
        "payload": {
            "metadata": metadata,
            "pt_info": {"frame_id": frame_id, "frame_compile_id": 0},
            "file_content": {},
            "file_path": {},
        },
    }


def _launch(
    kernel_hash: str | None,
    kernel_name: object = None,
    *,
    metadata_name: object = None,
    stack: list[dict] | None = None,
    pid: int = 1000,
) -> dict:
    compilation_metadata = {
        "num_warps": 4,
        "num_stages": 2,
        "num_ctas": 1,
    }
    if kernel_hash is not None:
        compilation_metadata["hash"] = kernel_hash
    if metadata_name is not None:
        compilation_metadata["name"] = metadata_name
    event = {
        "event_type": "launch",
        "pid": pid,
        "timestamp": "2026-09-02T00:00:01",
        "stack": stack or [],
        "compilation_metadata": compilation_metadata,
    }
    if kernel_name is not None:
        event["name"] = kernel_name
    return event


def _autotune_result(kernel_name: str, session: str) -> dict:
    return {
        "event_type": "autotune",
        "kernel_name": kernel_name,
        "stack": _autotune_stack(session, boundary="run"),
        "best_config": "num_warps=4",
        "configs_timings": {"num_warps=4": 1.0},
        "duration": 1.0,
        "cache_hit": False,
        "cache_key": "cache-key",
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


class ParseKernelFilterTest(unittest.TestCase):
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

    def test_filters_complete_hash_group_from_shared_frame(self) -> None:
        _write_trace(
            self.input_path,
            [
                _compilation("target_hash", "target_kernel"),
                _launch("target_hash", "target_kernel"),
                _compilation("other_hash", "other_kernel"),
                _launch("other_hash", "other_kernel"),
            ],
        )

        stats = parse_single_rank(
            [self.input_path],
            self.output_dir,
            kernel_patterns=("target*",),
        )

        self.assertIsNotNone(stats)
        self.assertEqual(stats.direct_matched_hash_count, 1)
        self.assertEqual(stats.selected_hash_count, 1)
        self.assertEqual(stats.emitted_kernel_group_count, 1)
        events = _read_output_events(self.output_dir)
        compilations = [e for e in events if e.get("event_type") == "compilation"]
        launches = [e for e in events if e.get("event_type") == "launch"]
        self.assertEqual(
            [e["payload"]["metadata"]["hash"] for e in compilations],
            ["target_hash"],
        )
        self.assertEqual(
            [e["compilation_metadata"]["hash"] for e in launches],
            ["target_hash"],
        )
        self.assertNotIn(
            "other_hash",
            {e.get("hash") for e in events if e.get("hash") is not None},
        )

    def test_later_launch_alias_selects_launch_only_kernel(self) -> None:
        first_stack = [{"filename": "/first.py", "name": "first", "line": 1}]
        _write_trace(
            self.input_path,
            [
                _launch("launch_only_hash", stack=first_stack, pid=1001),
                _launch(
                    "launch_only_hash",
                    "runtime_name",
                    metadata_name="target_alias",
                    stack=[{"filename": "/second.py", "name": "second", "line": 2}],
                    pid=1002,
                ),
            ],
        )

        stats = parse_single_rank(
            [self.input_path],
            self.output_dir,
            kernel_patterns=("target_*",),
        )

        self.assertIsNotNone(stats)
        self.assertEqual(stats.selected_hash_count, 1)
        events = _read_output_events(self.output_dir)
        compilations = [e for e in events if e.get("event_type") == "compilation"]
        launches = [e for e in events if e.get("event_type") == "launch"]
        self.assertEqual(len(compilations), 1)
        self.assertTrue(compilations[0]["is_fake"])
        self.assertIsNone(compilations[0]["payload"]["metadata"]["name"])
        self.assertEqual(compilations[0]["pid"], 1001)
        self.assertEqual(compilations[0]["stack"], first_stack)
        self.assertEqual(len(launches), 2)

    def test_autotune_match_keeps_complete_session_without_leaks(self) -> None:
        selected_stack = _autotune_stack("selected")
        winner_stack = _autotune_stack("selected", boundary="run")
        unrelated_stack = _autotune_stack("unrelated")
        unrelated_winner_stack = _autotune_stack("unrelated", boundary="run")
        _write_trace(
            self.input_path,
            [
                _compilation("hash_a", "generated_config_a", stack=selected_stack),
                _compilation("hash_b", "generated_config_b", stack=selected_stack),
                _launch("hash_a", "generated_config_a", stack=selected_stack),
                _launch("hash_b", "generated_config_b", stack=selected_stack),
                _launch("hash_a", "generated_config_a", stack=winner_stack),
                _autotune_result("target_kernel", "selected"),
                _compilation(
                    "hash_c",
                    "unrelated_kernel",
                    frame_id=1,
                    stack=unrelated_stack,
                ),
                _compilation(
                    "hash_d",
                    "another_unrelated_config",
                    frame_id=1,
                    stack=unrelated_stack,
                ),
                _launch("hash_c", "unrelated_kernel", stack=unrelated_stack),
                _launch("hash_d", "another_unrelated_config", stack=unrelated_stack),
                _launch("hash_c", "unrelated_kernel", stack=unrelated_winner_stack),
                _autotune_result("unrelated_kernel", "unrelated"),
            ],
        )

        stats = parse_single_rank(
            [self.input_path],
            self.output_dir,
            kernel_patterns=("target*",),
        )

        self.assertIsNotNone(stats)
        self.assertEqual(stats.direct_matched_hash_count, 0)
        self.assertEqual(stats.direct_matched_session_count, 1)
        self.assertEqual(stats.selected_hash_count, 2)
        events = _read_output_events(self.output_dir)
        compilation_hashes = {
            e["payload"]["metadata"]["hash"]
            for e in events
            if e.get("event_type") == "compilation"
        }
        self.assertEqual(compilation_hashes, {"hash_a", "hash_b"})

        analyses = [e for e in events if e.get("event_type") == "autotune_analysis"]
        self.assertEqual(len(analyses), 1)
        self.assertEqual(analyses[0]["possible_groups"], [["hash_a", "hash_b"]])
        self.assertIn("autotune_result", analyses[0])

        summaries = [e for e in events if e.get("event_type") == "autotune_summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["winner_run_counts"], {"hash_a": 1})

    def test_later_pid_alias_selects_complete_cross_pid_hash_group(self) -> None:
        second_input_path = os.path.join(
            self.temporary_directory.name,
            "dedicated_log_triton_trace_user_pid_2000_.ndjson",
        )
        _write_trace(
            self.input_path,
            [
                _compilation("shared_hash", "generated_name", pid=1000),
                _launch("shared_hash", "generated_name", pid=1000),
            ],
        )
        _write_trace(
            second_input_path,
            [_launch("shared_hash", "target_alias", pid=2000)],
        )

        stats = parse_single_rank(
            [self.input_path, second_input_path],
            self.output_dir,
            kernel_patterns=("target_*",),
        )

        self.assertIsNotNone(stats)
        self.assertEqual(stats.selected_hash_count, 1)
        events = _read_output_events(self.output_dir)
        self.assertEqual(
            [event["pid"] for event in events if event.get("event_type") == "launch"],
            [1000, 2000],
        )

    def test_selection_closure_crosses_multiple_sessions(self) -> None:
        session_one = _autotune_stack("one")
        session_two = _autotune_stack("two")
        _write_trace(
            self.input_path,
            [
                _compilation("hash_a", "target_kernel", stack=session_one),
                _compilation("hash_b", "middle_config", stack=session_one),
                _compilation("hash_b", "middle_config", stack=session_two),
                _compilation("hash_c", "last_config", stack=session_two),
            ],
        )

        stats = parse_single_rank(
            [self.input_path],
            self.output_dir,
            kernel_patterns=("target*",),
        )

        self.assertIsNotNone(stats)
        self.assertEqual(stats.selected_hash_count, 3)
        self.assertEqual(stats.selected_session_count, 2)
        events = _read_output_events(self.output_dir)
        self.assertEqual(
            {
                e["payload"]["metadata"]["hash"]
                for e in events
                if e.get("event_type") == "compilation"
            },
            {"hash_a", "hash_b", "hash_c"},
        )

    def test_single_file_reports_no_match_with_available_names(self) -> None:
        _write_trace(
            self.input_path,
            [_compilation("other_hash", "other_kernel")],
        )

        with self.assertRaises(RuntimeError) as context:
            parse_single_file(
                self.input_path,
                self.output_dir,
                kernel_patterns=("target*",),
            )

        message = str(context.exception)
        self.assertIn("parse-time kernel allowlist 'target*'", message)
        self.assertIn("other_kernel", message)
        self.assertEqual(os.listdir(self.output_dir), [])

    def test_no_filter_preserves_existing_return_contract(self) -> None:
        _write_trace(
            self.input_path,
            [_compilation("hash_a", "kernel_a")],
        )

        result = parse_single_rank([self.input_path], self.output_dir)

        self.assertIsNone(result)
        self.assertEqual(
            {
                e["payload"]["metadata"]["hash"]
                for e in _read_output_events(self.output_dir)
                if e.get("event_type") == "compilation"
            },
            {"hash_a"},
        )

    def test_filtered_empty_rank_returns_empty_stats(self) -> None:
        stats = parse_single_rank(
            [],
            self.output_dir,
            kernel_patterns=("target*",),
        )

        self.assertIsNotNone(stats)
        self.assertEqual(stats.selected_hash_count, 0)
        self.assertEqual(stats.emitted_kernel_group_count, 0)
