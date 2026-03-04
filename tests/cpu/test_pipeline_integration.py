# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Integration tests for the full parse pipeline with torch trace log support."""

import json
import os
import tempfile
import unittest

from tritonparse.parse.common import (
    _build_kernel_compile_mapping,
    parse_logs,
    RankConfig,
)


def _make_glog_line(metadata_dict: dict) -> str:
    """Helper to create a glog-formatted line with JSON metadata."""
    return f"V0302 14:30:00.123456 12345 torch/_logging/_internal.py:1489] {json.dumps(metadata_dict)}"


def _make_torch_trace_log(frame_id, frame_compile_id, kernel_paths):
    """Create content for a torch trace log file with inductor_output_code event."""
    metadata = {
        "inductor_output_code": {
            "filename": "output.py",
            "file_path": "/tmp/output.py",
        },
        "frame_id": frame_id,
        "frame_compile_id": frame_compile_id,
        "attempt": 0,
        "has_payload": "abc123",
    }
    lines = [_make_glog_line(metadata)]
    for kp in kernel_paths:
        lines.append(f"\t# kernel path: {kp}")
        lines.append("\ttriton_kernel = async_compile.triton('kernel', '''...''')")
    return "\n".join(lines) + "\n"


def _make_tritonparse_trace(events):
    """Create content for a tritonparse trace NDJSON file."""
    return "\n".join(json.dumps(e) for e in events) + "\n"


class TestBuildKernelCompileMapping(unittest.TestCase):
    """Tests for _build_kernel_compile_mapping."""

    def test_auto_discover_in_raw_log_dir(self):
        """Test that torch trace files are auto-discovered in the raw log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a torch trace log file in the same directory
            torch_log = os.path.join(tmpdir, "dedicated_log_torch_trace_rank_0_abc.log")
            content = _make_torch_trace_log(
                frame_id=0,
                frame_compile_id=0,
                kernel_paths=["/tmp/torchinductor_user/ab/kernel.py"],
            )
            with open(torch_log, "w") as f:
                f.write(content)

            mapping = _build_kernel_compile_mapping(tmpdir)
            self.assertIsNotNone(mapping)
            self.assertIn("/tmp/torchinductor_user/ab/kernel.py", mapping)
            self.assertEqual(
                mapping["/tmp/torchinductor_user/ab/kernel.py"].frame_id, 0
            )

    def test_explicit_torch_trace_dir(self):
        """Test using an explicit torch_trace_dir."""
        with tempfile.TemporaryDirectory() as log_dir:
            with tempfile.TemporaryDirectory() as torch_dir:
                torch_log = os.path.join(
                    torch_dir, "dedicated_log_torch_trace_rank_0_abc.log"
                )
                content = _make_torch_trace_log(
                    frame_id=1,
                    frame_compile_id=0,
                    kernel_paths=["/tmp/torchinductor_user/cd/kernel.py"],
                )
                with open(torch_log, "w") as f:
                    f.write(content)

                mapping = _build_kernel_compile_mapping(log_dir, torch_dir)
                self.assertIsNotNone(mapping)
                self.assertEqual(len(mapping), 1)

    def test_no_torch_trace_files(self):
        """Test that None is returned when no torch trace files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping = _build_kernel_compile_mapping(tmpdir)
            self.assertIsNone(mapping)


class TestParseLogsWithTorchTrace(unittest.TestCase):
    """End-to-end test for parse_logs with torch trace integration."""

    def test_end_to_end_mapping(self):
        """Test that kernels without pt_info are correctly attributed via torch trace logs."""
        kernel_path = "/tmp/torchinductor_user/ab/cabcdef.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create torch trace log
            torch_log_path = os.path.join(
                tmpdir, "dedicated_log_torch_trace_rank_0_test.log"
            )
            torch_content = _make_torch_trace_log(
                frame_id=3,
                frame_compile_id=1,
                kernel_paths=[kernel_path],
            )
            with open(torch_log_path, "w") as f:
                f.write(torch_content)

            # Create tritonparse trace log (compilation without pt_info)
            triton_events = [
                {
                    "event_type": "compilation",
                    "pid": 1000,
                    "stack": [],
                    "payload": {
                        "metadata": {"hash": "test_hash", "name": "test_kernel"},
                        "file_content": {},
                        "file_path": {},
                        "python_source": {"file_path": kernel_path},
                        # No pt_info — multi-process scenario
                    },
                },
                {
                    "event_type": "launch",
                    "name": "test_kernel",
                    "pid": 1000,
                    "stack": [],
                    "compilation_metadata": {"hash": "test_hash"},
                },
            ]
            triton_log_path = os.path.join(
                tmpdir, "dedicated_log_triton_trace_user_.ndjson"
            )
            with open(triton_log_path, "w") as f:
                f.write(_make_tritonparse_trace(triton_events))

            # Run parse_logs (use all_ranks=True to pick up no-rank files)
            rank_config = RankConfig(all_ranks=True)
            parsed_dir, file_mapping = parse_logs(
                tmpdir,
                rank_config,
                verbose=False,
                split_inductor_compilations=True,
            )

            # Check that the output was split into a frame-specific file
            # Walk the output directory to find all generated files
            all_files = []
            for root, dirs, files in os.walk(parsed_dir):
                for f in files:
                    all_files.append(f)

            frame_files = [f for f in all_files if f.startswith("f")]
            # Should have f3_fc1_a0_cai-.ndjson.gz (attributed via mapping)
            self.assertTrue(
                any("f3_fc1" in f for f in frame_files),
                f"Expected frame file with f3_fc1 but got: {all_files}",
            )


if __name__ == "__main__":
    unittest.main()
