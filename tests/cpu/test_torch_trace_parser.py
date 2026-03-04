# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for torch trace log parser."""

import json
import os
import tempfile
import unittest
from typing import List, Optional

from tritonparse.parse.torch_trace_parser import (
    _extract_json_from_glog_line,
    _parse_torch_trace_log,
    CompileInfo,
    discover_torch_trace_files,
    parse_torch_trace_logs,
)


def _make_glog_line(metadata_dict: dict) -> str:
    """Helper to create a glog-formatted line with JSON metadata."""
    return f"V0302 14:30:00.123456 12345 torch/_logging/_internal.py:1489] {json.dumps(metadata_dict)}"


def _make_output_code_event(
    frame_id: int = 0,
    frame_compile_id: int = 0,
    attempt: int = 0,
    compiled_autograd_id: Optional[int] = None,
    kernel_paths: Optional[List[str]] = None,
) -> str:
    """Helper to create a complete inductor_output_code log record (header + payload)."""
    metadata = {
        "inductor_output_code": {
            "filename": "/tmp/torchinductor_user/ab/test.py",
            "file_path": "/tmp/torchinductor_user/ab/test.py",
        },
        "frame_id": frame_id,
        "frame_compile_id": frame_compile_id,
        "attempt": attempt,
        "has_payload": "abc123",
    }
    if compiled_autograd_id is not None:
        metadata["compiled_autograd_id"] = compiled_autograd_id

    lines = [_make_glog_line(metadata)]

    # Build a minimal output_code.py payload with kernel path comments
    if kernel_paths is None:
        kernel_paths = ["/tmp/torchinductor_user/ab/cabcdef1234.py"]

    for kp in kernel_paths:
        lines.append(f"\t# kernel path: {kp}")
        lines.append("\t# Source Nodes: [add], Original ATen: [aten.add]")
        lines.append(
            "\ttriton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''"
        )
        lines.append("\t# kernel code here")
        lines.append("\t''')")

    return "\n".join(lines)


class TestExtractJsonFromGlogLine(unittest.TestCase):
    """Tests for _extract_json_from_glog_line."""

    def test_valid_glog_line(self):
        line = 'V0302 14:30:00.123456 12345 path.py:100] {"key": "value"}'
        result = _extract_json_from_glog_line(line)
        self.assertEqual(result, '{"key": "value"}')

    def test_no_bracket(self):
        line = "some random text without bracket"
        result = _extract_json_from_glog_line(line)
        self.assertIsNone(result)

    def test_bracket_without_space(self):
        line = "]no_space"
        result = _extract_json_from_glog_line(line)
        self.assertIsNone(result)


class TestCompileInfo(unittest.TestCase):
    """Tests for CompileInfo dataclass."""

    def test_defaults(self):
        info = CompileInfo()
        self.assertIsNone(info.frame_id)
        self.assertIsNone(info.frame_compile_id)
        self.assertEqual(info.attempt, 0)
        self.assertIsNone(info.compiled_autograd_id)

    def test_with_values(self):
        info = CompileInfo(
            frame_id=1, frame_compile_id=2, attempt=3, compiled_autograd_id=4
        )
        self.assertEqual(info.frame_id, 1)
        self.assertEqual(info.frame_compile_id, 2)
        self.assertEqual(info.attempt, 3)
        self.assertEqual(info.compiled_autograd_id, 4)


class TestParseTorchTraceLog(unittest.TestCase):
    """Tests for _parse_torch_trace_log."""

    def _write_log(self, content: str) -> str:
        """Write content to a temp file and return path."""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False, prefix="test_torch_trace_"
        )
        f.write(content)
        f.close()
        return f.name

    def test_single_kernel(self):
        """Test parsing a log with a single inductor_output_code event."""
        content = _make_output_code_event(
            frame_id=0,
            frame_compile_id=0,
            kernel_paths=["/tmp/torchinductor_user/ab/cabcdef1234.py"],
        )
        log_path = self._write_log(content)
        try:
            mapping = _parse_torch_trace_log(log_path)
            self.assertEqual(len(mapping), 1)
            self.assertIn("/tmp/torchinductor_user/ab/cabcdef1234.py", mapping)
            info = mapping["/tmp/torchinductor_user/ab/cabcdef1234.py"]
            self.assertEqual(info.frame_id, 0)
            self.assertEqual(info.frame_compile_id, 0)
            self.assertEqual(info.attempt, 0)
            self.assertIsNone(info.compiled_autograd_id)
        finally:
            os.unlink(log_path)

    def test_multiple_kernels_in_one_event(self):
        """Test parsing an output_code event with multiple kernel paths."""
        content = _make_output_code_event(
            frame_id=7,
            frame_compile_id=3,
            kernel_paths=[
                "/tmp/torchinductor_user/ab/kernel1.py",
                "/tmp/torchinductor_user/cd/kernel2.py",
                "/tmp/torchinductor_user/ef/kernel3.py",
            ],
        )
        log_path = self._write_log(content)
        try:
            mapping = _parse_torch_trace_log(log_path)
            self.assertEqual(len(mapping), 3)
            for kp in [
                "/tmp/torchinductor_user/ab/kernel1.py",
                "/tmp/torchinductor_user/cd/kernel2.py",
                "/tmp/torchinductor_user/ef/kernel3.py",
            ]:
                self.assertIn(kp, mapping)
                self.assertEqual(mapping[kp].frame_id, 7)
                self.assertEqual(mapping[kp].frame_compile_id, 3)
        finally:
            os.unlink(log_path)

    def test_multiple_events_different_frames(self):
        """Test parsing multiple output_code events from different compilation frames."""
        event1 = _make_output_code_event(
            frame_id=0,
            frame_compile_id=0,
            kernel_paths=["/tmp/torchinductor_user/ab/kernel_a.py"],
        )
        event2 = _make_output_code_event(
            frame_id=1,
            frame_compile_id=0,
            kernel_paths=["/tmp/torchinductor_user/cd/kernel_b.py"],
        )
        # Non-output_code event in between
        other_event = _make_glog_line(
            {"dynamo_start": {"stack_index": 0}, "frame_id": 0}
        )

        content = event1 + "\n" + other_event + "\n" + event2
        log_path = self._write_log(content)
        try:
            mapping = _parse_torch_trace_log(log_path)
            self.assertEqual(len(mapping), 2)
            self.assertEqual(
                mapping["/tmp/torchinductor_user/ab/kernel_a.py"].frame_id, 0
            )
            self.assertEqual(
                mapping["/tmp/torchinductor_user/cd/kernel_b.py"].frame_id, 1
            )
        finally:
            os.unlink(log_path)

    def test_with_compiled_autograd_id(self):
        """Test parsing events with compiled_autograd_id."""
        content = _make_output_code_event(
            frame_id=2,
            frame_compile_id=1,
            compiled_autograd_id=5,
            kernel_paths=["/tmp/torchinductor_user/ab/kernel.py"],
        )
        log_path = self._write_log(content)
        try:
            mapping = _parse_torch_trace_log(log_path)
            info = mapping["/tmp/torchinductor_user/ab/kernel.py"]
            self.assertEqual(info.compiled_autograd_id, 5)
        finally:
            os.unlink(log_path)

    def test_no_output_code_events(self):
        """Test parsing a log with no inductor_output_code events."""
        content = _make_glog_line({"dynamo_start": {"stack_index": 0}})
        log_path = self._write_log(content)
        try:
            mapping = _parse_torch_trace_log(log_path)
            self.assertEqual(len(mapping), 0)
        finally:
            os.unlink(log_path)

    def test_empty_file(self):
        """Test parsing an empty log file."""
        log_path = self._write_log("")
        try:
            mapping = _parse_torch_trace_log(log_path)
            self.assertEqual(len(mapping), 0)
        finally:
            os.unlink(log_path)

    def test_no_kernel_paths_in_payload(self):
        """Test an output_code event with no kernel path comments in payload."""
        metadata = {
            "inductor_output_code": {"filename": "test.py", "file_path": "test.py"},
            "frame_id": 0,
            "frame_compile_id": 0,
        }
        content = _make_glog_line(metadata) + "\n\t# some other comment\n\tcode = 42\n"
        log_path = self._write_log(content)
        try:
            mapping = _parse_torch_trace_log(log_path)
            self.assertEqual(len(mapping), 0)
        finally:
            os.unlink(log_path)

    def test_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        mapping = _parse_torch_trace_log("/nonexistent/path/to/log.log")
        self.assertEqual(len(mapping), 0)

    def test_malformed_json(self):
        """Test that malformed JSON lines are skipped gracefully."""
        good_event = _make_output_code_event(
            frame_id=0,
            frame_compile_id=0,
            kernel_paths=["/tmp/torchinductor_user/ab/kernel.py"],
        )
        bad_line = "V0302 14:30:00.000000 999 path.py:1] {invalid json here"
        content = bad_line + "\n" + good_event
        log_path = self._write_log(content)
        try:
            mapping = _parse_torch_trace_log(log_path)
            # Should still parse the good event
            self.assertEqual(len(mapping), 1)
            self.assertIn("/tmp/torchinductor_user/ab/kernel.py", mapping)
        finally:
            os.unlink(log_path)


class TestParseTorchTraceLogs(unittest.TestCase):
    """Tests for parse_torch_trace_logs (multi-file)."""

    def test_merge_multiple_files(self):
        """Test that mappings from multiple files are merged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File 1: rank 0
            content1 = _make_output_code_event(
                frame_id=0,
                frame_compile_id=0,
                kernel_paths=["/tmp/torchinductor_user/ab/kernel_r0.py"],
            )
            path1 = os.path.join(tmpdir, "log1.log")
            with open(path1, "w") as f:
                f.write(content1)

            # File 2: rank 1
            content2 = _make_output_code_event(
                frame_id=0,
                frame_compile_id=0,
                kernel_paths=["/tmp/torchinductor_user/cd/kernel_r1.py"],
            )
            path2 = os.path.join(tmpdir, "log2.log")
            with open(path2, "w") as f:
                f.write(content2)

            mapping = parse_torch_trace_logs([path1, path2])
            self.assertEqual(len(mapping), 2)
            self.assertIn("/tmp/torchinductor_user/ab/kernel_r0.py", mapping)
            self.assertIn("/tmp/torchinductor_user/cd/kernel_r1.py", mapping)


class TestDiscoverTorchTraceFiles(unittest.TestCase):
    """Tests for discover_torch_trace_files."""

    def test_discover_ranked_files(self):
        """Test discovery of rank-specific torch trace files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock torch trace files
            for rank in [0, 1, 2]:
                path = os.path.join(
                    tmpdir, f"dedicated_log_torch_trace_rank_{rank}_abc123.log"
                )
                with open(path, "w") as f:
                    f.write("")

            result = discover_torch_trace_files(tmpdir)
            self.assertEqual(len(result), 3)
            self.assertIn(0, result)
            self.assertIn(1, result)
            self.assertIn(2, result)

    def test_discover_no_rank_file(self):
        """Test discovery of a torch trace file without rank suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dedicated_log_torch_trace_abc123.log")
            with open(path, "w") as f:
                f.write("")

            result = discover_torch_trace_files(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertIn(None, result)

    def test_ignores_non_torch_trace_files(self):
        """Test that non-torch-trace files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Tritonparse file (should be ignored)
            with open(
                os.path.join(tmpdir, "dedicated_log_triton_trace_user_.ndjson"), "w"
            ) as f:
                f.write("")
            # Random file (should be ignored)
            with open(os.path.join(tmpdir, "other_file.log"), "w") as f:
                f.write("")
            # Torch trace file (should be found)
            with open(
                os.path.join(tmpdir, "dedicated_log_torch_trace_rank_0_abc.log"), "w"
            ) as f:
                f.write("")

            result = discover_torch_trace_files(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertIn(0, result)

    def test_empty_directory(self):
        """Test discovery in an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_torch_trace_files(tmpdir)
            self.assertEqual(len(result), 0)

    def test_nonexistent_directory(self):
        """Test discovery in a directory that doesn't exist."""
        result = discover_torch_trace_files("/nonexistent/directory")
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
