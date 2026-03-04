# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for kernel attribution via compile mapping in trace_processor."""

import json
import os
import tempfile
import unittest

from tritonparse.parse.torch_trace_parser import CompileInfo
from tritonparse.parse.trace_processor import (
    _determine_output_fname,
    _resolve_compile_info,
    parse_single_file,
)


class TestResolveCompileInfo(unittest.TestCase):
    """Tests for _resolve_compile_info."""

    def _make_mapping(self):
        return {
            "/tmp/torchinductor_user/ab/kernel1.py": CompileInfo(
                frame_id=0, frame_compile_id=0
            ),
            "/tmp/torchinductor_user/cd/kernel2.py": CompileInfo(
                frame_id=1, frame_compile_id=0, attempt=1
            ),
        }

    def test_resolve_via_python_source(self):
        """Test resolution via python_source.file_path."""
        event = {
            "payload": {
                "python_source": {"file_path": "/tmp/torchinductor_user/ab/kernel1.py"}
            },
            "stack": [],
        }
        result = _resolve_compile_info(event, self._make_mapping())
        self.assertIsNotNone(result)
        self.assertEqual(result.frame_id, 0)
        self.assertEqual(result.frame_compile_id, 0)

    def test_resolve_via_stack_trace(self):
        """Test resolution via stack trace when python_source is missing."""
        event = {
            "payload": {},
            "stack": [
                {"filename": "/user/code.py", "line": 10, "name": "main"},
                {
                    "filename": "/tmp/torchinductor_user/cd/kernel2.py",
                    "line": 1,
                    "name": "kernel",
                },
                {"filename": "triton/jit.py", "line": 50, "name": "run"},
            ],
        }
        result = _resolve_compile_info(event, self._make_mapping())
        self.assertIsNotNone(result)
        self.assertEqual(result.frame_id, 1)
        self.assertEqual(result.frame_compile_id, 0)
        self.assertEqual(result.attempt, 1)

    def test_no_match(self):
        """Test that None is returned when no match is found."""
        event = {
            "payload": {"python_source": {"file_path": "/tmp/unknown/path.py"}},
            "stack": [{"filename": "/user/code.py", "line": 10, "name": "main"}],
        }
        result = _resolve_compile_info(event, self._make_mapping())
        self.assertIsNone(result)

    def test_empty_event(self):
        """Test with minimal event data."""
        event = {}
        result = _resolve_compile_info(event, self._make_mapping())
        self.assertIsNone(result)

    def test_python_source_takes_priority(self):
        """Test that python_source.file_path is preferred over stack trace."""
        event = {
            "payload": {
                "python_source": {"file_path": "/tmp/torchinductor_user/ab/kernel1.py"}
            },
            "stack": [
                {
                    "filename": "/tmp/torchinductor_user/cd/kernel2.py",
                    "line": 1,
                    "name": "kernel",
                }
            ],
        }
        result = _resolve_compile_info(event, self._make_mapping())
        # Should use python_source path (kernel1 -> frame_id=0), not stack (kernel2 -> frame_id=1)
        self.assertEqual(result.frame_id, 0)


class TestDetermineOutputFname(unittest.TestCase):
    """Tests for _determine_output_fname."""

    def test_with_pt_info(self):
        """Test normal case where pt_info has frame_id/compile_id."""
        fname = _determine_output_fname(
            pt_info={"frame_id": 0, "frame_compile_id": 1, "attempt_id": 0},
            file_name_without_extension="trace",
            split_inductor_compilations=True,
        )
        self.assertEqual(fname, "f0_fc1_a0_cai-.ndjson")

    def test_without_pt_info_no_mapping(self):
        """Test fallback to mapped file when pt_info is missing and no mapping."""
        fname = _determine_output_fname(
            pt_info={},
            file_name_without_extension="trace",
            split_inductor_compilations=True,
        )
        self.assertEqual(fname, "trace_mapped.ndjson")

    def test_without_pt_info_with_mapping(self):
        """Test resolution via mapping when pt_info is missing."""
        mapping = {
            "/tmp/torchinductor_user/ab/kernel.py": CompileInfo(
                frame_id=3, frame_compile_id=2, attempt=1, compiled_autograd_id=5
            )
        }
        event = {
            "payload": {
                "python_source": {"file_path": "/tmp/torchinductor_user/ab/kernel.py"}
            },
            "stack": [],
        }
        fname = _determine_output_fname(
            pt_info={},
            file_name_without_extension="trace",
            split_inductor_compilations=True,
            event=event,
            kernel_compile_mapping=mapping,
        )
        self.assertEqual(fname, "f3_fc2_a1_cai5.ndjson")

    def test_split_disabled(self):
        """Test that splitting disabled always returns mapped filename."""
        fname = _determine_output_fname(
            pt_info={"frame_id": 0, "frame_compile_id": 0},
            file_name_without_extension="trace",
            split_inductor_compilations=False,
        )
        self.assertEqual(fname, "trace_mapped.ndjson")

    def test_compiled_autograd_id_none(self):
        """Test that compiled_autograd_id defaults to '-' when not set."""
        fname = _determine_output_fname(
            pt_info={"frame_id": 0, "frame_compile_id": 0},
            file_name_without_extension="trace",
            split_inductor_compilations=True,
        )
        self.assertEqual(fname, "f0_fc0_a0_cai-.ndjson")


class TestParseSingleFileWithMapping(unittest.TestCase):
    """Integration tests for parse_single_file with kernel_compile_mapping."""

    def test_mapping_redirects_compilation_to_frame_file(self):
        """Test that a compilation without pt_info is redirected when mapping is provided."""
        kernel_path = "/tmp/torchinductor_user/ab/cabcdef1234.py"
        mapping = {
            kernel_path: CompileInfo(frame_id=2, frame_compile_id=1),
        }

        trace_lines = [
            json.dumps(
                {
                    "event_type": "compilation",
                    "pid": 1000,
                    "stack": [],
                    "payload": {
                        "metadata": {"hash": "kernel_hash_1", "name": "kernel_1"},
                        "file_content": {},
                        "file_path": {},
                        "python_source": {"file_path": kernel_path},
                        # No pt_info — this is the multi-process scenario
                    },
                }
            ),
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "kernel_1",
                    "pid": 1000,
                    "stack": [],
                    "compilation_metadata": {"hash": "kernel_hash_1"},
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "test_trace.ndjson")
            with open(input_file, "w") as f:
                for line in trace_lines:
                    f.write(line + "\n")

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)

            parse_single_file(input_file, output_dir, kernel_compile_mapping=mapping)

            output_files = os.listdir(output_dir)
            # Should produce a frame-specific file, not _mapped
            frame_files = [f for f in output_files if f.startswith("f")]
            mapped_files = [f for f in output_files if "mapped" in f]
            self.assertEqual(len(frame_files), 1)
            self.assertEqual(len(mapped_files), 0)
            self.assertEqual(frame_files[0], "f2_fc1_a0_cai-.ndjson")

    def test_no_mapping_falls_back_to_mapped(self):
        """Test that without mapping, compilations without pt_info go to _mapped."""
        trace_lines = [
            json.dumps(
                {
                    "event_type": "compilation",
                    "pid": 1000,
                    "stack": [],
                    "payload": {
                        "metadata": {"hash": "kernel_hash_1", "name": "kernel_1"},
                        "file_content": {},
                        "file_path": {},
                        # No pt_info, no python_source
                    },
                }
            ),
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "kernel_1",
                    "pid": 1000,
                    "stack": [],
                    "compilation_metadata": {"hash": "kernel_hash_1"},
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "test_trace.ndjson")
            with open(input_file, "w") as f:
                for line in trace_lines:
                    f.write(line + "\n")

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)

            parse_single_file(input_file, output_dir)

            output_files = os.listdir(output_dir)
            mapped_files = [f for f in output_files if "mapped" in f]
            self.assertGreater(len(mapped_files), 0)

    def test_mixed_with_and_without_pt_info(self):
        """Test a mix of events: some with pt_info, some resolved via mapping."""
        kernel_path_a = "/tmp/torchinductor_user/ab/kernel_a.py"
        mapping = {
            kernel_path_a: CompileInfo(frame_id=0, frame_compile_id=0),
        }

        trace_lines = [
            # Compilation WITH pt_info (should be split normally)
            json.dumps(
                {
                    "event_type": "compilation",
                    "pid": 1000,
                    "stack": [],
                    "payload": {
                        "metadata": {"hash": "hash_with_pt", "name": "kernel_with_pt"},
                        "file_content": {},
                        "file_path": {},
                        "pt_info": {
                            "frame_id": 1,
                            "frame_compile_id": 0,
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "kernel_with_pt",
                    "pid": 1000,
                    "stack": [],
                    "compilation_metadata": {"hash": "hash_with_pt"},
                }
            ),
            # Compilation WITHOUT pt_info (should be resolved via mapping)
            json.dumps(
                {
                    "event_type": "compilation",
                    "pid": 1000,
                    "stack": [],
                    "payload": {
                        "metadata": {
                            "hash": "hash_without_pt",
                            "name": "kernel_without_pt",
                        },
                        "file_content": {},
                        "file_path": {},
                        "python_source": {"file_path": kernel_path_a},
                    },
                }
            ),
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "kernel_without_pt",
                    "pid": 1000,
                    "stack": [],
                    "compilation_metadata": {"hash": "hash_without_pt"},
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "test_trace.ndjson")
            with open(input_file, "w") as f:
                for line in trace_lines:
                    f.write(line + "\n")

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)

            parse_single_file(input_file, output_dir, kernel_compile_mapping=mapping)

            output_files = sorted(os.listdir(output_dir))
            # Should have two frame files: f0_fc0 and f1_fc0
            frame_files = sorted([f for f in output_files if f.startswith("f")])
            self.assertEqual(len(frame_files), 2)
            self.assertIn("f0_fc0_a0_cai-.ndjson", frame_files)
            self.assertIn("f1_fc0_a0_cai-.ndjson", frame_files)

    def test_fake_compilation_with_mapping(self):
        """Test that fake compilations can also be attributed via stack trace mapping."""
        kernel_path = "/tmp/torchinductor_user/ab/kernel_fake.py"
        mapping = {
            kernel_path: CompileInfo(frame_id=5, frame_compile_id=0),
        }

        trace_lines = [
            # Only launch event (will trigger fake compilation)
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "fake_kernel",
                    "pid": 1000,
                    "stack": [
                        {"filename": "/user/code.py", "line": 10, "name": "main"},
                        {
                            "filename": kernel_path,
                            "line": 1,
                            "name": "kernel_fn",
                        },
                    ],
                    "compilation_metadata": {
                        "hash": "fake_hash",
                        "name": "fake_kernel",
                        "num_warps": 4,
                    },
                }
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = os.path.join(temp_dir, "test_trace.ndjson")
            with open(input_file, "w") as f:
                for line in trace_lines:
                    f.write(line + "\n")

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)

            parse_single_file(input_file, output_dir, kernel_compile_mapping=mapping)

            output_files = os.listdir(output_dir)
            frame_files = [f for f in output_files if f.startswith("f")]
            self.assertEqual(len(frame_files), 1)
            self.assertEqual(frame_files[0], "f5_fc0_a0_cai-.ndjson")


if __name__ == "__main__":
    unittest.main()
