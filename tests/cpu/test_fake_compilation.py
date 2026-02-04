# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for fake compilation event functionality."""

import json
import os
import tempfile
import unittest

from tritonparse.parse.trace_processor import (
    _create_fake_compilation,
    _prescan_for_fake_compilations,
    parse_single_file,
)


class TestFakeCompilation(unittest.TestCase):
    """Tests for fake compilation event generation."""

    def test_prescan_identifies_launches_without_compilations(self):
        """Test that prescan correctly identifies kernels with only launch events."""
        # Create a trace file with:
        # - kernel_hash_1: has both compilation and launch
        # - kernel_hash_2: only has launch (no compilation)
        trace_lines = [
            json.dumps(
                {
                    "event_type": "compilation",
                    "payload": {
                        "metadata": {"hash": "kernel_hash_1", "name": "kernel_1"}
                    },
                }
            ),
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "kernel_1",
                    "compilation_metadata": {"hash": "kernel_hash_1"},
                }
            ),
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "kernel_2",
                    "compilation_metadata": {"hash": "kernel_hash_2", "num_warps": 4},
                }
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False) as f:
            for line in trace_lines:
                f.write(line + "\n")
            temp_path = f.name

        try:
            compilation_hashes, first_launch_by_hash = _prescan_for_fake_compilations(
                temp_path
            )

            # kernel_hash_1 should be in compilation_hashes
            self.assertIn("kernel_hash_1", compilation_hashes)
            # kernel_hash_2 should NOT be in compilation_hashes
            self.assertNotIn("kernel_hash_2", compilation_hashes)

            # Both should be in first_launch_by_hash
            self.assertIn("kernel_hash_1", first_launch_by_hash)
            self.assertIn("kernel_hash_2", first_launch_by_hash)

            # Kernels needing fake = launches - compilations
            kernels_needing_fake = set(first_launch_by_hash.keys()) - compilation_hashes
            self.assertEqual(kernels_needing_fake, {"kernel_hash_2"})

        finally:
            os.unlink(temp_path)

    def test_create_fake_compilation_structure(self):
        """Test that fake compilation has correct structure."""
        launch_event = {
            "event_type": "launch",
            "name": "test_kernel",
            "pid": 12345,
            "timestamp": "2024-01-01T00:00:00Z",
            "stack": [{"filename": "/test.py", "line": 10, "name": "main"}],
            "compilation_metadata": {
                "hash": "test_hash",
                "name": "test_kernel",
                "num_warps": 4,
                "num_stages": 2,
                "num_ctas": 1,
            },
        }

        fake_comp = _create_fake_compilation(launch_event, "test_hash")

        # Verify basic structure
        self.assertEqual(fake_comp["event_type"], "compilation")
        self.assertTrue(fake_comp["is_fake"])
        self.assertIn("fake_reason", fake_comp)
        self.assertEqual(fake_comp["pid"], 12345)
        self.assertEqual(fake_comp["timestamp"], "2024-01-01T00:00:00Z")
        self.assertEqual(
            fake_comp["stack"],
            [{"filename": "/test.py", "line": 10, "name": "main"}],
        )

        # Verify payload structure
        payload = fake_comp["payload"]
        self.assertIn("metadata", payload)
        self.assertEqual(payload["metadata"]["hash"], "test_hash")
        self.assertEqual(payload["metadata"]["name"], "test_kernel")
        self.assertEqual(payload["metadata"]["num_warps"], 4)
        self.assertEqual(payload["metadata"]["num_stages"], 2)
        self.assertEqual(payload["metadata"]["num_ctas"], 1)

        # Verify empty IR content
        self.assertEqual(payload["file_content"], {})
        self.assertEqual(payload["file_path"], {})

    def test_create_fake_compilation_name_fallback(self):
        """Test that name falls back to compilation_metadata if not in launch event."""
        launch_event = {
            "event_type": "launch",
            # No "name" field
            "compilation_metadata": {
                "hash": "test_hash",
                "name": "kernel_from_metadata",
            },
        }

        fake_comp = _create_fake_compilation(launch_event, "test_hash")
        self.assertEqual(
            fake_comp["payload"]["metadata"]["name"], "kernel_from_metadata"
        )

    def test_parse_single_file_with_only_launches(self):
        """Test that parse_single_file handles files with only launch events."""
        # Create a trace file with only launch events (no compilations)
        trace_lines = [
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "orphan_kernel",
                    "pid": 1000,
                    "stack": [{"filename": "/test.py", "line": 1, "name": "main"}],
                    "compilation_metadata": {
                        "hash": "orphan_hash",
                        "name": "orphan_kernel",
                        "num_warps": 4,
                    },
                }
            ),
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "orphan_kernel",
                    "pid": 1000,
                    "stack": [{"filename": "/test.py", "line": 1, "name": "main"}],
                    "compilation_metadata": {
                        "hash": "orphan_hash",
                        "name": "orphan_kernel",
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

            # Parse the file
            parse_single_file(input_file, output_dir)

            # Check that output file was created
            output_files = os.listdir(output_dir)
            self.assertGreater(len(output_files), 0, "Should create output file")

            # Read and verify output
            output_file = os.path.join(output_dir, output_files[0])
            with open(output_file, "r") as f:
                lines = f.readlines()

            # Should have at least: fake compilation + 2 launches + launch_diff
            self.assertGreaterEqual(len(lines), 4)

            # First line should be the fake compilation
            first_event = json.loads(lines[0])
            self.assertEqual(first_event["event_type"], "compilation")
            self.assertTrue(first_event.get("is_fake", False))
            self.assertIn("fake_reason", first_event)

    def test_parse_single_file_mixed_events(self):
        """Test parse_single_file with mixed real and fake compilation scenarios."""
        trace_lines = [
            # Real compilation for kernel_1
            json.dumps(
                {
                    "event_type": "compilation",
                    "pid": 1000,
                    "stack": [],
                    "payload": {
                        "metadata": {"hash": "kernel_1_hash", "name": "kernel_1"},
                        "file_content": {},
                        "file_path": {},
                    },
                }
            ),
            # Launch for kernel_1 (has real compilation)
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "kernel_1",
                    "pid": 1000,
                    "stack": [],
                    "compilation_metadata": {"hash": "kernel_1_hash"},
                }
            ),
            # Launch for kernel_2 (no compilation - needs fake)
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "kernel_2",
                    "pid": 1000,
                    "stack": [],
                    "compilation_metadata": {
                        "hash": "kernel_2_hash",
                        "name": "kernel_2",
                        "num_warps": 8,
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

            parse_single_file(input_file, output_dir)

            # Check output
            output_files = os.listdir(output_dir)
            self.assertGreater(len(output_files), 0)

            # Read all events from output
            all_events = []
            for fname in output_files:
                with open(os.path.join(output_dir, fname), "r") as f:
                    for line in f:
                        all_events.append(json.loads(line))

            # Find compilation events
            compilations = [
                e for e in all_events if e.get("event_type") == "compilation"
            ]
            self.assertEqual(len(compilations), 2, "Should have 2 compilations")

            # One should be fake, one should be real
            fake_comps = [c for c in compilations if c.get("is_fake")]
            real_comps = [c for c in compilations if not c.get("is_fake")]
            self.assertEqual(len(fake_comps), 1, "Should have 1 fake compilation")
            self.assertEqual(len(real_comps), 1, "Should have 1 real compilation")

            # Verify fake compilation is for kernel_2
            self.assertEqual(
                fake_comps[0]["payload"]["metadata"]["hash"], "kernel_2_hash"
            )
            self.assertEqual(fake_comps[0]["payload"]["metadata"]["num_warps"], 8)


class TestFakeCompilationAutotune(unittest.TestCase):
    """Tests for fake compilation with autotune scenarios."""

    def test_fake_compilation_autotune_session(self):
        """Test that fake compilations are correctly associated with autotune sessions."""
        # Create trace with autotune-like stack (contains autotuner.py)
        autotune_stack = [
            {"filename": "/user/code.py", "line": 10, "name": "main"},
            {"filename": "triton/runtime/autotuner.py", "line": 100, "name": "run"},
            {"filename": "triton/runtime/jit.py", "line": 50, "name": "run"},
        ]

        trace_lines = [
            # Launch with autotune stack (no compilation)
            json.dumps(
                {
                    "event_type": "launch",
                    "name": "autotune_kernel",
                    "pid": 1000,
                    "stack": autotune_stack,
                    "compilation_metadata": {
                        "hash": "autotune_hash",
                        "name": "autotune_kernel",
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

            parse_single_file(input_file, output_dir)

            # Read output
            output_files = os.listdir(output_dir)
            all_events = []
            for fname in output_files:
                with open(os.path.join(output_dir, fname), "r") as f:
                    for line in f:
                        all_events.append(json.loads(line))

            # Should have fake compilation with correct stack
            compilations = [
                e for e in all_events if e.get("event_type") == "compilation"
            ]
            self.assertEqual(len(compilations), 1)
            self.assertTrue(compilations[0].get("is_fake"))
            self.assertEqual(compilations[0]["stack"], autotune_stack)


if __name__ == "__main__":
    unittest.main()
