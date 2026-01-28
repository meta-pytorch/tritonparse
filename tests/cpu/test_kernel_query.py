# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for kernel query functionality."""

import unittest

from tests.test_utils import get_test_ndjson_file
from tritonparse.info.kernel_query import (
    find_launch_index_by_kernel,
    find_similar_kernels,
    list_kernels,
    list_kernels_fast,
    list_launches_for_kernel,
)
from tritonparse.tools.prettify_ndjson import load_ndjson


class TestKernelQuery(unittest.TestCase):
    """Tests for kernel query functions."""

    def test_load_ndjson_gzip_support(self):
        """Test that load_ndjson can load .ndjson.gz files."""
        gz_file = get_test_ndjson_file()

        # Load and verify
        events = load_ndjson(gz_file)
        self.assertIsInstance(events, list)
        self.assertGreater(len(events), 0, "Should load at least one event")

        # Verify we have expected event types
        event_types = {e.get("event_type") for e in events if isinstance(e, dict)}
        self.assertTrue(
            "compilation" in event_types or "launch" in event_types,
            f"Expected compilation or launch events, got: {event_types}",
        )

        print(f"✓ Successfully loaded {len(events)} events from .ndjson.gz file")

    def test_list_kernels_empty(self):
        """Test listing kernels from empty events list."""
        events = []
        result = list_kernels(events)
        self.assertEqual(result, [])

    def test_list_kernels_single(self):
        """Test listing kernels with single kernel and multiple launches."""
        gz_file = get_test_ndjson_file()
        events = load_ndjson(gz_file)

        # Filter to only fused_op_kernel launches (4 launches)
        filtered_events = []
        for event in events:
            if event.get("event_type") == "launch":
                kernel_name = event.get("compilation_metadata", {}).get("name")
                if kernel_name == "fused_op_kernel":
                    filtered_events.append(event)
            else:
                # Keep non-launch events to test filtering
                filtered_events.append(event)

        result = list_kernels(filtered_events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "fused_op_kernel")
        self.assertEqual(result[0].total_launches, 4)

    def test_list_kernels_multiple(self):
        """Test listing kernels with multiple different kernels."""
        gz_file = get_test_ndjson_file()
        events = load_ndjson(gz_file)

        result = list_kernels(events)
        self.assertEqual(len(result), 2)

        # Check that results are sorted by name
        names = [k.name for k in result]
        self.assertEqual(names, ["fused_op_kernel", "matmul_kernel"])

        # Check launch counts
        kernel_dict = {k.name: k for k in result}
        self.assertEqual(kernel_dict["matmul_kernel"].total_launches, 1553)
        self.assertEqual(kernel_dict["fused_op_kernel"].total_launches, 4)

    def test_find_launch_index_valid(self):
        """Test finding valid kernel name and launch_id."""
        gz_file = get_test_ndjson_file()
        events = load_ndjson(gz_file)

        # Test first launch of fused_op_kernel (launch_id=0)
        index = find_launch_index_by_kernel(events, "fused_op_kernel", 0)
        self.assertEqual(events[index].get("event_type"), "launch")
        self.assertEqual(
            events[index].get("compilation_metadata", {}).get("name"),
            "fused_op_kernel",
        )

        # Test second launch of fused_op_kernel (launch_id=1)
        index = find_launch_index_by_kernel(events, "fused_op_kernel", 1)
        self.assertEqual(events[index].get("event_type"), "launch")
        self.assertEqual(
            events[index].get("compilation_metadata", {}).get("name"),
            "fused_op_kernel",
        )

        # Test first launch of matmul_kernel (launch_id=0)
        index = find_launch_index_by_kernel(events, "matmul_kernel", 0)
        self.assertEqual(events[index].get("event_type"), "launch")
        self.assertEqual(
            events[index].get("compilation_metadata", {}).get("name"),
            "matmul_kernel",
        )

    def test_find_launch_index_kernel_not_found(self):
        """Test that ValueError is raised when kernel not found."""
        gz_file = get_test_ndjson_file()
        events = load_ndjson(gz_file)

        with self.assertRaises(ValueError) as cm:
            find_launch_index_by_kernel(events, "nonexistent_kernel", 0)

        error_msg = str(cm.exception)
        self.assertIn("not found", error_msg)
        self.assertIn("nonexistent_kernel", error_msg)

    def test_find_launch_index_out_of_range(self):
        """Test that ValueError is raised when launch_id is out of range."""
        gz_file = get_test_ndjson_file()
        events = load_ndjson(gz_file)

        # fused_op_kernel has only 4 launches (0-3), test with launch_id=10
        with self.assertRaises(ValueError) as cm:
            find_launch_index_by_kernel(events, "fused_op_kernel", 10)

        error_msg = str(cm.exception)
        self.assertIn("has only 4 launches", error_msg)
        self.assertIn("--launch-id 10", error_msg)
        self.assertIn("Valid range: 0 to 3", error_msg)

    def test_info_kernel_query_functions(self):
        """Test info module kernel query functions."""
        gz_file = get_test_ndjson_file()
        events = load_ndjson(gz_file)

        # Test list_launches_for_kernel
        launches = list_launches_for_kernel(events, "fused_op_kernel")
        self.assertGreater(len(launches), 0)
        self.assertEqual(launches[0].launch_id, 0)
        self.assertIsInstance(launches[0].grid, list)

        # Test list_launches_for_kernel with non-existent kernel
        with self.assertRaises(ValueError) as cm:
            list_launches_for_kernel(events, "nonexistent_kernel")
        self.assertIn("not found", str(cm.exception))

        # Test find_similar_kernels
        similar = find_similar_kernels(events, "fused_op", n=3)
        self.assertGreater(len(similar), 0)
        self.assertIn("fused_op_kernel", similar)

        similar = find_similar_kernels(events, "fused_op_kernel", n=3)
        self.assertIn("fused_op_kernel", similar)

        similar = find_similar_kernels(events, "xyz_abc_123", n=3)
        self.assertEqual(len(similar), 0)

        # Test list_kernels_fast (should use launch_diff and match list_kernels)
        kernels_fast = list_kernels_fast(events)
        self.assertGreater(len(kernels_fast), 0)

        kernels_slow = list_kernels(events)
        fast_dict = {k.name: k.total_launches for k in kernels_fast}
        slow_dict = {k.name: k.total_launches for k in kernels_slow}
        self.assertEqual(fast_dict, slow_dict)


if __name__ == "__main__":
    unittest.main()
