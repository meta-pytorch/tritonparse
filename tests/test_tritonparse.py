"""
Comprehensive tests for tritonparse using unittest.
Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TRITONPARSE_DEBUG=1 python -m unittest tests.test_tritonparse -v
```
"""

import gzip
import json
import os
import shutil
import tempfile
import unittest
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
import torch._inductor.config as inductor_config
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import tritonparse.context_manager
import tritonparse.reproducer.orchestrator
import tritonparse.structured_logging
import tritonparse.utils
from triton import knobs  # @manual=//triton:triton
from triton.compiler import ASTSource, IRSource  # @manual=//triton:triton
from triton.knobs import CompileTimes  # @manual=//triton:triton
from tritonparse.common import is_fbcode
from tritonparse.shared_vars import TEST_KEEP_OUTPUT
from tritonparse.structured_logging import convert, extract_python_source_info
from tritonparse.tools.disasm import is_nvdisasm_available


def create_fresh_triton_cache():
    """Create a fresh Triton cache directory and return cache management context"""
    cache_dir = tempfile.mkdtemp(prefix="triton_cache_")
    return cache_dir, knobs.cache.scope()


def setup_fresh_triton_environment(cache_dir):
    """Setup fresh Triton environment with isolated cache"""
    # Set up isolated cache directory
    original_cache_dir = getattr(knobs.cache, "dir", None)
    knobs.cache.dir = cache_dir

    # Save and reset compilation settings
    original_always_compile = knobs.compilation.always_compile
    knobs.compilation.always_compile = True

    # Reset hooks to clean state
    original_jit_cache_hook = knobs.runtime.jit_cache_hook
    original_jit_post_compile_hook = knobs.runtime.jit_post_compile_hook
    original_launch_enter_hook = knobs.runtime.launch_enter_hook
    original_compilation_listener = knobs.compilation.listener

    knobs.runtime.jit_cache_hook = None
    knobs.runtime.jit_post_compile_hook = None
    knobs.runtime.launch_enter_hook = None
    knobs.compilation.listener = None

    return {
        "original_cache_dir": original_cache_dir,
        "original_always_compile": original_always_compile,
        "original_jit_cache_hook": original_jit_cache_hook,
        "original_jit_post_compile_hook": original_jit_post_compile_hook,
        "original_launch_enter_hook": original_launch_enter_hook,
        "original_compilation_listener": original_compilation_listener,
    }


def restore_triton_environment(original_settings):
    """Restore original Triton environment settings"""
    if original_settings["original_cache_dir"] is not None:
        knobs.cache.dir = original_settings["original_cache_dir"]

    knobs.compilation.always_compile = original_settings["original_always_compile"]
    knobs.runtime.jit_cache_hook = original_settings["original_jit_cache_hook"]
    knobs.runtime.jit_post_compile_hook = original_settings[
        "original_jit_post_compile_hook"
    ]
    knobs.runtime.launch_enter_hook = original_settings["original_launch_enter_hook"]
    knobs.compilation.listener = original_settings["original_compilation_listener"]


def clear_all_caches(*kernels):
    """
    Clear all compilation caches comprehensively.

    Args:
        *kernels: Triton kernel objects to clear device caches for.
                 Can pass multiple kernels or none at all.

    This function performs a comprehensive cache clearing operation:
    1. Resets PyTorch compiler state (torch.compiler, dynamo, inductor)
    2. Clears Triton kernel device caches and resets hashes for provided kernels
    3. Creates a new Triton cache directory

    Returns:
        tuple: (new_cache_dir, original_cache_dir) for cleanup purposes
    """
    print("\n=== Clearing all caches ===")

    # Reset torch compiler state
    torch.compiler.reset()
    torch._dynamo.reset()
    print("✓ Reset torch compiler, dynamo, and inductor state")

    # Clear Triton kernel device caches for all provided kernels
    kernels_cleared = 0
    for kernel in kernels:
        if hasattr(kernel, "device_caches"):
            for device_id in kernel.device_caches:
                # device_caches[device_id] is a tuple of cache objects
                device_cache_tuple = kernel.device_caches[device_id]
                for cache_obj in device_cache_tuple:
                    if hasattr(cache_obj, "clear"):
                        cache_obj.clear()
            kernel.hash = None  # Reset kernel hash to force recompilation
            kernels_cleared += 1

    if kernels_cleared > 0:
        print(
            f"✓ Cleared device caches and reset hashes for {kernels_cleared} kernel(s)"
        )
    else:
        print("✓ No kernels provided for device cache clearing")

    # Create a completely fresh cache directory
    new_cache_dir = tempfile.mkdtemp(prefix="triton_fresh_cache_")
    original_cache_dir = knobs.cache.dir
    knobs.cache.dir = new_cache_dir
    print(f"✓ Created fresh Triton cache directory: {new_cache_dir}")

    return new_cache_dir, original_cache_dir


class TestTritonparseCPU(unittest.TestCase):
    """CPU-only tests (no CUDA required)"""

    def _get_test_ndjson_file(self):
        """Get the test NDJSON file path."""
        gz_file = (
            Path(__file__).parent
            / "example_output/parsed_output_complex/dedicated_log_triton_trace_findhao__mapped.ndjson.gz"
        )
        self.assertTrue(gz_file.exists(), f"Test file not found: {gz_file}")
        return gz_file

    def setup_temp_reproduce_dir(self):
        """Setup temporary directory for reproduce tests."""
        temp_dir = tempfile.mkdtemp()
        out_dir = os.path.join(temp_dir, "repro_output")
        return temp_dir, out_dir

    def cleanup_temp_reproduce_dir(self, temp_dir):
        """Cleanup temporary directory for reproduce tests."""
        if not TEST_KEEP_OUTPUT:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_callsite_parsing(self):
        """Test parsing of callsite locations in TTIR/TTGIR"""
        from tritonparse.ir_parser import extract_loc_definitions
        from tritonparse.trace_processor import generate_source_mappings

        # Test MLIR callsite location definitions
        ir_with_callsite = """
module {
  #loc7 = loc("/tmp/test.py":1091:8)
  #loc57 = loc("/tmp/test.py":421:16)
  #loc58 = loc("/tmp/test.py":853:16)
  #loc190 = loc(callsite(#loc58 at #loc7))
  #loc220 = loc(callsite(#loc57 at #loc190))
  %0 = tt.load %ptr loc(#loc220)
}
"""
        # Extract loc definitions
        locs = extract_loc_definitions(ir_with_callsite)

        # Verify loc220 (nested callsite)
        self.assertIn("220", locs)
        self.assertEqual(locs["220"]["file"], "/tmp/test.py")
        self.assertEqual(locs["220"]["line"], 421)  # Inherited from callee loc57
        self.assertEqual(locs["220"]["column"], 16)
        self.assertTrue(locs["220"].get("is_callsite"))
        self.assertEqual(locs["220"]["callsite_callee"], "57")
        self.assertEqual(locs["220"]["callsite_caller"], "190")

        # Verify loc190 (simple callsite)
        self.assertIn("190", locs)
        self.assertEqual(locs["190"]["line"], 853)  # Inherited from callee loc58
        self.assertTrue(locs["190"].get("is_callsite"))
        self.assertEqual(locs["190"]["callsite_callee"], "58")
        self.assertEqual(locs["190"]["callsite_caller"], "7")

        # Test source mappings generation
        mappings = generate_source_mappings(ir_with_callsite, "ttir")

        # Find the line with tt.load
        line_with_load = None
        for line_num, content in enumerate(ir_with_callsite.split("\n"), start=1):
            if "tt.load" in content:
                line_with_load = str(line_num)
                break

        self.assertIsNotNone(line_with_load)
        self.assertIn(line_with_load, mappings)

        mapping = mappings[line_with_load]
        self.assertEqual(mapping["file"], "/tmp/test.py")
        self.assertEqual(mapping["line"], 421)  # From loc220 -> loc57
        self.assertTrue(mapping.get("is_callsite"))
        self.assertEqual(mapping["callsite_callee"], "57")
        self.assertEqual(mapping["callsite_caller"], "190")

        print("✓ Callsite parsing tests passed")

    def test_convert(self):
        """Test convert function with various data types"""
        # Test with primitive types
        assert convert(42) == 42
        assert convert("hello") == "hello"
        assert convert(3.14) == 3.14
        assert convert(None) is None
        assert convert(True) is True

        # Test with a dictionary
        test_dict = {"a": 1, "b": "string", "c": 3.14}
        assert convert(test_dict) == test_dict

        # Test with a list
        test_list = [1, "string", 3.14]
        assert convert(test_list) == test_list

        # Test with a dataclass
        @dataclass
        class TestDataClass:
            x: int
            y: str
            z: float

        test_dataclass = TestDataClass(x=42, y="hello", z=3.14)
        expected_dict = {"x": 42, "y": "hello", "z": 3.14}
        assert convert(test_dataclass) == expected_dict

        # Test with nested structures
        @dataclass
        class NestedDataClass:
            name: str
            value: int

        nested_structure = {
            "simple_key": "simple_value",
            "list_key": [1, 2, NestedDataClass(name="test", value=42)],
            "dict_key": {"nested_key": NestedDataClass(name="nested", value=100)},
        }

        expected_nested = {
            "simple_key": "simple_value",
            "list_key": [1, 2, {"name": "test", "value": 42}],
            "dict_key": {"nested_key": {"name": "nested", "value": 100}},
        }

        assert convert(nested_structure) == expected_nested

    def test_loc_alias_parsing(self):
        """Test parsing of location aliases in TTIR/TTGIR"""
        from tritonparse.ir_parser import extract_loc_definitions

        # Test case 1: Bare #loc reference (no number)
        ir_with_bare_loc = """
module {
  #loc = loc("/tmp/test.py":10:5)
  #loc13 = loc("x_ptr"(#loc))
  func @kernel(%arg0: !tt.ptr<f32> loc(#loc13)) {
    return loc(#loc)
  }
}
"""
        locs = extract_loc_definitions(ir_with_bare_loc)
        # Main #loc should be stored with "" key
        assert "" in locs, "Main #loc not found"
        assert locs[""]["file"] == "/tmp/test.py"
        assert locs[""]["line"] == 10
        # Alias #loc13 should resolve to same location
        assert "13" in locs, "#loc13 not found"
        assert locs["13"]["file"] == "/tmp/test.py"
        assert locs["13"]["line"] == 10
        assert locs["13"]["alias_name"] == "x_ptr"
        assert locs["13"]["alias_of"] == ""

        # Test case 2: Named alias with numbered reference
        ir_with_numbered_alias = """
#loc = loc("/tmp/test.py":5:0)
#loc2 = loc("/tmp/test.py":20:28)
#loc16 = loc("pid"(#loc2))
%0 = tt.get_program_id x : i32 loc(#loc16)
"""
        locs = extract_loc_definitions(ir_with_numbered_alias)
        assert "2" in locs
        assert locs["2"]["line"] == 20
        assert "16" in locs
        assert locs["16"]["file"] == "/tmp/test.py"
        assert locs["16"]["line"] == 20
        assert locs["16"]["alias_name"] == "pid"
        assert locs["16"]["alias_of"] == "2"

        # Test case 3: Simple alias (no name)
        ir_with_simple_alias = """
#loc = loc("/tmp/test.py":1:1)
#loc1 = loc("/tmp/test.py":15:10)
#loc20 = loc(#loc1)
%1 = arith.constant 0 : i32 loc(#loc20)
"""
        locs = extract_loc_definitions(ir_with_simple_alias)
        assert "1" in locs
        assert "20" in locs
        assert locs["20"]["file"] == "/tmp/test.py"
        assert locs["20"]["line"] == 15
        assert locs["20"]["alias_of"] == "1"
        assert "alias_name" not in locs["20"]

        # Test case 4: Definition line tracking
        assert "def_line" in locs[""]
        assert "def_line" in locs["1"]
        assert "def_line" in locs["20"]

        print("✓ All loc alias parsing tests passed")

    def test_load_ndjson_gzip_support(self):
        """Test that load_ndjson can load .ndjson.gz files."""
        from pathlib import Path

        from tritonparse.tools.prettify_ndjson import load_ndjson

        # Use existing .ndjson.gz test file
        gz_file = (
            Path(__file__).parent
            / "example_output/parsed_output_complex/dedicated_log_triton_trace_findhao__mapped.ndjson.gz"
        )

        # Verify file exists
        self.assertTrue(gz_file.exists(), f"Test file not found: {gz_file}")

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
        from tritonparse.info.kernel_query import list_kernels

        events = []
        result = list_kernels(events)
        self.assertEqual(result, [])

    def test_list_kernels_single(self):
        """Test listing kernels with single kernel and multiple launches."""
        from pathlib import Path

        from tritonparse.info.kernel_query import list_kernels
        from tritonparse.tools.prettify_ndjson import load_ndjson

        # Load real test data
        gz_file = (
            Path(__file__).parent
            / "example_output/parsed_output_complex/dedicated_log_triton_trace_findhao__mapped.ndjson.gz"
        )
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
        from pathlib import Path

        from tritonparse.info.kernel_query import list_kernels
        from tritonparse.tools.prettify_ndjson import load_ndjson

        # Load real test data
        gz_file = (
            Path(__file__).parent
            / "example_output/parsed_output_complex/dedicated_log_triton_trace_findhao__mapped.ndjson.gz"
        )
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
        from pathlib import Path

        from tritonparse.info.kernel_query import find_launch_index_by_kernel
        from tritonparse.tools.prettify_ndjson import load_ndjson

        # Load real test data
        gz_file = (
            Path(__file__).parent
            / "example_output/parsed_output_complex/dedicated_log_triton_trace_findhao__mapped.ndjson.gz"
        )
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
        from pathlib import Path

        from tritonparse.info.kernel_query import find_launch_index_by_kernel
        from tritonparse.tools.prettify_ndjson import load_ndjson

        # Load real test data
        gz_file = (
            Path(__file__).parent
            / "example_output/parsed_output_complex/dedicated_log_triton_trace_findhao__mapped.ndjson.gz"
        )
        events = load_ndjson(gz_file)

        with self.assertRaises(ValueError) as cm:
            find_launch_index_by_kernel(events, "nonexistent_kernel", 0)

        error_msg = str(cm.exception)
        self.assertIn("not found", error_msg)
        self.assertIn("nonexistent_kernel", error_msg)

    def test_find_launch_index_out_of_range(self):
        """Test that ValueError is raised when launch_id is out of range."""
        from pathlib import Path

        from tritonparse.info.kernel_query import find_launch_index_by_kernel
        from tritonparse.tools.prettify_ndjson import load_ndjson

        # Load real test data
        gz_file = (
            Path(__file__).parent
            / "example_output/parsed_output_complex/dedicated_log_triton_trace_findhao__mapped.ndjson.gz"
        )
        events = load_ndjson(gz_file)

        # fused_op_kernel has only 4 launches (0-3), test with launch_id=10
        with self.assertRaises(ValueError) as cm:
            find_launch_index_by_kernel(events, "fused_op_kernel", 10)

        error_msg = str(cm.exception)
        self.assertIn("has only 4 launches", error_msg)
        self.assertIn("--launch-id 10", error_msg)
        self.assertIn("Valid range: 0 to 3", error_msg)

    def test_reproduce_mutual_exclusivity(self):
        """Test that --line and --kernel/--launch-id are mutually exclusive."""
        import argparse

        from tritonparse.reproducer.cli import _add_reproducer_args

        parser = argparse.ArgumentParser()
        _add_reproducer_args(parser)

        # Test: both --line and --kernel provided should raise error
        # Create a mock parser with error method
        mock_parser = argparse.ArgumentParser()
        _add_reproducer_args(mock_parser)
        args = mock_parser.parse_args(
            ["test.ndjson", "--line", "5", "--kernel", "matmul_kernel"]
        )

        # The mutual exclusivity check happens in cli.py main()
        # We test that args are parsed correctly, and the check will happen there
        self.assertEqual(args.kernel, "matmul_kernel")
        self.assertEqual(args.line, 5)

        # Test: only --kernel should work (line defaults to 0, which is allowed)
        args = parser.parse_args(["test.ndjson", "--kernel", "matmul_kernel"])
        self.assertEqual(args.kernel, "matmul_kernel")
        self.assertEqual(args.line, 0)  # default value, allowed with --kernel

        # Test: only --line should work
        args = parser.parse_args(["test.ndjson", "--line", "5"])
        self.assertEqual(args.line, 5)
        self.assertIsNone(args.kernel)

    def test_reproduce_kernel_launch_id(self):
        """End-to-end test: reproduce using --kernel and --launch-id."""
        gz_file = self._get_test_ndjson_file()
        temp_dir, out_dir = self.setup_temp_reproduce_dir()

        try:
            # Test reproducing fused_op_kernel launch_id=0
            result = tritonparse.reproducer.orchestrator.reproduce(
                input_path=str(gz_file),
                line_index=0,  # Placeholder, will be recalculated from kernel_name
                out_dir=out_dir,
                template="example",
                kernel_name="fused_op_kernel",
                launch_id=0,
            )

            # Verify output structure
            self.assertIn("kernel", result)
            self.assertIn("repro_script", result)
            self.assertIn("repro_context", result)
            self.assertTrue(os.path.exists(result["repro_script"]))
            self.assertTrue(os.path.exists(result["repro_context"]))

            # Verify the script contains kernel name
            script_content = Path(result["repro_script"]).read_text()
            self.assertIn("fused_op_kernel", script_content)

        finally:
            self.cleanup_temp_reproduce_dir(temp_dir)

    def test_reproduce_kernel_not_found(self):
        """Test that proper error is raised when kernel not found."""
        gz_file = self._get_test_ndjson_file()
        temp_dir, out_dir = self.setup_temp_reproduce_dir()

        try:
            with self.assertRaises(ValueError) as cm:
                tritonparse.reproducer.orchestrator.reproduce(
                    input_path=str(gz_file),
                    line_index=0,  # Placeholder, will be recalculated from kernel_name
                    out_dir=out_dir,
                    template="example",
                    kernel_name="nonexistent_kernel",
                    launch_id=0,
                )

            error_msg = str(cm.exception)
            self.assertIn("not found", error_msg)
            self.assertIn("nonexistent_kernel", error_msg)

        finally:
            self.cleanup_temp_reproduce_dir(temp_dir)

    def test_reproduce_launch_id_out_of_range(self):
        """Test that proper error is raised when launch_id is out of range."""
        gz_file = self._get_test_ndjson_file()
        temp_dir, out_dir = self.setup_temp_reproduce_dir()

        try:
            # fused_op_kernel has only 4 launches (0-3), test with launch_id=10
            with self.assertRaises(ValueError) as cm:
                tritonparse.reproducer.orchestrator.reproduce(
                    input_path=str(gz_file),
                    line_index=0,  # Placeholder, will be recalculated from kernel_name
                    out_dir=out_dir,
                    template="example",
                    kernel_name="fused_op_kernel",
                    launch_id=10,
                )

            error_msg = str(cm.exception)
            self.assertIn("has only 4 launches", error_msg)
            self.assertIn("--launch-id 10", error_msg)
            self.assertIn("Valid range: 0 to 3", error_msg)

        finally:
            self.cleanup_temp_reproduce_dir(temp_dir)

    def test_info_kernel_query_functions(self):
        """Test info module kernel query functions."""
        from tritonparse.info.kernel_query import (
            find_similar_kernels,
            list_kernels,
            list_kernels_fast,
            list_launches_for_kernel,
        )
        from tritonparse.tools.prettify_ndjson import load_ndjson

        gz_file = self._get_test_ndjson_file()
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

    def test_info_list_kernels(self):
        """Integration test: info command lists all kernels."""
        import sys
        from io import StringIO

        from tritonparse.info.cli import info_command

        gz_file = self._get_test_ndjson_file()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            info_command(str(gz_file), kernel_name=None)
            output = captured_output.getvalue()
            self.assertIn("Kernels in", output)
            self.assertIn("launches", output)
        finally:
            sys.stdout = old_stdout

    def test_info_kernel_launches(self):
        """Integration test: info command lists launches for specific kernel."""
        import sys
        from io import StringIO

        from tritonparse.info.cli import info_command

        gz_file = self._get_test_ndjson_file()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            info_command(str(gz_file), kernel_name="fused_op_kernel")
            output = captured_output.getvalue()
            self.assertIn("Launches for 'fused_op_kernel'", output)
            self.assertIn("id=", output)
            self.assertIn("line", output)
        finally:
            sys.stdout = old_stdout

    def test_info_kernel_not_found(self):
        """Integration test: info command handles kernel not found."""
        import sys
        from io import StringIO

        from tritonparse.info.cli import info_command

        gz_file = self._get_test_ndjson_file()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            with self.assertRaises(ValueError):
                info_command(str(gz_file), kernel_name="nonexistent_kernel")
            output = captured_output.getvalue()
            self.assertIn("not found", output)
        finally:
            sys.stdout = old_stdout


class TestTritonparseCUDA(unittest.TestCase):
    """CUDA tests (require GPU)"""

    def setUp(self):
        """Set up triton hooks and compilation settings"""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.cuda_device = torch.device("cuda:0")

        # Set up fresh Triton cache environment
        self.triton_cache_dir, self.cache_scope = create_fresh_triton_cache()
        self.cache_scope.__enter__()  # Enter the cache scope context
        self.original_triton_settings = setup_fresh_triton_environment(
            self.triton_cache_dir
        )

        # Save original settings for restoration
        self.prev_listener = knobs.compilation.listener
        self.prev_always_compile = knobs.compilation.always_compile
        self.prev_jit_post_compile_hook = knobs.runtime.jit_post_compile_hook
        self.prev_launch_enter_hook = knobs.runtime.launch_enter_hook

    def tearDown(self):
        """Restore original triton settings"""
        # Always restore original settings, even if test fails
        try:
            # Restore Triton environment
            restore_triton_environment(self.original_triton_settings)

            # Exit cache scope and cleanup
            self.cache_scope.__exit__(None, None, None)
            if os.path.exists(self.triton_cache_dir):
                shutil.rmtree(self.triton_cache_dir, ignore_errors=True)

        except Exception as e:
            print(f"Warning: Failed to cleanup Triton environment: {e}")

    def setup_test_with_fresh_cache(self):
        """Setup individual test with completely fresh cache"""
        # Create a new cache directory for this specific test
        test_cache_dir = tempfile.mkdtemp(prefix="triton_test_cache_")

        # Save current cache dir and set new one
        prev_cache_dir = knobs.cache.dir
        knobs.cache.dir = test_cache_dir

        return test_cache_dir, prev_cache_dir

    def cleanup_test_cache(self, test_cache_dir, prev_cache_dir):
        """Cleanup test-specific cache"""
        # Restore previous cache dir
        knobs.cache.dir = prev_cache_dir

        # Cleanup test cache directory
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir, ignore_errors=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_extract_python_source_info(self):
        """Test extract_python_source_info function"""

        # Define kernel inside the test function
        @triton.jit
        def extract_test_kernel(
            x_ptr,
            y_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = x * 3.0  # Simple operation: multiply by 3
            tl.store(y_ptr + offsets, y, mask=mask)

        trace_data = defaultdict(dict)

        def compile_listener(
            src: Union[ASTSource, IRSource],
            metadata: dict[str, str],
            metadata_group: dict[str, Any],
            times: CompileTimes,
            cache_hit: bool,
        ) -> None:
            nonlocal trace_data
            extract_python_source_info(trace_data, src)

        # Set up compilation listener
        triton.knobs.compilation.listener = compile_listener

        torch.manual_seed(0)
        size = (512, 512)
        a = torch.randn(size, device=self.cuda_device, dtype=torch.float32)

        # Use the kernel defined inside this test function
        n_elements = a.numel()
        c = torch.empty_like(a)
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        extract_test_kernel[grid](a, c, n_elements, BLOCK_SIZE)

        torch.cuda.synchronize()
        assert "python_source" in trace_data
        assert "file_path" in trace_data["python_source"]
        triton.knobs.compilation.listener = None

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_whole_workflow(self):
        """Test unified_parse functionality including SASS extraction"""

        # Define a simple kernel directly in the test function
        @triton.jit
        def test_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = x + 1.0  # Simple operation: add 1
            tl.store(y_ptr + offsets, y, mask=mask)

        # Simple function to run the kernel
        def run_test_kernel(x):
            n_elements = x.numel()
            y = torch.empty_like(x)
            BLOCK_SIZE = 256
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            test_kernel[grid](x, y, n_elements, BLOCK_SIZE)
            return y

        # Set up test environment
        temp_dir = tempfile.mkdtemp()
        temp_dir_logs = os.path.join(temp_dir, "logs")
        temp_dir_parsed = os.path.join(temp_dir, "parsed_output")
        os.makedirs(temp_dir_logs, exist_ok=True)
        os.makedirs(temp_dir_parsed, exist_ok=True)
        print(f"Temporary directory: {temp_dir}")
        nvdisasm_available = is_nvdisasm_available()
        if nvdisasm_available:
            print("✓ nvdisasm tool is available, enabling SASS dumping")
        else:
            print("⚠️  nvdisasm tool not available, SASS dumping will be disabled")

        # Initialize logging with conditional SASS dumping
        tritonparse.structured_logging.init(
            temp_dir_logs,
            enable_trace_launch=True,
            enable_sass_dump=nvdisasm_available,
        )

        # Generate test data and run kernels
        torch.manual_seed(0)
        size = (512, 512)  # Smaller size for faster testing
        x = torch.randn(size, device=self.cuda_device, dtype=torch.float32)

        # Run kernel twice to generate compilation and launch events
        run_test_kernel(x)
        run_test_kernel(x)
        torch.cuda.synchronize()

        # Verify log directory
        assert os.path.exists(
            temp_dir_logs
        ), f"Log directory {temp_dir_logs} does not exist."
        log_files = os.listdir(temp_dir_logs)
        assert len(log_files) > 0, (
            f"No log files found in {temp_dir_logs}. "
            "Expected log files to be generated during Triton compilation."
        )
        print(f"Found {len(log_files)} log files in {temp_dir_logs}: {log_files}")

        def check_event_type_counts_in_logs(log_dir: str) -> dict:
            """Count 'launch' and unique 'compilation' events in all log files and verify SASS content"""
            event_counts = {"launch": 0, "sass_found": False}
            # Track unique compilation hashes
            compilation_hashes = set()

            for log_file in os.listdir(log_dir):
                if log_file.endswith(".ndjson"):
                    log_file_path = os.path.join(log_dir, log_file)
                    with open(log_file_path, "r") as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                event_data = json.loads(line.strip())
                                event_type = event_data.get("event_type")
                                if event_type == "launch":
                                    event_counts["launch"] += 1
                                    print(
                                        f"  Line {line_num}: event_type = 'launch' (count: {event_counts['launch']})"
                                    )
                                elif event_type == "compilation":
                                    # Extract hash from compilation metadata
                                    compilation_hash = (
                                        event_data.get("payload", {})
                                        .get("metadata", {})
                                        .get("hash")
                                    )
                                    if compilation_hash:
                                        compilation_hashes.add(compilation_hash)
                                        print(
                                            f"  Line {line_num}: event_type = 'compilation' (unique hash: {compilation_hash[:8]}...)"
                                        )

                                    # Check for SASS content in compilation events
                                    file_content = event_data.get("payload", {}).get(
                                        "file_content", {}
                                    )
                                    sass_files = [
                                        key
                                        for key in file_content.keys()
                                        if key.endswith(".sass")
                                    ]

                                    if sass_files and not event_counts["sass_found"]:
                                        event_counts["sass_found"] = True
                                        sass_content = file_content[sass_files[0]]
                                        print(f"✓ Found SASS file: {sass_files[0]}")
                                        print(
                                            f"  SASS content preview (first 200 chars): {sass_content[:200]}..."
                                        )

                                        # Verify SASS content looks like assembly
                                        assert (
                                            "Function:" in sass_content
                                        ), "SASS content should contain function declaration"
                                        # Basic check for NVIDIA GPU assembly patterns
                                        assert any(
                                            pattern in sass_content.lower()
                                            for pattern in [
                                                "mov",
                                                "add",
                                                "mul",
                                                "ld",
                                                "st",
                                                "lop",
                                                "s2r",
                                            ]
                                        ), "SASS content should contain GPU assembly instructions"

                            except (json.JSONDecodeError, KeyError, TypeError) as e:
                                print(f"  Line {line_num}: Error processing line - {e}")

            # Add the count of unique compilation hashes to the event_counts
            event_counts["compilation"] = len(compilation_hashes)
            print(
                f"Event type counts: {event_counts} (unique compilation hashes: {len(compilation_hashes)})"
            )
            return event_counts

        # Verify event counts and conditional SASS extraction
        event_counts = check_event_type_counts_in_logs(temp_dir_logs)
        assert (
            event_counts["compilation"] == 1
        ), f"Expected 1 unique 'compilation' hash, found {event_counts['compilation']}"
        assert (
            event_counts["launch"] == 2
        ), f"Expected 2 'launch' events, found {event_counts['launch']}"

        # Conditionally verify SASS content based on nvdisasm availability
        if nvdisasm_available:
            assert event_counts[
                "sass_found"
            ], "SASS content was not found in compilation events"
            print("✓ Successfully verified SASS extraction functionality")
        else:
            print("⚠️  SASS verification skipped: nvdisasm not available")

        print(
            "✓ Verified correct event type counts: 1 unique compilation hash, 2 launch events"
        )

        # Test parsing functionality
        tritonparse.utils.unified_parse(
            source=temp_dir_logs, out=temp_dir_parsed, overwrite=True
        )
        try:
            # Verify parsing output
            parsed_files = os.listdir(temp_dir_parsed)
            assert len(parsed_files) > 0, "No files found in parsed output directory"

            # Verify that SASS is preserved in parsed output
            ndjson_gz_files = [f for f in parsed_files if f.endswith(".ndjson.gz")]
            assert (
                len(ndjson_gz_files) > 0
            ), "No .ndjson.gz files found in parsed output"

            sass_found_in_parsed = False
            for ndjson_gz_file in ndjson_gz_files:
                ndjson_gz_path = os.path.join(temp_dir_parsed, ndjson_gz_file)
                with gzip.open(ndjson_gz_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event_data = json.loads(line.strip())
                            if event_data.get("event_type") == "compilation":
                                file_content = event_data.get("payload", {}).get(
                                    "file_content", {}
                                )
                                sass_files = [
                                    key
                                    for key in file_content.keys()
                                    if key.endswith(".sass")
                                ]
                                if sass_files:
                                    sass_found_in_parsed = True
                                    print("✓ SASS content preserved in parsed output")
                                    break
                        except json.JSONDecodeError:
                            continue

                if sass_found_in_parsed:
                    break

            # Conditionally verify SASS content is preserved in parsed output
            if nvdisasm_available:
                assert (
                    sass_found_in_parsed
                ), "SASS content was not preserved in parsed output"
            else:
                print(
                    "⚠️  SASS preservation verification skipped: nvdisasm not available"
                )

        finally:
            # Clean up
            if TEST_KEEP_OUTPUT:
                print(
                    f"✓ Preserving temporary directory (TEST_KEEP_OUTPUT=1): {temp_dir}"
                )
            else:
                shutil.rmtree(temp_dir)
                print("✓ Cleaned up temporary directory")
            tritonparse.structured_logging.clear_logging_config()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_context_manager_with_split_compilations(self):
        """Test TritonParseManager context manager with split_inductor_compilations parameter"""

        # Setup fresh cache for this test (on top of the class-level fresh cache)
        test_cache_dir, prev_cache_dir = self.setup_test_with_fresh_cache()

        # Define Triton kernel
        @triton.jit
        def add_kernel(
            a_ptr,
            b_ptr,
            c_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
            c = a + b
            tl.store(c_ptr + offsets, c, mask=mask)

        def tensor_add_triton(a, b):
            n_elements = a.numel()
            c = torch.empty_like(a)
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
            return c

        # Simple function for torch.compile (triggers inductor compilation)
        def simple_add(a, b):
            return a + b

        # Prepare test data
        torch.manual_seed(0)
        size = (512, 512)
        a = torch.randn(size, device=self.cuda_device, dtype=torch.float32)
        b = torch.randn(size, device=self.cuda_device, dtype=torch.float32)

        # Create temp directories for output
        temp_output_dir_split_true = tempfile.mkdtemp()
        temp_output_dir_split_false = tempfile.mkdtemp()

        # Test 1: split_inductor_compilations=True
        print("\n=== Testing split_inductor_compilations=True ===")
        with tritonparse.context_manager.TritonParseManager(
            enable_trace_launch=True,
            split_inductor_compilations=True,
            out=temp_output_dir_split_true,
        ) as manager:
            assert os.path.exists(manager.dir_path), "Temporary directory should exist"
            print(f"Temporary directory created: {manager.dir_path}")

            # Run Triton kernel
            c_triton = tensor_add_triton(a, b)
            c_triton.sum()
            torch.compiler.reset()
            with inductor_config.patch(force_disable_caches=True):
                # Run torch.compile to trigger inductor compilation
                compiled_add = torch.compile(simple_add)
                c_compiled = compiled_add(a, b)
                c_compiled.sum()

            torch.cuda.synchronize()

            # Verify log files are generated
            log_files = os.listdir(manager.dir_path)
            assert len(log_files) > 0, "Log files should be generated"
            print(f"Generated {len(log_files)} log file(s)")
        # After exiting context manager, verify behavior
        # Verify parsed output exists
        assert os.path.exists(
            temp_output_dir_split_true
        ), "Parsed output directory should exist"
        print(f"Parsed output directory: {temp_output_dir_split_true}")

        # Check output files for split=True
        output_files_split_true = sorted(os.listdir(temp_output_dir_split_true))
        num_files_split_true = len(output_files_split_true)
        print(f"Output files (split=True): {num_files_split_true} files")
        for f in output_files_split_true:
            print(f"  - {f}")

        # === Clear caches between tests ===
        second_test_cache_dir, original_cache_dir = clear_all_caches(add_kernel)

        # Test 2: split_inductor_compilations=False
        print("\n=== Testing split_inductor_compilations=False ===")
        with tritonparse.context_manager.TritonParseManager(
            enable_trace_launch=True,
            split_inductor_compilations=False,
            out=temp_output_dir_split_false,
        ) as manager:
            assert os.path.exists(manager.dir_path), "Temporary directory should exist"
            print(f"Temporary directory created: {manager.dir_path}")

            # Run the same operations
            c_triton = tensor_add_triton(a, b)
            c_triton.sum()
            torch.compiler.reset()
            with inductor_config.patch(force_disable_caches=True):
                compiled_add = torch.compile(simple_add)
                c_compiled = compiled_add(a, b)
                c_compiled.sum()

            torch.cuda.synchronize()

            log_files = os.listdir(manager.dir_path)
            assert len(log_files) > 0, "Log files should be generated"
            print(f"Generated {len(log_files)} log file(s)")
        # After exiting context manager, verify behavior
        # Verify parsed output exists
        assert os.path.exists(
            temp_output_dir_split_false
        ), "Parsed output directory should exist"
        print(f"Parsed output directory: {temp_output_dir_split_false}")

        # Check output files for split=False
        output_files_split_false = sorted(os.listdir(temp_output_dir_split_false))
        num_files_split_false = len(output_files_split_false)
        print(f"Output files (split=False): {num_files_split_false} files")
        for f in output_files_split_false:
            print(f"  - {f}")

        # Check compilation events in parsed output for split=False
        ndjson_gz_files_split_false = [
            f for f in output_files_split_false if f.endswith(".ndjson.gz")
        ]
        assert (
            len(ndjson_gz_files_split_false) > 0
        ), "No .ndjson.gz files found in split=False parsed output"

        compilation_count_split_false = 0
        compilation_names_found = []
        expected_compilation_names = {"add_kernel", "triton_poi_fused_add_0"}

        for ndjson_gz_file in ndjson_gz_files_split_false:
            ndjson_gz_path = os.path.join(temp_output_dir_split_false, ndjson_gz_file)
            with gzip.open(ndjson_gz_path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        event_data = json.loads(line.strip())
                        if event_data.get("event_type") == "compilation":
                            compilation_count_split_false += 1

                            # Extract and validate the compilation name
                            compilation_name = (
                                event_data.get("payload", {})
                                .get("metadata", {})
                                .get("name")
                            )
                            if compilation_name:
                                compilation_names_found.append(compilation_name)
                                assert compilation_name in expected_compilation_names, (
                                    f"Unexpected compilation name: '{compilation_name}'. "
                                    f"Expected one of: {expected_compilation_names}"
                                )
                    except json.JSONDecodeError:
                        continue

        print(
            f"Compilation events found (split=False): {compilation_count_split_false}"
        )
        print(f"Compilation names found: {compilation_names_found}")

        assert (
            compilation_count_split_false > 0
        ), "Expected at least 1 compilation event in split=False output"

        # Verify all compilation names are from the expected set
        unique_names_found = set(compilation_names_found)
        assert unique_names_found.issubset(expected_compilation_names), (
            f"Found unexpected compilation names: {unique_names_found - expected_compilation_names}. "
            f"Expected only: {expected_compilation_names}"
        )
        print(f"✓ All compilation names are valid: {unique_names_found}")

        # Verify the key difference: split=False should have one fewer file
        assert (
            num_files_split_false == num_files_split_true - 1
        ), f"split=False should have one fewer file (expected {num_files_split_true - 1}, got {num_files_split_false})"
        print(
            f"✓ Verified: split=False has {num_files_split_false} files, split=True has {num_files_split_true} files (difference: 1)"
        )

        # Clean up test outputs
        try:
            if TEST_KEEP_OUTPUT:
                print(
                    f"\n✓ Preserving output directories (TEST_KEEP_OUTPUT=1):\n  split=True: {temp_output_dir_split_true}\n  split=False: {temp_output_dir_split_false}"
                )
            else:
                if os.path.exists(temp_output_dir_split_true):
                    shutil.rmtree(temp_output_dir_split_true)
                if os.path.exists(temp_output_dir_split_false):
                    shutil.rmtree(temp_output_dir_split_false)
                print("✓ Cleaned up output directories")
        except Exception as e:
            print(f"Warning: Failed to clean up output directories: {e}")

        finally:
            # Cleanup test-specific caches
            self.cleanup_test_cache(test_cache_dir, prev_cache_dir)

            # Cleanup second test cache directory
            if "second_test_cache_dir" in locals():
                knobs.cache.dir = original_cache_dir  # Restore cache dir first
                if os.path.exists(second_test_cache_dir):
                    shutil.rmtree(second_test_cache_dir, ignore_errors=True)
                    print(f"✓ Cleaned up second test cache: {second_test_cache_dir}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_complex_kernels(self):
        """
        A more complex test case involving two distinct Triton kernels, one of which uses autotuning.
        This test is designed to validate the launch_diff functionality with multiple, varied launches.
        """

        # Kernel 1: Autotuned Matmul (simplified configs for small scale)
        @triton.autotune(
            configs=[
                triton.Config(
                    {
                        "BLOCK_SIZE_M": 16,
                        "BLOCK_SIZE_N": 16,
                        "BLOCK_SIZE_K": 16,
                        "GROUP_SIZE_M": 1,
                    },
                    num_stages=1,
                    num_warps=1,
                ),
                triton.Config(
                    {
                        "BLOCK_SIZE_M": 32,
                        "BLOCK_SIZE_N": 16,
                        "BLOCK_SIZE_K": 16,
                        "GROUP_SIZE_M": 1,
                    },
                    num_stages=1,
                    num_warps=1,
                ),
                triton.Config(
                    {
                        "BLOCK_SIZE_M": 16,
                        "BLOCK_SIZE_N": 32,
                        "BLOCK_SIZE_K": 16,
                        "GROUP_SIZE_M": 1,
                    },
                    num_stages=1,
                    num_warps=1,
                ),
            ],
            key=["M", "N", "K"],
        )
        @triton.jit
        def matmul_kernel(
            a,
            b,
            c,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr,
            ACTIVATION: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (pid % group_size)
            pid_n = (pid % num_pid_in_group) // group_size

            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a_block = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b_block = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )
                accumulator += tl.dot(a_block, b_block)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk
            c_block = accumulator.to(tl.float16)

            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c_block, mask=c_mask)

        def matmul(a, b):
            assert a.shape[1] == b.shape[0], "Incompatible dimensions"
            M, K = a.shape
            K, N = b.shape
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)

            def grid(META):
                return (
                    triton.cdiv(M, META["BLOCK_SIZE_M"])
                    * triton.cdiv(N, META["BLOCK_SIZE_N"]),
                )

            matmul_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                ACTIVATION=None,
            )
            return c

        # Kernel 2: Fused element-wise operation
        @triton.jit
        def fused_op_kernel(
            a_ptr,
            b_ptr,
            c_ptr,
            output_ptr,
            n_elements,
            scale_factor: float,
            ACTIVATION: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            a = tl.load(a_ptr + offsets, mask=mask)
            b = tl.load(b_ptr + offsets, mask=mask)
            c = tl.load(c_ptr + offsets, mask=mask)

            result = a * b * scale_factor + c
            if ACTIVATION == "relu":
                result = tl.where(result > 0, result, 0.0)

            tl.store(output_ptr + offsets, result, mask=mask)

        def fused_op(a, b, c, scale_factor: float, activation: str):
            n_elements = a.numel()
            output = torch.empty_like(a)
            BLOCK_SIZE = 8  # Reduced from 1024 for small scale testing
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            fused_op_kernel[grid](
                a,
                b,
                c,
                output,
                n_elements,
                scale_factor,
                ACTIVATION=activation,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return output

        # Set up test environment
        temp_dir = tempfile.mkdtemp()
        log_path = os.path.join(temp_dir, "logs_complex")
        parsed_output_path = os.path.join(temp_dir, "parsed_output_complex")
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(parsed_output_path, exist_ok=True)
        print(f"Temporary directory: {temp_dir}")

        # Initialize logging
        tritonparse.structured_logging.init(log_path, enable_trace_launch=True)

        try:
            # Main test function logic
            torch.manual_seed(0)

            # --- Matmul Launches (3 times with different configs) ---
            print("--- Testing Matmul Kernel (3 launches) ---")
            # Launch 1
            a1 = torch.randn((16, 16), device="cuda", dtype=torch.float16)
            b1 = torch.randn((16, 16), device="cuda", dtype=torch.float16)
            c1 = matmul(a1, b1)
            c1.sum()  # Synchronize
            print("Matmul Launch 1 (16x16 @ 16x16) done.")

            # Launch 2
            a2 = torch.randn((32, 16), device="cuda", dtype=torch.float16)
            b2 = torch.randn((16, 32), device="cuda", dtype=torch.float16)
            c2 = matmul(a2, b2)
            c2.sum()  # Synchronize
            print("Matmul Launch 2 (32x16 @ 16x32) done.")

            # Launch 3
            a3 = torch.randn((16, 32), device="cuda", dtype=torch.float16)
            b3 = torch.randn((32, 16), device="cuda", dtype=torch.float16)
            c3 = matmul(a3, b3)
            c3.sum()  # Synchronize
            print("Matmul Launch 3 (16x32 @ 32x16) done.")

            # --- Fused Op Launches (4 times with different parameters) ---
            print("\n--- Testing Fused Op Kernel (4 launches) ---")
            x = torch.randn((8,), device="cuda", dtype=torch.float32)
            y = torch.randn((8,), device="cuda", dtype=torch.float32)
            z = torch.randn((8,), device="cuda", dtype=torch.float32)

            # Launch 1
            print("Fused Op Launch 1: scale=1.0, activation=None")
            out1 = fused_op(x, y, z, scale_factor=1.0, activation="none")
            out1.sum()  # Synchronize

            # Launch 2
            print("Fused Op Launch 2: scale=2.5, activation=None")
            out2 = fused_op(x, y, z, scale_factor=2.5, activation="none")
            out2.sum()  # Synchronize

            # Launch 3
            print("Fused Op Launch 3: scale=1.0, activation='relu'")
            out3 = fused_op(x, y, z, scale_factor=1.0, activation="relu")
            out3.sum()  # Synchronize

            # Launch 4 (different size)
            print("Fused Op Launch 4: scale=1.0, activation='relu', different size")
            x_large = torch.randn((6,), device="cuda", dtype=torch.float32)
            y_large = torch.randn((6,), device="cuda", dtype=torch.float32)
            z_large = torch.randn((6,), device="cuda", dtype=torch.float32)
            out4 = fused_op(
                x_large, y_large, z_large, scale_factor=1.0, activation="relu"
            )
            out4.sum()  # Synchronize
            print("All kernels executed.")

            # Use unified_parse to process the generated logs
            tritonparse.utils.unified_parse(
                source=log_path, out=parsed_output_path, overwrite=True
            )

            # Verify that logs and parsed output were generated
            log_files = os.listdir(log_path)
            assert len(log_files) > 0, f"No log files found in {log_path}"
            print(f"✓ Generated {len(log_files)} log files")

            parsed_files = os.listdir(parsed_output_path)
            assert (
                len(parsed_files) > 0
            ), f"No parsed files found in {parsed_output_path}"
            print(f"✓ Generated {len(parsed_files)} parsed files")

            # Verify we have both json and ndjson.gz files
            json_files = [f for f in parsed_files if f.endswith(".json")]
            ndjson_gz_files = [f for f in parsed_files if f.endswith(".ndjson.gz")]

            assert len(json_files) > 0, f"No .json files found in {parsed_output_path}"
            assert (
                len(ndjson_gz_files) > 0
            ), f"No .ndjson.gz files found in {parsed_output_path}"
            print(
                f"✓ Found {len(json_files)} .json files and {len(ndjson_gz_files)} .ndjson.gz files"
            )

            # Unzip and check launch_diff events in the .ndjson.gz file
            for ndjson_gz_file in ndjson_gz_files:
                ndjson_gz_path = os.path.join(parsed_output_path, ndjson_gz_file)
                launch_diff_count = 0

                print(f"Checking launch_diff events in {ndjson_gz_file}")
                with gzip.open(ndjson_gz_path, "rt", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            event_data = json.loads(line.strip())
                            event_type = event_data.get("event_type")
                            if event_type == "launch_diff":
                                launch_diff_count += 1
                                print(
                                    f"  Line {line_num}: Found launch_diff event (count: {launch_diff_count})"
                                )
                        except json.JSONDecodeError as e:
                            print(f"  Line {line_num}: JSON decode error - {e}")
                        except Exception as e:
                            print(f"  Line {line_num}: Error processing line - {e}")

                print(f"✓ Total launch_diff events found: {launch_diff_count}")
                assert (
                    launch_diff_count == 5
                ), f"Expected 5 launch_diff events, found {launch_diff_count}"
                print("✓ Verified 5 launch_diff events in parsed output")

        finally:
            # Clean up
            if TEST_KEEP_OUTPUT:
                print(
                    f"✓ Preserving temporary directory (TEST_KEEP_OUTPUT=1): {temp_dir}"
                )
            else:
                shutil.rmtree(temp_dir)
                print("✓ Cleaned up temporary directory")
            tritonparse.structured_logging.clear_logging_config()

    @unittest.skipIf(is_fbcode(), "Skip in internal FB environment")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_reproducer_end_to_end(self):
        """End-to-end test for reproducer: generate logs, build script, run it."""
        import subprocess as _subprocess
        import sys as _sys
        from pathlib import Path as _Path

        # 1) Prepare temp dirs
        temp_dir = tempfile.mkdtemp()
        logs_dir = os.path.join(temp_dir, "logs")
        out_dir = os.path.join(temp_dir, "repro_output")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # 2) Write a simple module-level Triton kernel to a temp file
        kernel_dir = os.path.join(temp_dir, "kernels")
        os.makedirs(kernel_dir, exist_ok=True)
        kernel_file = os.path.join(kernel_dir, "simple_kernel.py")
        kernel_src = (
            "import triton\n"
            "import triton.language as tl\n"
            "import torch\n"
            "\n"
            "@triton.jit\n"
            "def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(axis=0)\n"
            "    block_start = pid * BLOCK_SIZE\n"
            "    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n_elements\n"
            "    x = tl.load(x_ptr + offsets, mask=mask)\n"
            "    y = tl.load(y_ptr + offsets, mask=mask)\n"
            "    tl.store(out_ptr + offsets, x + y, mask=mask)\n"
        )
        with open(kernel_file, "w", encoding="utf-8") as f:
            f.write(kernel_src)

        # 3) Generate logs by running the kernel once
        tritonparse.structured_logging.init(
            logs_dir, enable_trace_launch=True, enable_more_tensor_information=True
        )
        try:
            if kernel_dir not in _sys.path:
                _sys.path.insert(0, kernel_dir)
            import importlib as _importlib

            mod = _importlib.import_module("simple_kernel")
            device = torch.device("cuda:0")
            torch.manual_seed(0)
            n = 256
            x = torch.randn((n,), device=device, dtype=torch.float32)
            y = torch.randn((n,), device=device, dtype=torch.float32)
            out = torch.empty_like(x)
            BLOCK_SIZE = 64
            grid = (triton.cdiv(n, BLOCK_SIZE),)
            mod.add_kernel[grid](x, y, out, n, BLOCK_SIZE)
            torch.cuda.synchronize()
        finally:
            tritonparse.structured_logging.clear_logging_config()

        # 4) Find the NDJSON and compute launch event index
        ndjson_files = [
            os.path.join(logs_dir, f)
            for f in os.listdir(logs_dir)
            if f.endswith(".ndjson")
        ]
        assert ndjson_files, f"No ndjson found in {logs_dir}"
        ndjson_path = max(ndjson_files, key=os.path.getmtime)

        from tritonparse.tools.prettify_ndjson import load_ndjson as _load_ndjson

        events = _load_ndjson(_Path(ndjson_path))
        launch_indices = [
            i for i, ev in enumerate(events) if ev.get("event_type") == "launch"
        ]
        assert launch_indices, "No launch event found in ndjson"
        line_index = launch_indices[0]

        # 5) Build reproducer
        from tritonparse.reproducer.orchestrator import reproduce

        reproduce(
            input_path=ndjson_path,
            line_index=line_index,
            out_dir=out_dir,
            template="example",
        )

        # 6) Locate generated script and context under out_dir/add_kernel/
        kernel_out_dir = os.path.join(out_dir, "add_kernel")
        assert os.path.isdir(
            kernel_out_dir
        ), f"Kernel output dir not found: {kernel_out_dir}"
        gen_scripts = [f for f in os.listdir(kernel_out_dir) if f.endswith(".py")]
        gen_jsons = [f for f in os.listdir(kernel_out_dir) if f.endswith(".json")]
        assert gen_scripts, f"No generated script in {kernel_out_dir}"
        assert gen_jsons, f"No generated context json in {kernel_out_dir}"
        script_path = os.path.join(kernel_out_dir, sorted(gen_scripts)[-1])

        # 7) Execute generated script and assert success output
        proc = _subprocess.run(
            [
                _sys.executable,
                script_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Kernel execution finished.", proc.stdout)

        # Cleanup
        if TEST_KEEP_OUTPUT:
            print(f"✓ Preserving temporary directory (TEST_KEEP_OUTPUT=1): {temp_dir}")
        else:
            shutil.rmtree(temp_dir)
            print("✓ Cleaned up temporary directory")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_tensor_blob_manager(self):
        """Test TensorBlobManager functionality with context manager"""

        # Setup fresh cache for this test
        test_cache_dir, prev_cache_dir = self.setup_test_with_fresh_cache()

        # Define a simple kernel that accepts tensor inputs
        @triton.jit
        def tensor_input_kernel(
            input_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(input_ptr + offsets, mask=mask)
            y = x * 2.0
            tl.store(output_ptr + offsets, y, mask=mask)

        def run_kernel(input_tensor):
            n_elements = input_tensor.numel()
            output = torch.empty_like(input_tensor)
            BLOCK_SIZE = 256
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            tensor_input_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE)
            return output

        def collect_blob_files(manager_dir_path):
            """Collect all .bin and .bin.gz files from saved_tensors directory."""
            saved_tensors_dir = os.path.join(manager_dir_path, "saved_tensors")
            bin_files = []
            gz_files = []

            if not os.path.exists(saved_tensors_dir):
                return bin_files, gz_files

            for subdir in os.listdir(saved_tensors_dir):
                subdir_path = os.path.join(saved_tensors_dir, subdir)
                if os.path.isdir(subdir_path):
                    for filename in os.listdir(subdir_path):
                        full_path = os.path.join(subdir_path, filename)
                        if filename.endswith(".bin.gz"):
                            gz_files.append(full_path)
                        elif filename.endswith(".bin"):
                            bin_files.append(full_path)

            return bin_files, gz_files

        def count_all_blobs(manager_dir_path):
            """Count total number of blob files (.bin and .bin.gz)."""
            bin_files, gz_files = collect_blob_files(manager_dir_path)
            return len(bin_files) + len(gz_files)

        # Prepare test data
        torch.manual_seed(0)

        # === Test 1: Mixed tensor sizes with compression threshold ===
        print("\n=== Test 1: Mixed Tensor Sizes with Compression Threshold ===")
        temp_output_dir_1 = tempfile.mkdtemp()

        with tritonparse.context_manager.TritonParseManager(
            enable_trace_launch=True,
            enable_tensor_blob_storage=True,
            out=temp_output_dir_1,
        ) as manager:
            # Test different tensor sizes around the 1MB compression threshold
            test_cases = [
                ((512,), "Tiny 2KB"),  # 2KB < 1MB -> .bin
                ((100 * 1024,), "Medium 400KB"),  # 400KB < 1MB -> .bin
                ((5 * 1024 * 1024,), "Large 20MB"),  # 20MB > 1MB -> .bin.gz
                ((100 * 1024 * 1024,), "Very large 400MB"),  # 400MB > 1MB -> .bin.gz
            ]

            # Create tensors and run kernels
            for size, _ in test_cases:
                x = torch.randn(size, device=self.cuda_device, dtype=torch.float32)
                y = run_kernel(x)
                y.sum()
            torch.cuda.synchronize()

            # Collect and verify blob files
            bin_files, gz_files = collect_blob_files(manager.dir_path)
            assert len(bin_files) + len(gz_files) > 0, "No blob files found"

            print(f"Found {len(bin_files)} .bin files:")
            for f in bin_files:
                print(f"  {f} ({os.path.getsize(f)} bytes)")
            print(f"Found {len(gz_files)} .bin.gz files:")
            for f in gz_files:
                print(f"  {f} ({os.path.getsize(f)} bytes)")

            # Verify correct number of files (2 small uncompressed, 2 large compressed)
            assert (
                len(bin_files) == 4
            ), f"Expected 4 .bin files (2KB, 400KB), got {len(bin_files)}"
            assert (
                len(gz_files) == 4
            ), f"Expected 4 .bin.gz files (20MB, 400MB), got {len(gz_files)}"

            print(
                f"✓ Mixed sizes: {len(bin_files)} uncompressed (.bin), {len(gz_files)} compressed (.bin.gz)"
            )

            # Verify both formats can be loaded
            from tritonparse.tools.load_tensor import load_tensor

            if bin_files:
                loaded = load_tensor(bin_files[0])
                assert loaded is not None, "Failed to load .bin file"
                print("✓ Successfully loaded .bin file")

            if gz_files:
                loaded = load_tensor(gz_files[0])
                assert loaded is not None, "Failed to load .bin.gz file"
                print("✓ Successfully loaded .bin.gz file")

            print("✓ Both formats (.bin and .bin.gz) verified")

        # === Test 2: Deduplication ===
        print("\n=== Test 2: Deduplication ===")
        temp_output_dir_2 = tempfile.mkdtemp()

        with tritonparse.context_manager.TritonParseManager(
            enable_trace_launch=True,
            enable_tensor_blob_storage=True,
            out=temp_output_dir_2,
        ) as manager:
            # Use the same tensor multiple times
            x = torch.randn((512,), device=self.cuda_device, dtype=torch.float32)

            # Run kernel 3 times with same input
            for _ in range(3):
                y = run_kernel(x)
                y.sum()
            torch.cuda.synchronize()

            # Count blob files
            # Note: The system may save both input and output tensors.
            # - Input tensor x: reused 3 times → should deduplicate to 1 blob
            # - Output tensors y: 3 separate allocations → may be 3 blobs (if different) or 1 blob (if identical)
            # Expected: fewer blobs than total tensor references due to deduplication
            blob_count = count_all_blobs(manager.dir_path)
            # With deduplication, we should have significantly fewer blobs than 6 (3 inputs + 3 outputs)
            assert (
                blob_count < 6
            ), f"Deduplication should reduce blob count, got {blob_count} for 3 launches"
            # We expect at least 1 blob (the deduplicated input)
            assert blob_count >= 1, f"Should have at least 1 blob, got {blob_count}"
            print(
                f"✓ Deduplication working: {blob_count} unique blob(s) for 3 launches (< 6 without dedup)"
            )

        # === Test 3: Quota limit ===
        print("\n=== Test 3: Quota Limit ===")
        temp_output_dir_3 = tempfile.mkdtemp()

        # Calculate quota to allow exactly one tensor to be saved
        # A 10000 element float32 tensor = 10000 * 4 bytes = 40KB
        # After torch.save serialization, it will be larger (includes metadata)
        # Compressed size will be smaller for random data (but still substantial)
        # Set quota to ~60KB to allow first tensor but not second
        # Note: Random data doesn't compress as well as zeros
        quota_for_one_tensor = 60 * 1024  # 60KB should fit one serialized tensor

        with tritonparse.context_manager.TritonParseManager(
            enable_trace_launch=True,
            enable_tensor_blob_storage=True,
            tensor_storage_quota=quota_for_one_tensor,
            out=temp_output_dir_3,
        ) as manager:
            # Create first tensor - should be saved successfully
            large_x1 = torch.randn(
                (10000,), device=self.cuda_device, dtype=torch.float32
            )
            y1 = run_kernel(large_x1)
            y1.sum()
            torch.cuda.synchronize()

            # Check that first tensor was saved
            blob_count_after_first = count_all_blobs(manager.dir_path)
            print(f"  Blobs after first kernel launch: {blob_count_after_first}")

            # Create second tensor - should exceed quota and trigger storage disable
            large_x2 = torch.randn(
                (10000,), device=self.cuda_device, dtype=torch.float32
            )
            y2 = run_kernel(large_x2)
            y2.sum()
            torch.cuda.synchronize()

            # Verify quota enforcement
            blob_count_final = count_all_blobs(manager.dir_path)
            print(f"  Blobs after second kernel launch: {blob_count_final}")

            # We expect at least 1 blob was saved (from first launch)
            assert (
                blob_count_after_first >= 1
            ), f"First tensor should be saved, got {blob_count_after_first} blobs"

            # After quota exceeded, no more blobs should be added
            # (blob_count_final should equal blob_count_after_first or be slightly higher
            # if some outputs were saved before quota was hit)
            assert (
                blob_count_final <= blob_count_after_first + 1
            ), f"Quota should prevent saving many more blobs: first={blob_count_after_first}, final={blob_count_final}"

            print(
                f"✓ Quota enforced: {blob_count_after_first} blob(s) saved before quota limit"
            )

        # The test passes if it doesn't crash - storage should be disabled after quota exceeded
        print("✓ Quota limit test passed (storage disabled when quota exceeded)")

        # Reset global variables to default after Test 3 to avoid polluting Test 4
        tritonparse.structured_logging.TRITONPARSE_TENSOR_STORAGE_QUOTA = (
            100 * 1024 * 1024 * 1024
        )  # 100GB default
        tritonparse.structured_logging.TRITONPARSE_SAVE_TENSOR_BLOBS = (
            False  # Reset to default (disabled)
        )

        # === Test 4: Disabled storage ===
        print("\n=== Test 4: Disabled Storage ===")
        temp_output_dir_4 = tempfile.mkdtemp()

        # When storage is explicitly disabled, don't set quota to avoid confusion
        with tritonparse.context_manager.TritonParseManager(
            enable_trace_launch=True,
            enable_tensor_blob_storage=False,  # Explicitly disabled
            out=temp_output_dir_4,
        ) as manager:
            x = torch.randn((512,), device=self.cuda_device, dtype=torch.float32)
            y = run_kernel(x)
            y.sum()
            torch.cuda.synchronize()

            # Verify no saved_tensors directory or it's empty
            total_blobs = count_all_blobs(manager.dir_path)
            assert (
                total_blobs == 0
            ), f"Expected no blobs when storage disabled, found {total_blobs}"
            print("✓ Storage correctly disabled when enable_tensor_blob_storage=False")

        # Clean up all test outputs
        try:
            if TEST_KEEP_OUTPUT:
                print(
                    f"\n✓ Preserving output directories (TEST_KEEP_OUTPUT=1):\n"
                    f"  Test 1: {temp_output_dir_1}\n"
                    f"  Test 2: {temp_output_dir_2}\n"
                    f"  Test 3: {temp_output_dir_3}\n"
                    f"  Test 4: {temp_output_dir_4}"
                )
            else:
                for temp_dir in [
                    temp_output_dir_1,
                    temp_output_dir_2,
                    temp_output_dir_3,
                    temp_output_dir_4,
                ]:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                print("✓ Cleaned up all test output directories")
        except Exception as e:
            print(f"Warning: Failed to clean up output directories: {e}")

        finally:
            # Cleanup test-specific cache
            self.cleanup_test_cache(test_cache_dir, prev_cache_dir)


if __name__ == "__main__":
    unittest.main()
