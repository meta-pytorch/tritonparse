# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for tritonparse structured logging functionality.

Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python -m unittest tests.gpu.test_structured_logging -v
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
from typing import Any, Union

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import tritonparse.parse.utils
import tritonparse.structured_logging
from tests.test_utils import GPUTestBase
from triton.compiler import ASTSource, IRSource  # @manual=//triton:triton
from triton.knobs import CompileTimes  # @manual=//triton:triton
from tritonparse.shared_vars import TEST_KEEP_OUTPUT
from tritonparse.structured_logging import convert, extract_python_source_info
from tritonparse.tools.compression import open_compressed_file
from tritonparse.tools.disasm import is_nvdisasm_available


class TestStructuredLogging(GPUTestBase):
    """Tests for structured logging functionality."""

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
        self.assertIn("python_source", trace_data)
        self.assertIn("file_path", trace_data["python_source"])
        triton.knobs.compilation.listener = None

    def test_convert(self):
        """Test convert function with various data types"""
        # Test with primitive types
        self.assertEqual(convert(42), 42)
        self.assertEqual(convert("hello"), "hello")
        self.assertEqual(convert(3.14), 3.14)
        self.assertIsNone(convert(None))
        self.assertTrue(convert(True))

        # Test with a dictionary
        test_dict = {"a": 1, "b": "string", "c": 3.14}
        self.assertEqual(convert(test_dict), test_dict)

        # Test with a list
        test_list = [1, "string", 3.14]
        self.assertEqual(convert(test_list), test_list)

        # Test with a dataclass
        @dataclass
        class TestDataClass:
            x: int
            y: str
            z: float

        test_dataclass = TestDataClass(x=42, y="hello", z=3.14)
        expected_dict = {"x": 42, "y": "hello", "z": 3.14}
        self.assertEqual(convert(test_dataclass), expected_dict)

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

        self.assertEqual(convert(nested_structure), expected_nested)
        print("✓ Convert function tests passed")

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
        self.assertTrue(
            os.path.exists(temp_dir_logs),
            f"Log directory {temp_dir_logs} does not exist.",
        )
        log_files = os.listdir(temp_dir_logs)
        self.assertGreater(
            len(log_files),
            0,
            f"No log files found in {temp_dir_logs}. "
            "Expected log files to be generated during Triton compilation.",
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
                    # Use open_compressed_file which auto-detects compression format
                    with open_compressed_file(log_file_path) as f:
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
                                        assert "Function:" in sass_content, (
                                            "SASS content should contain function declaration"
                                        )
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
                                        ), (
                                            "SASS content should contain GPU assembly instructions"
                                        )

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
        self.assertEqual(
            event_counts["compilation"],
            1,
            f"Expected 1 unique 'compilation' hash, found {event_counts['compilation']}",
        )
        self.assertEqual(
            event_counts["launch"],
            2,
            f"Expected 2 'launch' events, found {event_counts['launch']}",
        )

        # Conditionally verify SASS content based on nvdisasm availability
        if nvdisasm_available:
            self.assertTrue(
                event_counts["sass_found"],
                "SASS content was not found in compilation events",
            )
            print("✓ Successfully verified SASS extraction functionality")
        else:
            print("⚠️  SASS verification skipped: nvdisasm not available")

        print(
            "✓ Verified correct event type counts: 1 unique compilation hash, 2 launch events"
        )

        # Test parsing functionality
        tritonparse.parse.utils.unified_parse(
            source=temp_dir_logs, out=temp_dir_parsed, overwrite=True
        )
        try:
            # Verify parsing output
            parsed_files = os.listdir(temp_dir_parsed)
            self.assertGreater(
                len(parsed_files), 0, "No files found in parsed output directory"
            )

            # Verify that SASS is preserved in parsed output
            ndjson_gz_files = [f for f in parsed_files if f.endswith(".ndjson.gz")]
            self.assertGreater(
                len(ndjson_gz_files),
                0,
                "No .ndjson.gz files found in parsed output",
            )

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
                self.assertTrue(
                    sass_found_in_parsed,
                    "SASS content was not preserved in parsed output",
                )
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


if __name__ == "__main__":
    unittest.main()
