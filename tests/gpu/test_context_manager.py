# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for tritonparse context manager functionality.

Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python -m unittest tests.gpu.test_context_manager -v
```
"""

import gzip
import json
import os
import shutil
import tempfile
import unittest

import torch
import torch._inductor.config as inductor_config
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import tritonparse.context_manager
from tests.test_utils import clear_all_caches, GPUTestBase
from triton import knobs  # @manual=//triton:triton
from tritonparse.shared_vars import TEST_KEEP_OUTPUT


class TestContextManager(GPUTestBase):
    """Tests for TritonParseManager context manager."""

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
            self.assertTrue(
                os.path.exists(manager.dir_path), "Temporary directory should exist"
            )
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
            self.assertGreater(len(log_files), 0, "Log files should be generated")
            print(f"Generated {len(log_files)} log file(s)")

        # After exiting context manager, verify behavior
        # Verify parsed output exists
        self.assertTrue(
            os.path.exists(temp_output_dir_split_true),
            "Parsed output directory should exist",
        )
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
            self.assertTrue(
                os.path.exists(manager.dir_path), "Temporary directory should exist"
            )
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
            self.assertGreater(len(log_files), 0, "Log files should be generated")
            print(f"Generated {len(log_files)} log file(s)")

        # After exiting context manager, verify behavior
        # Verify parsed output exists
        self.assertTrue(
            os.path.exists(temp_output_dir_split_false),
            "Parsed output directory should exist",
        )
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
        self.assertGreater(
            len(ndjson_gz_files_split_false),
            0,
            "No .ndjson.gz files found in split=False parsed output",
        )

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
                                self.assertIn(
                                    compilation_name,
                                    expected_compilation_names,
                                    f"Unexpected compilation name: '{compilation_name}'. "
                                    f"Expected one of: {expected_compilation_names}",
                                )
                    except json.JSONDecodeError:
                        continue

        print(
            f"Compilation events found (split=False): {compilation_count_split_false}"
        )
        print(f"Compilation names found: {compilation_names_found}")

        self.assertGreater(
            compilation_count_split_false,
            0,
            "Expected at least 1 compilation event in split=False output",
        )

        # Verify all compilation names are from the expected set
        unique_names_found = set(compilation_names_found)
        self.assertTrue(
            unique_names_found.issubset(expected_compilation_names),
            f"Found unexpected compilation names: {unique_names_found - expected_compilation_names}. "
            f"Expected only: {expected_compilation_names}",
        )
        print(f"✓ All compilation names are valid: {unique_names_found}")

        # Verify the key difference: split=False should have one fewer file
        self.assertEqual(
            num_files_split_false,
            num_files_split_true - 1,
            f"split=False should have one fewer file (expected {num_files_split_true - 1}, got {num_files_split_false})",
        )
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


if __name__ == "__main__":
    unittest.main()
