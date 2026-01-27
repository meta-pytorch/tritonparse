"""
Tests for tritonparse autotune analysis functionality.

Test Plan:
```
buck test //pytorch/tritonparse/tests/gpu:test_autotune
```
"""

import gzip
import json
import os
import shutil

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import tritonparse.parse.utils
import tritonparse.structured_logging
from tests.test_utils import GPUTestBase, setup_temp_log_dir
from triton import knobs  # @manual=//triton:triton
from tritonparse.shared_vars import TEST_KEEP_OUTPUT
from tritonparse.tools.compression import open_compressed_file


class TestAutotuneAnalysis(GPUTestBase):
    """Tests for autotune analysis functionality."""

    def test_autotune_two_simple_kernels(self):
        """
        Tests autotuning for two simple, distinct Triton kernels.
        Verifies that tritonparse correctly captures all compilation events from autotuning
        and subsequent launch events, as well as autotune_analysis events.
        """

        # Kernel 1: Vector Addition
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 128, "ADD_K": 0}, num_warps=4),
                triton.Config({"BLOCK_SIZE": 1024, "ADD_K": 1}, num_warps=8),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel(
            x_ptr,
            y_ptr,
            output_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            ADD_K: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y + ADD_K
            tl.store(output_ptr + offsets, output, mask=mask)

        # Kernel 2: Vector Scaling
        @triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
                triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def scale_kernel(
            x_ptr,
            output_ptr,
            scale_factor,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            output = x * scale_factor
            tl.store(output_ptr + offsets, output, mask=mask)

        # Setup temporary directories for logging
        temp_dir, log_path, parsed_output_path = setup_temp_log_dir()
        print(f"Temporary directory for autotune test: {temp_dir}")

        # Initialize logging
        tritonparse.structured_logging.init(log_path, enable_trace_launch=True)

        try:
            # Disable always_compile to allow autotune caching to work properly.
            # This allows the second add_kernel call to use cached winner instead of re-autotuning.
            original_always_compile = knobs.compilation.always_compile
            knobs.compilation.always_compile = False

            torch.manual_seed(0)
            size = 2048
            x = torch.randn(size, device=self.cuda_device, dtype=torch.float32)
            y = torch.randn(size, device=self.cuda_device, dtype=torch.float32)

            # --- Run Add Kernel ---
            # This first call will trigger autotuning, running both configs.
            print("--- Testing Add Kernel (autotuning) ---")
            output_add = torch.empty_like(x)

            def add_grid(META):
                return (triton.cdiv(size, META["BLOCK_SIZE"]),)

            add_kernel[add_grid](x, y, output_add, size)
            torch.cuda.synchronize()
            print("Add kernel autotuning complete.")

            # Run again to generate a second launch event for the best config.
            add_kernel[add_grid](x, y, output_add, size)
            torch.cuda.synchronize()
            print("Add kernel second launch complete.")

            # --- Run Add Kernel again with new shape to trigger autotuning ---
            print("\n--- Testing Add Kernel (autotuning with new shape) ---")
            new_size = 1024
            x2 = torch.randn(new_size, device=self.cuda_device, dtype=torch.float32)
            y2 = torch.randn(new_size, device=self.cuda_device, dtype=torch.float32)
            output_add2 = torch.empty_like(x2)

            def add_grid_new(META):
                return (triton.cdiv(new_size, META["BLOCK_SIZE"]),)

            add_kernel[add_grid_new](x2, y2, output_add2, new_size)
            torch.cuda.synchronize()
            print("Add kernel autotuning on new shape complete.")

            # --- Run Scale Kernel ---
            # This will also trigger autotuning for its configs.
            print("\n--- Testing Scale Kernel (autotuning) ---")
            output_scale = torch.empty_like(x)

            def scale_grid(META):
                return (triton.cdiv(size, META["BLOCK_SIZE"]),)

            scale_kernel[scale_grid](x, output_scale, 2.0, size)
            torch.cuda.synchronize()
            print("Scale kernel autotuning complete.")

            # Parse the logs
            tritonparse.parse.utils.unified_parse(
                source=log_path, out=parsed_output_path, overwrite=True
            )

            # --- Verification of raw logs ---
            compilation_hashes = set()
            launch_count = 0
            for log_file in os.listdir(log_path):
                if log_file.endswith(".ndjson"):
                    with open_compressed_file(os.path.join(log_path, log_file)) as f:
                        for line in f:
                            event = json.loads(line)
                            if event["event_type"] == "compilation":
                                compilation_hashes.add(
                                    event["payload"]["metadata"]["hash"]
                                )
                            elif event["event_type"] == "launch":
                                launch_count += 1

            print(f"Found {len(compilation_hashes)} unique compilation hashes.")
            print(f"Found {launch_count} launch events.")
            self.assertEqual(
                len(compilation_hashes),
                4,
                "Expected 4 unique compilation events from autotuning.",
            )

            # --- Verification of parsed output ---
            launch_diff_count = 0
            autotune_analysis_count = 0
            session_ids = set()
            autotune_launch_types = set()

            for parsed_file in os.listdir(parsed_output_path):
                if parsed_file.endswith(".ndjson.gz"):
                    with gzip.open(
                        os.path.join(parsed_output_path, parsed_file), "rt"
                    ) as f:
                        for line in f:
                            event = json.loads(line)
                            event_type = event.get("event_type")

                            if event_type == "launch":
                                # Collect autotune_launch_type from parsed launch events
                                launch_type = event.get("autotune_launch_type")
                                if launch_type:
                                    autotune_launch_types.add(launch_type)

                            elif event_type == "launch_diff":
                                launch_diff_count += 1

                            elif event_type == "autotune_analysis":
                                autotune_analysis_count += 1
                                session_id = event.get("session_id")
                                if session_id:
                                    session_ids.add(session_id)
                                # Verify autotune_analysis event has expected fields
                                self.assertIn(
                                    "winner_compilation_hash",
                                    event,
                                    "autotune_analysis event should have winner_compilation_hash",
                                )
                                self.assertIn(
                                    "compilation_analysis",
                                    event,
                                    "autotune_analysis event should have compilation_analysis",
                                )
                                # Verify possible_groups field for kernel association
                                self.assertIn(
                                    "possible_groups",
                                    event,
                                    "autotune_analysis event should have possible_groups",
                                )
                                possible_groups = event["possible_groups"]
                                self.assertIsInstance(
                                    possible_groups,
                                    list,
                                    "possible_groups should be a list",
                                )
                                # Each group should be a list of compilation hashes
                                for group in possible_groups:
                                    self.assertIsInstance(
                                        group,
                                        list,
                                        "Each group in possible_groups should be a list",
                                    )
                                    self.assertGreater(
                                        len(group),
                                        0,
                                        "Each group should have at least one hash",
                                    )
                                # Verify launch_occurrence_ids structure
                                self.assertIn(
                                    "launch_occurrence_ids",
                                    event,
                                    "autotune_analysis event should have launch_occurrence_ids",
                                )
                                launch_ids = event["launch_occurrence_ids"]
                                self.assertIn(
                                    "benchmark",
                                    launch_ids,
                                    "launch_occurrence_ids should have benchmark",
                                )
                                self.assertIn(
                                    "winner",
                                    launch_ids,
                                    "launch_occurrence_ids should have winner",
                                )
                                # Verify launch_ranges structure
                                self.assertIn(
                                    "launch_ranges",
                                    event,
                                    "autotune_analysis event should have launch_ranges",
                                )
                                # Verify session_stack exists
                                self.assertIn(
                                    "session_stack",
                                    event,
                                    "autotune_analysis event should have session_stack",
                                )

            print(f"Found {launch_diff_count} launch_diff events.")
            print(f"Found {autotune_analysis_count} autotune_analysis events.")
            print(f"Found {len(session_ids)} unique session IDs.")
            print(f"Found autotune_launch_types: {autotune_launch_types}")

            self.assertEqual(launch_diff_count, 4, "Expected 4 launch_diff events.")

            # Verify autotune_launch_type values
            # Expected types: benchmark (during autotuning), winner (first use),
            # cached_winner (subsequent uses of cached winner)
            self.assertIn(
                "benchmark",
                autotune_launch_types,
                "Should have benchmark launch type",
            )
            self.assertIn(
                "winner",
                autotune_launch_types,
                "Should have winner launch type",
            )
            self.assertIn(
                "cached_winner",
                autotune_launch_types,
                "Should have cached_winner launch type (from second add_kernel call)",
            )

            # Expected: 3 autotune sessions
            # 1. add_kernel with size=2048
            # 2. add_kernel with size=1024 (new shape)
            # 3. scale_kernel with size=2048
            self.assertEqual(
                autotune_analysis_count, 3, "Expected 3 autotune_analysis events."
            )
            self.assertEqual(len(session_ids), 3, "Expected 3 unique session IDs.")

            print("✓ Verification successful")

        finally:
            # Restore always_compile setting
            knobs.compilation.always_compile = original_always_compile
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
    import unittest

    unittest.main()
