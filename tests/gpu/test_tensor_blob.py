# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Test TensorBlobManager functionality."""

import os
import shutil
import tempfile
import unittest

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import tritonparse.context_manager
import tritonparse.structured_logging
from tests.test_utils import GPUTestBase
from tritonparse.shared_vars import TEST_KEEP_OUTPUT


class TestTensorBlob(GPUTestBase):
    """Test TensorBlobManager functionality."""

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
            assert len(bin_files) == 4, (
                f"Expected 4 .bin files (2KB, 400KB), got {len(bin_files)}"
            )
            assert len(gz_files) == 4, (
                f"Expected 4 .bin.gz files (20MB, 400MB), got {len(gz_files)}"
            )

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
            assert blob_count < 6, (
                f"Deduplication should reduce blob count, got {blob_count} for 3 launches"
            )
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
            assert blob_count_after_first >= 1, (
                f"First tensor should be saved, got {blob_count_after_first} blobs"
            )

            # After quota exceeded, no more blobs should be added
            # (blob_count_final should equal blob_count_after_first or be slightly higher
            # if some outputs were saved before quota was hit)
            assert blob_count_final <= blob_count_after_first + 1, (
                f"Quota should prevent saving many more blobs: first={blob_count_after_first}, final={blob_count_final}"
            )

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
            assert total_blobs == 0, (
                f"Expected no blobs when storage disabled, found {total_blobs}"
            )
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
