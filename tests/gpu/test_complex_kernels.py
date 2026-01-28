# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for complex Triton kernels including autotuning and launch_diff functionality.

Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python -m unittest tests.gpu.test_complex_kernels -v
```
"""

import gzip
import json
import os
import shutil
import tempfile
import unittest

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import tritonparse.context_manager
from tests.test_utils import GPUTestBase
from tritonparse.shared_vars import TEST_KEEP_OUTPUT


class TestComplexKernels(GPUTestBase):
    """Tests for complex Triton kernels with autotuning and multiple launches."""

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
        parsed_output_path = tempfile.mkdtemp()
        print(f"Parsed output directory: {parsed_output_path}")

        # Use TritonParseManager context manager for proper logging setup
        with tritonparse.context_manager.TritonParseManager(
            enable_trace_launch=True,
            out=parsed_output_path,
        ):
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

            torch.cuda.synchronize()

        # After exiting context manager, parsed output should be available
        # Verify that parsed output was generated
        parsed_files = os.listdir(parsed_output_path)
        assert len(parsed_files) > 0, f"No parsed files found in {parsed_output_path}"
        print(f"✓ Generated {len(parsed_files)} parsed files")

        # Verify we have both json and ndjson.gz files
        json_files = [f for f in parsed_files if f.endswith(".json")]
        ndjson_gz_files = [f for f in parsed_files if f.endswith(".ndjson.gz")]

        assert len(json_files) > 0, f"No .json files found in {parsed_output_path}"
        assert len(ndjson_gz_files) > 0, (
            f"No .ndjson.gz files found in {parsed_output_path}"
        )
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
            assert launch_diff_count == 5, (
                f"Expected 5 launch_diff events, found {launch_diff_count}"
            )
            print("✓ Verified 5 launch_diff events in parsed output")

        # Clean up
        if TEST_KEEP_OUTPUT:
            print(
                f"✓ Preserving parsed output directory (TEST_KEEP_OUTPUT=1): {parsed_output_path}"
            )
        else:
            shutil.rmtree(parsed_output_path)
            print("✓ Cleaned up parsed output directory")


if __name__ == "__main__":
    unittest.main()
