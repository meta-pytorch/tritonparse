"""
Comprehensive tests for tritonparse using unittest.
Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TRITONPARSE_DEBUG=1 python -m unittest tests.test_tritonparse -v
```
"""

import gzip
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Union

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

import tritonparse.structured_logging
import tritonparse.utils

# @manual=//triton:triton
from triton.compiler import ASTSource, IRSource

# @manual=//triton:triton
from triton.knobs import CompileTimes
from tritonparse.structured_logging import convert, extract_python_source_info

HAS_TRITON_KERNELS = importlib.util.find_spec("triton_kernels") is not None


def should_keep_output() -> bool:
    """Return True if test outputs (e.g., temp dirs) should be preserved.

    Controlled by environment variable TEST_KEEP_OUTPUT=1.
    """
    return os.environ.get("TEST_KEEP_OUTPUT") == "1"


class TestTritonparseCPU(unittest.TestCase):
    """CPU-only tests (no CUDA required)"""

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


class TestTritonparseCUDA(unittest.TestCase):
    """CUDA tests (require GPU)"""

    def setUp(self):
        """Set up triton hooks and compilation settings"""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.cuda_device = torch.device("cuda:0")

        # Save original settings
        self.prev_listener = triton.knobs.compilation.listener
        self.prev_always_compile = triton.knobs.compilation.always_compile
        self.prev_jit_post_compile_hook = triton.knobs.runtime.jit_post_compile_hook
        self.prev_launch_enter_hook = triton.knobs.runtime.launch_enter_hook

        # Set up new settings
        triton.knobs.compilation.always_compile = True

    def tearDown(self):
        """Restore original triton settings"""
        # Always restore original settings, even if test fails
        triton.knobs.compilation.always_compile = self.prev_always_compile
        triton.knobs.compilation.listener = self.prev_listener
        triton.knobs.runtime.jit_post_compile_hook = self.prev_jit_post_compile_hook
        triton.knobs.runtime.launch_enter_hook = self.prev_launch_enter_hook

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
        """Test unified_parse functionality"""

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

        # Initialize logging
        tritonparse.structured_logging.init(temp_dir_logs, enable_trace_launch=True)

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
            """Count 'launch' and unique 'compilation' events in all log files"""
            event_counts = {"launch": 0}
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
                            except (json.JSONDecodeError, KeyError, TypeError) as e:
                                print(f"  Line {line_num}: Error processing line - {e}")

            # Add the count of unique compilation hashes to the event_counts
            event_counts["compilation"] = len(compilation_hashes)
            print(
                f"Event type counts: {event_counts} (unique compilation hashes: {len(compilation_hashes)})"
            )
            return event_counts

        # Verify event counts
        event_counts = check_event_type_counts_in_logs(temp_dir_logs)
        assert (
            event_counts["compilation"] == 1
        ), f"Expected 1 unique 'compilation' hash, found {event_counts['compilation']}"
        assert (
            event_counts["launch"] == 2
        ), f"Expected 2 'launch' events, found {event_counts['launch']}"
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
        finally:
            # Clean up
            if should_keep_output():
                print(
                    f"✓ Preserving temporary directory (TEST_KEEP_OUTPUT=1): {temp_dir}"
                )
            else:
                shutil.rmtree(temp_dir)
                print("✓ Cleaned up temporary directory")
            tritonparse.structured_logging.clear_logging_config()

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
            if should_keep_output():
                print(
                    f"✓ Preserving temporary directory (TEST_KEEP_OUTPUT=1): {temp_dir}"
                )
            else:
                shutil.rmtree(temp_dir)
                print("✓ Cleaned up temporary directory")
            tritonparse.structured_logging.clear_logging_config()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    @unittest.skipUnless(HAS_TRITON_KERNELS, "triton_kernels not installed")
    def test_triton_kernels_Tensor(self):
        from triton_kernels.topk_details._topk_forward import _topk_forward
        from tritonparse.reproducer import utils as reproducer_utils

        input_json_path = os.path.join(
            os.path.dirname(__file__),
            "example_output",
            "repro",
            "repro_context_20250816192455.json",
        )
        grid, args_dict = reproducer_utils.create_args_from_json(input_json_path)
        print("Generated kernel arguments dictionary:")
        for name, arg in args_dict.items():
            print(f"  {name}: {arg}")
        print(f"Grid: {grid}")

        # Set up test environment
        temp_dir = tempfile.mkdtemp()
        temp_dir_logs = os.path.join(temp_dir, "logs")
        temp_dir_parsed = os.path.join(temp_dir, "parsed_output")
        os.makedirs(temp_dir_logs, exist_ok=True)
        os.makedirs(temp_dir_parsed, exist_ok=True)
        print(f"Temporary directory: {temp_dir}")

        # Initialize logging
        tritonparse.structured_logging.init(temp_dir_logs, enable_trace_launch=True)
        try:
            _topk_forward[tuple(grid)](
                args_dict["X"],
                args_dict["stride_xm"],
                args_dict["Yv"],
                args_dict["Yi"],
                args_dict["stride_ym"],
                args_dict["USE_PROVIDED_INDX"],
                args_dict["Bits"],
                args_dict["stride_rm"],
                args_dict["stride_rn"],
                args_dict["n_rows"],
                args_dict["n_expts_tot"],
                args_dict["S"],
                args_dict["BLOCK_S"],
                args_dict["s_blocks"],
                args_dict["APPLY_SOFTMAX"],
                args_dict["BLOCK_M"],
                args_dict["N_EXPTS_PAD"],
                args_dict["N_EXPTS_ACT"],
                args_dict["BLOCK_N"],
            )
            torch.cuda.synchronize()

            # Read the only ndjson file under temp_dir_logs/logs/
            ndjson_files = [
                f for f in os.listdir(temp_dir_logs) if f.endswith(".ndjson")
            ]
            assert (
                len(ndjson_files) == 1
            ), f"Expected exactly one ndjson in {temp_dir_logs}, found {len(ndjson_files)}: {ndjson_files}"

            ndjson_path = os.path.join(temp_dir_logs, ndjson_files[0])
            launches = []
            with open(ndjson_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        event = json.loads(line.strip())
                        if event.get("event_type") == "launch":
                            launches.append(event)
                    except json.JSONDecodeError as e:
                        self.fail(f"JSON decode error at line {line_num}: {e}")

            assert (
                len(launches) == 1
            ), f"Expected exactly one launch event, found {len(launches)}"

            launch = launches[0]
            extracted_args = launch.get("extracted_args", {})
            assert "X" in extracted_args, "Missing 'X' in extracted_args"
            assert (
                extracted_args["X"].get("type") == "triton_kernels.tensor.Tensor"
            ), f"X.type is {extracted_args['X'].get('type')}"

        finally:
            # Clean up
            if should_keep_output():
                print(
                    f"✓ Preserving temporary directory (TEST_KEEP_OUTPUT=1): {temp_dir}"
                )
            else:
                shutil.rmtree(temp_dir)
                print("✓ Cleaned up temporary directory")
            tritonparse.structured_logging.clear_logging_config()

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
        tritonparse.structured_logging.init(logs_dir, enable_trace_launch=True)
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

        events = _load_ndjson(_Path(ndjson_path), save_irs=True)
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
        if should_keep_output():
            print(f"✓ Preserving temporary directory (TEST_KEEP_OUTPUT=1): {temp_dir}")
        else:
            shutil.rmtree(temp_dir)
            print("✓ Cleaned up temporary directory")


if __name__ == "__main__":
    unittest.main()
