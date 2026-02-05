# (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
Unit tests for tritonparse launch tracing integration with torch.profiler.

Test cases:
1. No additional flag - only compilation events collected
2. Enable trace launch - all launch events collected
3. Profile-aware launch tracing - only launch events during RECORD phase (transparent)
"""

import glob
import json
import os
import shutil
import tempfile
import unittest
from typing import Tuple

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
from tritonparse.structured_logging import init
from tritonparse.tools.compression import iter_lines


@triton.jit
def add_kernel(
    x_ptr: tl.pointer_type,  # pyre-ignore[11]
    y_ptr: tl.pointer_type,  # pyre-ignore[11]
    output_ptr: tl.pointer_type,  # pyre-ignore[11]
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,  # pyre-ignore[11]
) -> None:
    """Simple add kernel for testing."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def run_kernel(block_size: int = 256) -> None:
    """Run a simple triton kernel."""
    size = 1024
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)  # noqa: E731
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()


def count_events(trace_folder: str) -> Tuple[int, int]:
    """
    Count compilation and launch events in trace folder.

    Returns:
        Tuple of (compilation_count, launch_count)
    """
    compilation_count = 0
    launch_count = 0

    # Get all ndjson files, avoiding double-counting
    # Use *.bin.ndjson pattern only since raw logs are compressed
    for filepath in glob.glob(os.path.join(trace_folder, "*.bin.ndjson")):
        try:
            for line in iter_lines(filepath):
                try:
                    event = json.loads(line)
                    event_type = event.get("event_type")
                    if event_type == "compilation":
                        compilation_count += 1
                    elif event_type == "launch":
                        launch_count += 1
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue

    return compilation_count, launch_count


class TestTritonparseLaunchTracing(unittest.TestCase):
    """Test tritonparse launch tracing modes."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create fresh trace folder for each test
        self.trace_folder = tempfile.mkdtemp(prefix="tritonparse_test_")

        # Create fresh Triton cache folder to avoid kernel caching between tests
        self.triton_cache_folder = tempfile.mkdtemp(prefix="triton_cache_")
        self._old_triton_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = self.triton_cache_folder

        # Clear any existing tritonparse state
        from tritonparse import structured_logging

        structured_logging.clear_logging_config()

        # Also reset the global flags
        structured_logging.TRITON_TRACE_LAUNCH = False
        structured_logging.TRITON_TRACE_LAUNCH_WITHIN_PROFILING = False
        structured_logging._trace_launch_enabled = False

    def tearDown(self) -> None:
        # Clean up tritonparse state
        from tritonparse import structured_logging

        structured_logging.clear_logging_config()

        # Restore original Triton cache dir
        if self._old_triton_cache is not None:
            os.environ["TRITON_CACHE_DIR"] = self._old_triton_cache
        else:
            os.environ.pop("TRITON_CACHE_DIR", None)

        # Remove temporary folders
        if os.path.exists(self.trace_folder):
            shutil.rmtree(self.trace_folder)
        if os.path.exists(self.triton_cache_folder):
            shutil.rmtree(self.triton_cache_folder)

    def test_case1_no_flag_only_compilation_events(self) -> None:
        """
        Case 1: No additional flag - only compilation events collected.

        With just init(), only compilation events should be logged.
        Launch events should NOT be collected.
        """

        init(self.trace_folder)

        # Run kernel multiple times (use block_size=256 for this test)
        num_launches = 5
        for _ in range(num_launches):
            run_kernel(block_size=256)

        compilation_count, launch_count = count_events(self.trace_folder)

        # Should have at least 1 compilation event
        self.assertGreaterEqual(
            compilation_count, 1, "Should have at least 1 compilation event"
        )
        self.assertEqual(launch_count, 0, "Should NOT have launch events without flag")

    def test_case2_trace_launch_all_events(self) -> None:
        """
        Case 2: Enable trace launch - all launch events collected.

        With init(enable_trace_launch=True), ALL launch events should be logged.
        """

        init(self.trace_folder, enable_trace_launch=True)

        # Run kernel multiple times (use block_size=512 for this test)
        num_launches = 5
        for _ in range(num_launches):
            run_kernel(block_size=512)

        compilation_count, launch_count = count_events(self.trace_folder)

        # Should have at least 1 compilation event
        self.assertGreaterEqual(
            compilation_count, 1, "Should have at least 1 compilation event"
        )
        # Should have launch events matching number of launches
        self.assertEqual(
            launch_count,
            num_launches,
            f"Expected {num_launches} launch events, got {launch_count}",
        )

    def test_case3_profile_aware_launch_tracing(self) -> None:
        """
        Case 3: Profile-aware launch tracing - only RECORD phase events.

        With init(enable_trace_launch_within_profiling=True), torch.profiler.schedule
        is automatically patched to enable launch tracing during RECORD phase.
        This is transparent to users - no code changes needed.
        """
        from torch.profiler import ProfilerActivity

        # Initialize tritonparse with enable_trace_launch_within_profiling=True
        # This patches torch.profiler.schedule automatically
        init(self.trace_folder, enable_trace_launch_within_profiling=True)

        num_none_launches = 2
        num_warmup_launches = 1
        num_record_launches = 5
        num_extra_launches = 1
        total_steps = num_none_launches + num_warmup_launches + num_record_launches

        # User just uses regular torch.profiler.schedule - no wrapping needed
        schedule = torch.profiler.schedule(
            wait=num_none_launches,
            warmup=num_warmup_launches,
            active=num_record_launches,
            repeat=num_extra_launches,
        )

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
        ) as prof:
            for _ in range(total_steps):
                run_kernel(block_size=128)  # Use block_size=128 for this test
                prof.step()

        compilation_count, launch_count = count_events(self.trace_folder)

        # Should have at least 1 compilation event
        self.assertGreaterEqual(
            compilation_count, 1, "Should have at least 1 compilation event"
        )
        # Should only have launch events from RECORD phase
        self.assertEqual(
            launch_count,
            num_record_launches,
            f"Expected {num_record_launches} launch events (RECORD phase only), "
            f"got {launch_count}. "
            f"NONE={num_none_launches}, WARMUP={num_warmup_launches}, "
            f"RECORD={num_record_launches}, EXTRA={num_extra_launches}",
        )


if __name__ == "__main__":
    unittest.main()
