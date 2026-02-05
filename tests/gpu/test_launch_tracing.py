# (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
Unit tests for tritonparse launch tracing integration with torch.profiler.

Test cases:
1. No additional flag - only compilation events collected
2. Enable trace launch - all launch events collected
3. Profile-aware launch tracing - only launch events during RECORD phase (transparent)
4. Both flags set - enable_trace_launch takes priority (traces all launches)
"""

import glob
import json
import os
import shutil
import tempfile
import unittest
from typing import Optional, Tuple

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
from torch.profiler import ProfilerActivity
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


def run_training_loop(
    block_size: int,
    num_steps: int,
    wait: Optional[int] = None,
    warmup: Optional[int] = None,
    active: Optional[int] = None,
    repeat: int = 1,
) -> int:
    """
    Run a training loop with optional torch.profiler integration.

    Args:
        block_size: Block size for the triton kernel (use unique values per test)
        num_steps: Total number of kernel launches to execute
        wait: Profiler schedule wait steps (None to disable profiler)
        warmup: Profiler schedule warmup steps
        active: Profiler schedule active/record steps
        repeat: Profiler schedule repeat count

    Returns:
        Total number of steps executed
    """
    use_profiler = wait is not None and warmup is not None and active is not None

    if use_profiler:
        schedule = torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )

        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
        ) as prof:
            for _ in range(num_steps):
                run_kernel(block_size=block_size)
                prof.step()
    else:
        for _ in range(num_steps):
            run_kernel(block_size=block_size)

    return num_steps


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

        num_launches = 5
        run_training_loop(block_size=256, num_steps=num_launches)

        compilation_count, launch_count = count_events(self.trace_folder)

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

        num_launches = 5
        run_training_loop(block_size=512, num_steps=num_launches)

        compilation_count, launch_count = count_events(self.trace_folder)

        self.assertGreaterEqual(
            compilation_count, 1, "Should have at least 1 compilation event"
        )
        self.assertEqual(
            launch_count,
            num_launches,
            f"Expected {num_launches} launch events (ALL launches)",
        )

    def test_case3_profile_aware_launch_tracing(self) -> None:
        """
        Case 3: Profile-aware launch tracing - only RECORD phase events.

        With init(enable_trace_launch_within_profiling=True), torch.profiler.schedule
        is automatically patched to enable launch tracing during RECORD phase.
        This is transparent to users - no code changes needed.
        """
        init(self.trace_folder, enable_trace_launch_within_profiling=True)

        wait, warmup, active, repeat = 2, 1, 5, 1
        total_steps = wait + warmup + active

        run_training_loop(
            block_size=128,
            num_steps=total_steps,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )

        compilation_count, launch_count = count_events(self.trace_folder)

        self.assertGreaterEqual(
            compilation_count, 1, "Should have at least 1 compilation event"
        )
        self.assertEqual(
            launch_count, active, f"Expected {active} launch events (RECORD phase only)"
        )

    def test_case4_both_flags_trace_launch_takes_priority(self) -> None:
        """
        Case 4: Both flags set - enable_trace_launch takes priority.

        When both enable_trace_launch=True and enable_trace_launch_within_profiling=True
        are set, enable_trace_launch takes priority and ALL launches are traced
        (not just during RECORD phase).
        """
        init(
            self.trace_folder,
            enable_trace_launch=True,
            enable_trace_launch_within_profiling=True,
        )

        wait, warmup, active, repeat = 2, 1, 3, 1
        total_steps = wait + warmup + active

        run_training_loop(
            block_size=64,
            num_steps=total_steps,
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
        )

        compilation_count, launch_count = count_events(self.trace_folder)

        self.assertGreaterEqual(
            compilation_count, 1, "Should have at least 1 compilation event"
        )
        self.assertEqual(
            launch_count,
            total_steps,
            f"Expected {total_steps} launch events (ALL launches, enable_trace_launch priority)",
        )


if __name__ == "__main__":
    unittest.main()
