# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Shared utility functions and base classes for tritonparse tests."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from triton import knobs  # @manual=//triton:triton
from tritonparse import structured_logging
from tritonparse.shared_vars import is_fbcode, TEST_KEEP_OUTPUT


# =============================================================================
# Skip decorators (unittest style)
# =============================================================================

requires_gpu = unittest.skipUnless(torch.cuda.is_available(), "GPU not available")

skip_in_fbcode = unittest.skipIf(is_fbcode(), "Skip in internal FB environment")


# =============================================================================
# Helper functions
# =============================================================================


def get_test_ndjson_file():
    """Get the test NDJSON file path."""
    gz_file = (
        Path(__file__).parent
        / "example_output/parsed_output_complex/dedicated_log_triton_trace_findhao__mapped.ndjson.gz"
    )
    assert gz_file.exists(), f"Test file not found: {gz_file}"
    return gz_file


def get_sass_test_file(filename: str) -> Path:
    """Get path to SASS test data file.

    Args:
        filename: Name of the file in sass_test_data directory (e.g., "test_kernel.sass")

    Returns:
        Path to the test file
    """
    test_file = Path(__file__).parent / "example_output/sass_test_data" / filename
    assert test_file.exists(), f"Test file not found: {test_file}"
    return test_file


def setup_temp_reproduce_dir():
    """Setup temporary directory for reproduce tests."""
    temp_dir = tempfile.mkdtemp()
    out_dir = os.path.join(temp_dir, "repro_output")
    os.makedirs(out_dir, exist_ok=True)
    return temp_dir, out_dir


def cleanup_temp_dir(temp_dir):
    """Cleanup temporary directory."""
    if not TEST_KEEP_OUTPUT:
        shutil.rmtree(temp_dir, ignore_errors=True)


def setup_temp_log_dir():
    """Setup temporary directory for logging tests."""
    temp_dir = tempfile.mkdtemp()
    logs_dir = os.path.join(temp_dir, "logs")
    parsed_dir = os.path.join(temp_dir, "parsed_output")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(parsed_dir, exist_ok=True)
    return temp_dir, logs_dir, parsed_dir


# =============================================================================
# Triton cache management functions
# =============================================================================


def create_fresh_triton_cache():
    """Create a fresh Triton cache directory and return cache management context."""
    cache_dir = tempfile.mkdtemp(prefix="triton_cache_")
    return cache_dir, knobs.cache.scope()


def setup_fresh_triton_environment(cache_dir):
    """Setup fresh Triton environment with isolated cache."""
    original_cache_dir = getattr(knobs.cache, "dir", None)
    knobs.cache.dir = cache_dir

    original_always_compile = knobs.compilation.always_compile
    knobs.compilation.always_compile = True

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
    """Restore original Triton environment settings."""
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

    Returns:
        tuple: (new_cache_dir, original_cache_dir) for cleanup purposes
    """
    print("\n=== Clearing all caches ===")

    torch.compiler.reset()
    torch._dynamo.reset()
    print("✓ Reset torch compiler, dynamo, and inductor state")

    kernels_cleared = 0
    for kernel in kernels:
        if hasattr(kernel, "device_caches"):
            for device_id in kernel.device_caches:
                device_cache_tuple = kernel.device_caches[device_id]
                for cache_obj in device_cache_tuple:
                    if hasattr(cache_obj, "clear"):
                        cache_obj.clear()
            kernel.hash = None
            kernels_cleared += 1

    if kernels_cleared > 0:
        print(
            f"✓ Cleared device caches and reset hashes for {kernels_cleared} kernel(s)"
        )
    else:
        print("✓ No kernels provided for device cache clearing")

    new_cache_dir = tempfile.mkdtemp(prefix="triton_fresh_cache_")
    original_cache_dir = knobs.cache.dir
    knobs.cache.dir = new_cache_dir
    print(f"✓ Created fresh Triton cache directory: {new_cache_dir}")

    return new_cache_dir, original_cache_dir


# =============================================================================
# Base test classes
# =============================================================================


class GPUTestBase(unittest.TestCase):
    """Base class for GPU tests with common setup/teardown."""

    def setUp(self):
        """Set up triton hooks and compilation settings."""
        # Clear any global logging state from previous tests
        structured_logging.clear_logging_config()

        if not torch.cuda.is_available():
            self.skipTest("GPU not available")

        self.cuda_device = torch.device("cuda:0")

        # Set up fresh Triton cache environment
        self.triton_cache_dir, self.cache_scope = create_fresh_triton_cache()
        self.cache_scope.__enter__()
        self.original_triton_settings = setup_fresh_triton_environment(
            self.triton_cache_dir
        )

        # Save original settings for restoration
        self.prev_listener = knobs.compilation.listener
        self.prev_always_compile = knobs.compilation.always_compile
        self.prev_jit_post_compile_hook = knobs.runtime.jit_post_compile_hook
        self.prev_launch_enter_hook = knobs.runtime.launch_enter_hook

    def tearDown(self):
        """Restore original triton settings."""
        try:
            restore_triton_environment(self.original_triton_settings)
            self.cache_scope.__exit__(None, None, None)
            if os.path.exists(self.triton_cache_dir):
                shutil.rmtree(self.triton_cache_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to cleanup Triton environment: {e}")
        finally:
            # Always clear logging config to prevent state pollution between tests
            structured_logging.clear_logging_config()

    def setup_test_with_fresh_cache(self):
        """Setup individual test with completely fresh cache."""
        test_cache_dir = tempfile.mkdtemp(prefix="triton_test_cache_")
        prev_cache_dir = knobs.cache.dir
        knobs.cache.dir = test_cache_dir
        return test_cache_dir, prev_cache_dir

    def cleanup_test_cache(self, test_cache_dir, prev_cache_dir):
        """Cleanup test-specific cache."""
        knobs.cache.dir = prev_cache_dir
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir, ignore_errors=True)
