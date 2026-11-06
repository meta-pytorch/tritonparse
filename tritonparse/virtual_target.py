# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Process-lifetime compile-only patches; not an in-process GPU emulator."""

from __future__ import annotations

import importlib
import inspect
import os
from functools import wraps
from typing import Any

from .hook_compat import hook_is_set
from .target_profiles import TargetProfile


_installed = False


class UnsupportedCaptureError(RuntimeError):
    """The workload needs execution or an unsupported runtime path."""


def _unsupported(*args: Any, **kwargs: Any) -> Any:
    raise UnsupportedCaptureError(
        "Virtual-target capture does not support GPU loading, benchmarking, "
        "CUDA graphs or torch.compile. Use an eager, deterministic capture script."
    )


class _VirtualUtils:
    def __init__(self, profile: TargetProfile) -> None:
        self.profile = profile

    def get_device_properties(self, device: Any) -> dict[str, int]:
        return {
            "max_shared_mem": self.profile.max_shared_mem,
            "multiprocessor_count": self.profile.multiprocessor_count,
            "warpSize": self.profile.warp_size,
        }

    load_binary = staticmethod(_unsupported)
    unload_module = staticmethod(_unsupported)


class _VirtualDriver:
    def __init__(self, host: Any, profile: TargetProfile) -> None:
        self.host = host
        self.profile = profile
        self.utils = _VirtualUtils(profile)

    def get_current_target(self) -> Any:
        return self.profile.target()

    def get_benchmarker(self) -> Any:
        return _unsupported

    launcher_cls = staticmethod(_unsupported)

    def __getattr__(self, name: str) -> Any:
        # Allocation devices and streams remain host-runtime objects.
        return getattr(self.host, name)


def _capture_launch(
    grid_x: int,
    grid_y: int,
    grid_z: int,
    stream: Any,
    function: Any,
    packed_metadata: Any,
    launch_metadata: Any,
    enter_hook: Any,
    exit_hook: Any,
    *args: Any,
) -> None:
    if function is not None:
        raise UnsupportedCaptureError("Capture received an executable function handle")
    if enter_hook:
        enter_hook(launch_metadata)
    if exit_hook:
        exit_hook(launch_metadata)


def _install_kernel_patch(compiled_kernel: Any, profile: TargetProfile) -> None:
    target = profile.target()

    def init_handles(kernel: Any) -> None:
        if kernel.module is not None or kernel.function is not None:
            raise UnsupportedCaptureError("Capture cannot reuse loaded kernel handles")
        if kernel.metadata.target != target:
            raise UnsupportedCaptureError("A cached kernel has a different target")
        kernel._dispatcher = None
        kernel._run = _capture_launch
        kernel.n_regs = None
        kernel.n_spills = None
        kernel.n_max_threads = None

    compiled_kernel._init_handles = init_handles


def _normalize_grid(grid: Any) -> Any:
    return (grid,) if type(grid) is int else grid


def _install_grid_patch(jit_function: Any) -> None:
    original = jit_function.run

    @wraps(original)
    def run(function: Any, *args: Any, grid: Any, **kwargs: Any) -> Any:
        if callable(grid):
            grid_fn = grid

            def normalized(metadata: Any) -> Any:
                return _normalize_grid(grid_fn(metadata))

            grid = normalized
        return original(function, *args, grid=_normalize_grid(grid), **kwargs)

    jit_function.run = run


def _check_runtime(triton: Any, knobs: Any, jit: Any, compiled_kernel: Any) -> None:
    if not triton.__version__.startswith("3.9."):
        raise UnsupportedCaptureError(
            "Virtual-target capture currently requires Triton 3.9"
        )
    if tuple(inspect.signature(compiled_kernel._init_handles).parameters) != ("self",):
        raise UnsupportedCaptureError(
            "Unsupported CompiledKernel initialization contract"
        )
    if knobs.runtime.interpret or knobs.runtime.override_arch:
        raise UnsupportedCaptureError(
            "Capture cannot use interpretation or TRITON_OVERRIDE_ARCH"
        )
    if knobs.compilation.override or knobs.runtime.add_stages_inspection_hook:
        raise UnsupportedCaptureError(
            "Capture cannot use IR overrides or custom stage hooks"
        )
    for name in (
        "launch_enter_hook",
        "launch_exit_hook",
        "kernel_load_start_hook",
        "kernel_load_end_hook",
        "kernel_unload_hook",
        "jit_cache_hook",
        "jit_post_compile_hook",
    ):
        if hook_is_set(getattr(knobs.runtime, name, None)):
            raise UnsupportedCaptureError(f"Capture cannot share runtime hook {name}")
    registry = getattr(jit, "_triton_jit_function_registry", {})
    if any(function.device_caches for function in registry.values()):
        raise UnsupportedCaptureError(
            "Capture must start before any JIT specialization"
        )


def _configure_runtime(knobs: Any, cache_dir: str) -> None:
    from triton.runtime.cache import FileCacheManager

    knobs.cache.dir = cache_dir
    knobs.cache.dump_dir = os.path.join(cache_dir, "dump")
    knobs.cache.override_dir = os.path.join(cache_dir, "override")
    knobs.cache.manager_class = FileCacheManager
    knobs.cache.remote_manager_class = None
    knobs.compilation.store_binary_only = False
    knobs.compilation.dump_ir = False
    knobs.autotuning.cache = False
    knobs.autotuning.dump_best_config_ir = False
    knobs.runtime.use_triton_dispatcher = False
    knobs.runtime.enable_c_cache = False
    knobs.runtime.use_autotune_c_cache = False
    os.environ["TRITON_COMPILE_IQ_APPLY"] = "0"
    os.environ["TRITON_COMPILE_IQ_COLLECT"] = "0"


def _install_torch_queries(torch: Any, profile: TargetProfile) -> None:
    original = torch.cuda.get_device_properties

    def properties(device: Any = None) -> Any:
        original(device)
        return profile.properties()

    def capability(device: Any = None) -> tuple[int, int]:
        props = properties(device)
        if profile.backend != "cuda":
            raise UnsupportedCaptureError("A HIP target has no CUDA compute capability")
        return props.major, props.minor

    torch.cuda.get_device_properties = properties
    torch.cuda.get_device_capability = capability
    torch.cuda.get_device_name = lambda device=None: properties(device).name
    torch.compile = _unsupported
    torch.cuda.graph = _unsupported
    torch.cuda.CUDAGraph = _unsupported


def _install_tlx_query(profile: TargetProfile) -> None:
    module_name = "triton.language.extra.tlx.hw.target"
    try:
        target = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and (
            module_name == error.name or module_name.startswith(error.name + ".")
        ):
            return
        raise
    target.is_rocm = lambda: profile.backend == "hip"
    target.current_target.cache_clear()


def install(profile: TargetProfile, cache_dir: str) -> dict[str, Any]:
    """Install once in a fresh capture process. Deliberately has no restore API."""
    global _installed
    if _installed:
        raise UnsupportedCaptureError("Use a new process for each virtual target")

    import torch
    import triton
    import triton.runtime.jit as jit
    import triton.testing as testing
    from triton import knobs
    from triton.compiler.compiler import CompiledKernel, make_backend
    from triton.runtime import driver
    from triton.runtime.autotuner import Autotuner

    _check_runtime(triton, knobs, jit, CompiledKernel)
    if not torch.cuda.is_available():
        raise UnsupportedCaptureError(
            "Capture requires a host GPU for real tensor allocation"
        )
    _configure_runtime(knobs, cache_dir)
    host = driver.active
    host_name = torch.cuda.get_device_name()
    virtual = _VirtualDriver(host, profile)
    driver.set_active(virtual)
    _install_kernel_patch(CompiledKernel, profile)
    _install_grid_patch(jit.JITFunction)
    _install_torch_queries(torch, profile)
    _install_tlx_query(profile)
    Autotuner._bench = _unsupported
    testing.do_bench = _unsupported
    testing.do_bench_cudagraph = _unsupported
    _installed = True
    try:
        make_backend(profile.target()).hash()
    except (OSError, ImportError) as error:
        raise UnsupportedCaptureError(
            f"Compiler tools for {profile.key} are unavailable; install the target backend/codegen"
        ) from error
    return {**profile.metadata(), "host_device": host_name}
