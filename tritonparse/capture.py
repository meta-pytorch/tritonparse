# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run a controlled Python script with one virtual compilation target."""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys
import tempfile
from pathlib import Path

from .target_profiles import target_profiles


logger: logging.Logger = logging.getLogger(__name__)


def add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", required=True, choices=tuple(target_profiles()))
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument(
        "script", type=Path, help="Eager Python capture script (not a benchmark)"
    )
    parser.add_argument("script_args", nargs=argparse.REMAINDER)


def _prepare_environment(cache_dir: str) -> None:
    os.environ.update(
        {
            "TRITON_CACHE_DIR": cache_dir,
            "TRITON_CACHE_MANAGER": "triton.runtime.cache:FileCacheManager",
            "TRITON_CACHE_AUTOTUNING": "0",
            "TRITON_USE_C_DISPATCHER": "0",
            "TRITON_ENABLE_C_CACHE": "0",
            "TRITON_AUTOTUNE_USE_C_CACHE": "0",
            "TRITON_COMPILE_IQ_APPLY": "0",
            "TRITON_COMPILE_IQ_COLLECT": "0",
            "TRITON_STORE_BINARY_ONLY": "0",
            "TRITONPARSE_MORE_TENSOR_INFORMATION": "0",
            "TRITONPARSE_SAVE_TENSOR_BLOBS": "0",
            "TRITONPARSE_TRACE_MANIFOLD": "0",
            "TRITON_TRACE_LAUNCH": "0",
            "TRITON_TRACE_LAUNCH_WITHIN_PROFILING": "0",
        }
    )
    os.environ.pop("TRITON_REMOTE_CACHE_BACKEND", None)
    os.environ.pop("TRITON_TRACE", None)


def run_capture(args: argparse.Namespace) -> None:
    script = args.script.resolve(strict=True)
    if not script.is_file():
        raise ValueError(f"Capture script is not a file: {script}")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = target_profiles()[args.target]
    with tempfile.TemporaryDirectory(prefix="tritonparse-capture-") as cache_dir:
        _prepare_environment(cache_dir)
        from . import structured_logging
        from .virtual_target import install

        metadata = install(profile, cache_dir)
        structured_logging._virtual_target = metadata
        structured_logging.TRITONPARSE_MORE_TENSOR_INFORMATION = False
        structured_logging.TRITONPARSE_SAVE_TENSOR_BLOBS = False
        structured_logging.TRITONPARSE_TRACE_MANIFOLD = False
        structured_logging.init(str(output), enable_trace_launch=True)
        logger.warning(
            "Virtual target %s: target kernels will not execute. Tensor outputs, "
            "numerical assertions and runtime resource validity are not evaluated.",
            profile.key,
        )
        sys.argv = [str(script), *args.script_args]
        sys.path.insert(0, str(script.parent))
        try:
            runpy.run_path(str(script), run_name="__main__")
        finally:
            handler = structured_logging.TRITON_TRACE_HANDLER
            if handler is not None:
                handler.flush()
                handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_capture_args(parser)
    run_capture(parser.parse_args())


if __name__ == "__main__":
    main()
