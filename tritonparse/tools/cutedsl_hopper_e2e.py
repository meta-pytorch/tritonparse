# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Trace and validate the upstream Hopper TMA/WGMMA GEMM on real hardware.

Requires a CUTLASS checkout and a Python environment with Torch, CuTeDSL, and
the upstream `cutlass.hooks` events: `POST_COMPILE` (NVIDIA/cutlass#3697) for
compilation events and `PRE_EXECUTE` (NVIDIA/cutlass#3703) for launch events.
With only `POST_COMPILE`, the trace check below fails on the missing launch
events instead of silently validating a compilation-only trace.
Run through the local Buck launcher::

    buck run fbcode//pytorch/tritonparse:cutedsl_hopper_e2e -- \
        --python /path/to/conda/bin/python \
        --tritonparse-root /path/to/tritonparse \
        --cutlass-root /path/to/cutlass --output /tmp/cutedsl-hopper-run

``--mode eager`` calls the ``@cute.jit`` object on every launch, while
``--mode compiled`` launches the result of one ``cute.compile`` per case.
``--static-layout`` keeps the tensor layouts static so that the compiled
launch configuration fixes the grid, and ``--sass`` also derives SASS from the
CUBIN with nvdisasm.

The output contains raw logs and a numerical/trace report. Compiler artifacts
stay in a temporary directory that tracing removes at teardown. Pass the logs
directory to the normal tritonparse parser to generate the viewer data. The
upstream example is imported without modifying its kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


CASES = {
    "batched": {
        "mnkl": (512, 512, 256, 2),
        "tile_shape_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
    },
    "boundary": {
        "mnkl": (272, 384, 192, 3),
        "tile_shape_mn": (64, 64),
        "cluster_shape_mn": (1, 1),
    },
    "multicast": {
        "mnkl": (1024, 512, 512, 1),
        "tile_shape_mn": (128, 256),
        "cluster_shape_mn": (2, 1),
    },
}
EXAMPLE = "examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py"
ARTIFACT_SUFFIXES = {".mlir", ".ptx", ".cubin", ".sass"}


def _load_example(path: Path):
    spec = importlib.util.spec_from_file_location("cutedsl_hopper_gemm_upstream", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CUTLASS example: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expected_grid(config: dict) -> list[int]:
    # Mirrors HopperWgmmaGemmKernel._compute_grid: tiles rounded up to clusters.
    m, n, _, batch = config["mnkl"]
    tile_m, tile_n = config["tile_shape_mn"]
    tiles = (math.ceil(m / tile_m), math.ceil(n / tile_n), batch)
    cluster = (*config["cluster_shape_mn"], 1)
    return [math.ceil(t / c) * c for t, c in zip(tiles, cluster)]


def _check_trace(
    logs: Path, expected_launches: list[dict], args: argparse.Namespace
) -> dict:
    from tritonparse.validation.json_validator import validate_record

    events = []
    for path in sorted(logs.glob("dedicated_log_triton_trace_*.ndjson")):
        events.extend(
            json.loads(line) for line in path.read_text().splitlines() if line
        )
    for event in events:
        ok, errors = validate_record(event)
        assert ok, errors
    compilation_events = [
        event
        for event in events
        if event.get("event_type") == "compilation"
        and event.get("payload", {}).get("metadata", {}).get("backend_name")
        == "cutedsl"
    ]
    compilations = [event["payload"] for event in compilation_events]
    assert all(not event["metadata"]["cache_hit"] for event in compilations)
    by_hash = {event["metadata"]["hash"]: event for event in compilations}
    launches = [event for event in events if event.get("event_type") == "launch"]
    assert len(by_hash) == len(args.cases), (len(by_hash), len(args.cases))
    if args.mode == "compiled":
        # One cute.compile per case; every launch reuses its result.
        assert len(compilations) == len(args.cases), (
            len(compilations),
            len(args.cases),
        )
    assert len(launches) == len(expected_launches), (
        f"expected {len(expected_launches)} launch events, got "
        f"{len(launches)}; launch events require cutlass.hooks "
        "PRE_EXECUTE (NVIDIA/cutlass#3703) in the CuTeDSL build"
    )
    case_hashes = _check_launches(launches, expected_launches, by_hash, args)
    return {
        "compilation_events": len(compilations),
        "launch_events": len(launches),
        "case_hashes": case_hashes,
        "specializations": _check_artifacts(by_hash, case_hashes, args),
        "source_mappings": _check_source_mappings(compilation_events),
    }


def _check_source_mappings(compilations: list[dict]) -> list[dict]:
    """Parse real dumps to check that raw and clean IR retain source mappings."""
    from tritonparse.parse.trace_processor import parse_single_trace_content

    checked = {}
    for event in compilations:
        module_hash = event["payload"]["metadata"]["hash"]
        if module_hash in checked:
            continue
        parsed = json.loads(parse_single_trace_content(json.dumps(event)))
        mappings = parsed["payload"]["source_mappings"]
        stages = ["mlir", "clean_mlir", "ptx", "python"]
        if "kernel.sass" in event["payload"]["file_content"]:
            stages.append("sass")
        for stage in stages:
            assert mappings.get(stage), (module_hash, stage)
        checked[module_hash] = {
            "hash": module_hash,
            "stages": {stage: len(mappings[stage]) for stage in stages},
        }
    return list(checked.values())


def _check_launches(
    launches: list[dict],
    expected_launches: list[dict],
    by_hash: dict,
    args: argparse.Namespace,
) -> dict:
    case_hashes: dict[str, set[str]] = {}
    for launch, expected in zip(launches, expected_launches):
        module_hash = launch["compilation_metadata"]["hash"]
        assert module_hash in by_hash
        case_hashes.setdefault(expected["case"], set()).add(module_hash)
        extracted = launch["extracted_args"]
        # Named after the parameters of HopperWgmmaGemmKernel.__call__.
        assert list(extracted)[:3] == ["a", "b", "c"], list(extracted)
        tensors = [
            arg for arg in extracted.values() if arg.get("type") == "cute.Tensor"
        ]
        assert len(tensors) == 3, extracted
        for actual, tensor in zip(tensors, expected["tensors"]):
            assert actual["shape"] == tensor["shape"], (actual, tensor)
            assert actual["stride"] == tensor["stride"], (actual, tensor)
            assert actual["data_ptr"] == tensor["data_ptr"], (actual, tensor)
        if args.static_layout:
            assert launch["grid"] == expected["grid"], (launch["grid"], expected)
            assert "grid_unknown" not in launch, launch
        else:
            assert launch["grid"] == [] and launch["grid_unknown"], launch["grid"]
    assert all(len(hashes) == 1 for hashes in case_hashes.values()), case_hashes
    return {case: sorted(hashes) for case, hashes in case_hashes.items()}


def _check_artifacts(
    by_hash: dict, case_hashes: dict, args: argparse.Namespace
) -> list[dict]:
    case_by_hash = {h: case for case, hashes in case_hashes.items() for h in hashes}
    artifacts = []
    for module_hash, compilation in by_hash.items():
        config = CASES[case_by_hash[module_hash]]
        content = compilation["file_content"]
        ptx = content["kernel.ptx"]
        assert "wgmma.mma_async" in ptx
        assert "cp.async.bulk.tensor" in ptx
        assert "cuda.launch" in content["kernel.clean_mlir"]
        if args.sass:
            assert "HGMMA" in content["kernel.sass"], content["kernel.sass"][:200]
        metadata = compilation["metadata"]
        (launch_config,) = metadata["launch_configs"]
        assert all(isinstance(dim, int) for dim in launch_config["block"])
        assert metadata["cluster_dims"] == [*config["cluster_shape_mn"], 1], metadata
        if args.static_layout:
            assert launch_config["grid"] == _expected_grid(config), launch_config
        else:
            assert None in launch_config["grid"], launch_config
        source = compilation["python_source"]
        assert source["start_line"] == 1
        assert "def kernel(" in source["code"]
        artifacts.append(
            {
                "hash": module_hash,
                "case": case_by_hash[module_hash],
                "launch_config": launch_config,
                "lines": {
                    name: len(text.splitlines())
                    for name, text in sorted(content.items())
                },
                "python_lines": len(source["code"].splitlines()),
            }
        )
    return artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mode", choices=("eager", "compiled"), default="eager")
    parser.add_argument("--static-layout", action="store_true")
    parser.add_argument("--sass", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Use at least two repetitions to exercise launch comparisons")
    if len(args.cases) != len(set(args.cases)):
        parser.error("Choose each case at most once")
    example_path = args.cutlass_root.resolve() / EXAMPLE
    if not example_path.is_file():
        parser.error(f"CUTLASS example is missing: {example_path}")
    return args


def _make_tensors(mnkl: tuple[int, ...], static_layout: bool):
    import torch
    from cutlass.cute.runtime import from_dlpack

    m, n, k, batch = mnkl
    # Citrine C3: allocate workload inputs directly on the GPU.
    a = torch.randn((batch, m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((batch, n, k), device="cuda", dtype=torch.float16)
    c = torch.full((batch, m, n), float("nan"), device="cuda", dtype=a.dtype)
    views = [tensor.permute(1, 2, 0) for tensor in (a, b, c)]
    cute_tensors = [from_dlpack(view, assumed_align=16) for view in views]
    if not static_layout:
        cute_tensors = [
            tensor.mark_layout_dynamic(leading_dim=1) for tensor in cute_tensors
        ]
    return (a, b, c), views, cute_tensors


def _check_result(a, b, c) -> None:
    import torch

    torch.cuda.synchronize()
    reference = torch.bmm(a.float(), b.transpose(1, 2).float()).to(c.dtype)
    torch.testing.assert_close(c, reference, atol=0.02, rtol=0.001)


def _run_case(example, case_name: str, args: argparse.Namespace) -> list[dict]:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import torch

    config = CASES[case_name]
    gemm = example.HopperWgmmaGemmKernel(
        cutlass.Float32, config["tile_shape_mn"], config["cluster_shape_mn"]
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = None
    expected_launches = []
    # Keep allocations alive so repeated launches have distinct pointer metadata.
    retained_tensors = []
    for repetition in range(args.repeats):
        (a, b, c), views, cute_tensors = _make_tensors(
            config["mnkl"], args.static_layout
        )
        if args.mode == "eager":
            gemm(*cute_tensors, stream)
        else:
            if compiled is None:
                compiled = cute.compile(gemm, *cute_tensors, stream)
            compiled(*cute_tensors, stream)
        _check_result(a, b, c)
        expected_launches.append(
            {
                "case": case_name,
                "repetition": repetition,
                **config,
                "grid": _expected_grid(config),
                "tensors": [
                    {
                        "shape": list(tensor.shape),
                        "stride": list(tensor.stride()),
                        "data_ptr": hex(tensor.data_ptr()),
                    }
                    for tensor in views
                ],
            }
        )
        retained_tensors.append((a, b, c, cute_tensors))
        print(
            f"PASS {case_name} repetition={repetition} mnkl={config['mnkl']}",
            flush=True,
        )
    return expected_launches


def _run_cases(example_path: Path, args: argparse.Namespace) -> tuple[list[dict], dict]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This workload requires a Hopper GPU")
    torch.manual_seed(20260921)
    torch.set_float32_matmul_precision("highest")
    example = _load_example(example_path)
    expected_launches = []
    for case_name in args.cases:
        expected_launches.extend(_run_case(example, case_name, args))
    environment = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
    }
    return expected_launches, environment


def _cutedsl_dump_dir() -> str:
    from cutlass.cutlass_dsl import CuTeDSL

    return str(CuTeDSL._get_dsl().envar.dump_dir)


def main() -> None:
    from tritonparse.structured_logging import clear_logging_config, init

    args = _parse_args()
    example_path = args.cutlass_root.resolve() / EXAMPLE
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()

    cwd_before = set(Path.cwd().iterdir())
    try:
        init(
            trace_folder=str(logs),
            enable_cutedsl_tracing=True,
            enable_trace_launch=True,
            enable_full_python_source=True,
            enable_sass_dump=args.sass,
        )
        dump_dir = _cutedsl_dump_dir()
        expected_launches, environment = _run_cases(example_path, args)
    finally:
        clear_logging_config()
    if not {"CUTE_DSL_KEEP", "CUTE_DSL_DUMP_DIR"} & set(os.environ):
        # Nothing kept by the user: artifacts lived in a private temporary
        # directory that teardown removed, never in the working directory.
        stray = sorted(
            str(path)
            for path in set(Path.cwd().iterdir()) - cwd_before
            if path.suffix in ARTIFACT_SUFFIXES
        )
        assert not stray, stray
        assert not Path(dump_dir).exists(), dump_dir

    report = {
        "environment": environment,
        "options": {
            "mode": args.mode,
            "static_layout": args.static_layout,
            "sass": args.sass,
        },
        "upstream_example": str(example_path),
        "upstream_sha256": hashlib.sha256(example_path.read_bytes()).hexdigest(),
        "numerical_checks": expected_launches,
        "trace": _check_trace(logs, expected_launches, args),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["trace"], indent=2), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
