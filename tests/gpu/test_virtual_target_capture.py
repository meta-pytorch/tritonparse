# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Real cross-target codegen with host allocations, never target execution.

Requires Triton beta (3.9), an NVIDIA host GPU, and both compiler backends.
Every capture/reference runs in a fresh spawn process: install has no undo.
The sentinel checks prove non-execution, not numerical correctness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from parameterized import parameterized


_SCRIPT_COMMON = """
import inspect
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton import knobs


# `c_cache` is an fb-Triton jit() option; upstream Triton does not accept
# it. Request it only where the installed Triton supports it.
_C_CACHE_KWARGS = (
    {"c_cache": True}
    if "c_cache" in inspect.signature(triton.jit).parameters
    else {}
)


def capture_jit(fn):
    return triton.jit(fn, **_C_CACHE_KWARGS)


def binary_info(binary):
    assert isinstance(binary, bytes)
    assert binary[:4] == b"\\x7fELF"
    assert len(binary) > 64
    return {
        "size": len(binary),
        "machine": int.from_bytes(binary[18:20], "little"),
    }


def save_report(report):
    Path(__file__).with_suffix(".json").write_text(json.dumps(report))
"""

_JIT_SCRIPT = (
    _SCRIPT_COMMON
    + """
@capture_jit
def store_kernel(x, output, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(x + offsets, offsets < N, other=0) + 7
    tl.store(output + offsets, value, offsets < N)


@capture_jit
def tuple_kernel(values, output, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(values[0] + offsets, offsets < N, other=0)
    y = tl.load(values[1][0] + offsets, offsets < N, other=0)
    tl.store(output + offsets, x + y + values[1][1], offsets < N)


# Citrine C3: allocate on the real host device, including during capture.
x = torch.ones(64, device="cuda", dtype=torch.float32)
output = torch.full_like(x, -123)
reference = os.environ["TRITONPARSE_TEST_REFERENCE"] == "1"
report = {"kernels": [], "unchanged": [], "hot_reused": []}
for kernel, args in ((store_kernel, (x, output)),
                     (tuple_kernel, ((x, (x, 3)), output))):
    if _C_CACHE_KWARGS:
        assert kernel.c_cache is True
    for block_size in (32, 64):
        # Different specializations give each grid form a cold and a hot call.
        if block_size == 32:
            grid = 2
        else:
            grid = lambda meta: triton.cdiv(64, meta["BLOCK_SIZE"])
        first = None
        for _ in range(2):
            if reference:
                compiled = kernel.run(
                    *args, N=64, BLOCK_SIZE=block_size,
                    grid=(triton.cdiv(64, block_size),), warmup=True,
                )
            else:
                compiled = kernel[grid](*args, N=64, BLOCK_SIZE=block_size)
            assert compiled.module is None
            assert compiled.function is None
            assert getattr(compiled, "_dispatcher", None) is None
            report["unchanged"].append(output.cpu().tolist() == [-123] * 64)
            if first is None:
                first = compiled
                ext = "cubin" if compiled.metadata.target.backend == "cuda" else "hsaco"
                stages = ("ttir", "ttgir", "llir") + (("ptx",) if ext == "cubin" else ())
                report["kernels"].append({
                    "name": compiled.name,
                    "hash": compiled.hash,
                    "ir": {stage: compiled.asm[stage] for stage in stages},
                    "binary": binary_info(compiled.asm[ext]),
                })
            else:
                report["hot_reused"].append(compiled is first)
save_report(report)
"""
)

_AUTOTUNE_SCRIPT = (
    _SCRIPT_COMMON
    + """
from tritonparse.virtual_target import UnsupportedCaptureError


output = torch.full((32,), -123, device="cuda", dtype=torch.float32)
effects = []


def pre_hook(args):
    effects.append("config_pre_hook")
    args["output"].zero_()


def benchmark(fn, **kwargs):
    effects.append("benchmark")
    output.zero_()
    raise AssertionError("Measured autotuning must be rejected before benchmarking")


rejected = []
for custom_bench in (False, True):
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 32}, pre_hook=pre_hook),
            triton.Config({"BLOCK_SIZE": 64}, pre_hook=pre_hook),
        ],
        key=["N"],
        reset_to_zero=["output"],
        do_bench=benchmark if custom_bench else None,
        cache_results=True,
    )
    @capture_jit
    def measured_kernel(output, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        tl.store(output + offsets, 0, offsets < N)

    try:
        measured_kernel[(1,)](output, N=32)
    except UnsupportedCaptureError:
        rejected.append(custom_bench)
    else:
        raise AssertionError("Measured autotuning was not rejected")

save_report({
    "rejected": rejected,
    "effects": effects,
    "unchanged": output.cpu().tolist() == [-123] * 32,
    "autotune_files": [str(p) for p in Path(knobs.cache.dir).rglob("*.autotune.json")],
})
"""
)

_MISMATCH_SCRIPT = (
    _SCRIPT_COMMON
    + """
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def other_target_kernel(output):
    tl.store(output, 0)


try:
    triton.compile(
        ASTSource(other_target_kernel, signature={"output": "*fp32"}),
        target=GPUTarget("cuda", 80, 32),
    )
except ValueError as error:
    assert "does not match the virtual-target capture profile" in str(error)
else:
    raise AssertionError("A compile-only conflicting target was accepted")
save_report({"rejected": True})
"""
)


_TLX_SCRIPT = (
    _SCRIPT_COMMON
    + """
from triton.language.extra.tlx.hw.target import current_target
from triton.tlx.ops import mm


target = current_target()
# Citrine C3: these tensors stay on the host GPU even for virtual gfx950.
a = torch.ones((32, 32), device="cuda", dtype=torch.float16)
b = torch.ones_like(a)
output = torch.full_like(a, -123)
unchanged = []
for _ in range(2):
    result = mm(a, b, out=output, space="heuristic")
    assert result is output
    unchanged.append(output.cpu().tolist() == [[-123] * 32 for _ in range(32)])

# Inspect binaries before run_capture removes its private compilation cache.
kernels = []
pattern = "*.hsaco" if target.is_rocm else "*.cubin"
stages = ("ttir", "ttgir", "llir") + (() if target.is_rocm else ("ptx",))
for binary in Path(knobs.cache.dir).rglob(pattern):
    kernels.append({
        "ir": {stage: binary.with_suffix("." + stage).read_text()
               for stage in stages},
        "binary": binary_info(binary.read_bytes()),
    })
save_report({"target": target.key, "tmem_columns": target.tmem_columns,
             "kernels": kernels, "unchanged": unchanged})
"""
)


def _prepare_worker_environment(root: str, reference: bool) -> None:
    # GPU/Triton imports must follow environment setup in this fresh process.
    directory = Path(root)
    cache = directory / ("reference-cache" if reference else "host-cache")
    os.environ.update(
        {
            "TMPDIR": root,
            "TRITON_CACHE_DIR": str(cache),
            "TRITON_DUMP_DIR": str(cache / "dump"),
            "TRITON_OVERRIDE_DIR": str(cache / "override"),
            "TRITON_CACHE_MANAGER": "triton.runtime.cache:FileCacheManager",
            "TRITON_STORE_BINARY_ONLY": "0",
            "TRITON_ALWAYS_COMPILE": "0",
            "TRITON_COMPILE_IQ_APPLY": "0",
            "TRITON_COMPILE_IQ_COLLECT": "0",
            "TRITON_TRACE_LAUNCH": "0",
            "TRITON_TRACE_LAUNCH_WITHIN_PROFILING": "0",
            "TRITON_TRACE_COMPRESSION": "none",
            "TRITONPARSE_MORE_TENSOR_INFORMATION": "0",
            "TRITONPARSE_SAVE_TENSOR_BLOBS": "0",
            "TRITONPARSE_TRACE_MANIFOLD": "0",
            "TRITONPARSE_TEST_REFERENCE": str(int(reference)),
            "TORCHINDUCTOR_CACHE_DIR": str(directory / "inductor-cache"),
        }
    )
    os.environ.pop("TRITON_TRACE", None)
    os.environ.pop("TRITON_REMOTE_CACHE_BACKEND", None)
    tempfile.tempdir = root


def _run_worker(root: str, target: str, reference: bool = False) -> dict:
    _prepare_worker_environment(root, reference)

    import torch
    from triton.backends.compiler import GPUTarget  # @manual=//triton:triton
    from triton.runtime import driver  # @manual=//triton:triton

    if not torch.cuda.is_available():
        raise RuntimeError("Virtual-target GPU regression tests require a host GPU")
    torch.cuda.init()
    host = driver.active
    host_target = host.get_current_target()
    host_name = torch.cuda.get_device_name()
    host.get_current_stream(host.get_current_device())
    script = Path(root) / "workload.py"
    with patch.object(
        host.utils,
        "load_binary",
        side_effect=AssertionError("A target binary reached the host GPU loader"),
    ) as load_binary:
        if reference:
            # No virtual_target.install, handle patch, or capture logging here.
            backend, arch = target.split(":")
            explicit_target = GPUTarget(
                backend,
                int(arch) if backend == "cuda" else arch,
                32 if backend == "cuda" else 64,
            )
            with patch.object(host, "get_current_target", return_value=explicit_target):
                runpy.run_path(str(script), run_name="__main__")
        else:
            from tritonparse.capture import run_capture

            run_capture(
                argparse.Namespace(
                    target=target,
                    out=Path(root) / "trace",
                    script=script,
                    script_args=[],
                )
            )
        report = json.loads(script.with_suffix(".json").read_text())
        report["load_calls"] = load_binary.call_count
        report["host_backend"] = host_target.backend
        report["host_device"] = host_name
        return report


class VirtualTargetCaptureTest(unittest.TestCase):
    def _spawn(self, root: Path, target: str, *, reference: bool = False) -> dict:
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            return pool.apply_async(_run_worker, (str(root), target, reference)).get(
                timeout=600
            )

    def _events(self, root: Path) -> list[dict]:
        return [
            json.loads(line)
            for trace in (root / "trace").glob("*.ndjson")
            for line in trace.read_text().splitlines()
        ]

    def _assert_trace(self, events: list[dict], target: str) -> tuple[list, list]:
        compilations = [e for e in events if e["event_type"] == "compilation"]
        launches = [e for e in events if e["event_type"] == "launch"]
        self.assertTrue(compilations)
        self.assertTrue(launches)
        backend, arch = target.split(":")
        expected_target = {
            "backend": backend,
            "arch": int(arch) if backend == "cuda" else arch,
            "warp_size": 32 if backend == "cuda" else 64,
        }
        for compilation in compilations:
            metadata = compilation["payload"]["metadata"]
            self.assertEqual(metadata["target"], expected_target)
            self.assertEqual(metadata["virtual_target"]["profile"], target)
            self.assertIs(metadata["runtime_validated"], False)
            self.assertEqual(compilation["virtual_target"], metadata["virtual_target"])
            self.assertIs(compilation["runtime_validated"], False)
            files = compilation["payload"]["file_content"]
            for stage in ("ttir", "ttgir", "llir"):
                contents = [
                    text for name, text in files.items() if name.endswith("." + stage)
                ]
                self.assertEqual(len(contents), 1, stage)
                self.assertIn("module" if stage != "llir" else "define", contents[0])
        hashes = {c["payload"]["metadata"]["hash"] for c in compilations}
        for launch in launches:
            self.assertEqual(launch["virtual_target"]["profile"], target)
            self.assertIs(launch["executed"], False)
            self.assertIsNone(launch["function"])
            self.assertIn(launch["compilation_metadata"]["hash"], hashes)
        return compilations, launches

    @parameterized.expand(
        [("cuda:90", 190), ("cuda:100", 190), ("hip:gfx942", 224), ("hip:gfx950", 224)]
    )
    def test_jit_cold_hot_grids_and_tuples(self, target: str, elf_machine: int) -> None:
        with tempfile.TemporaryDirectory(prefix="virtual-target-jit-") as temp:
            root = Path(temp)
            (root / "workload.py").write_text(_JIT_SCRIPT)
            captured = self._spawn(root, target)
            reference = self._spawn(root, target, reference=True)
            self.assertEqual(captured["load_calls"], 0)
            self.assertEqual(reference["load_calls"], 0)
            self.assertEqual(captured["unchanged"], [True] * 8)
            self.assertEqual(reference["unchanged"], [True] * 8)
            self.assertEqual(captured["hot_reused"], [True] * 4)
            self.assertEqual(reference["hot_reused"], [True] * 4)
            self.assertEqual(len(captured["kernels"]), 4)
            self.assertEqual(len(reference["kernels"]), 4)
            for actual, expected in zip(captured["kernels"], reference["kernels"]):
                self.assertEqual(actual["binary"]["machine"], elf_machine)
                self.assertEqual(expected["binary"]["machine"], elf_machine)
                self.assertEqual(actual["hash"], expected["hash"])
                self.assertEqual(actual["ir"], expected["ir"])
            if target == "cuda:100":
                self.assertTrue(captured["host_device"])
                self.assertEqual(captured["host_device"], reference["host_device"])
                self.assertTrue(
                    all(
                        ".target sm_100" in kernel["ir"]["ptx"]
                        for kernel in captured["kernels"]
                    )
                )
            compilations, launches = self._assert_trace(self._events(root), target)
            self.assertEqual(len(compilations), 4)
            self.assertEqual(len(launches), 8)
            self.assertEqual([e["grid"] for e in launches], [[2], [2], [1], [1]] * 2)
            expected_ir = {k["hash"]: k["ir"] for k in reference["kernels"]}
            for compilation in compilations:
                payload = compilation["payload"]
                for stage, text in expected_ir[payload["metadata"]["hash"]].items():
                    self.assertEqual(
                        [
                            value
                            for name, value in payload["file_content"].items()
                            if name.endswith("." + stage)
                        ],
                        [text],
                    )
            self.assertEqual(
                {e["compilation_metadata"]["hash"] for e in launches},
                set(expected_ir),
            )

    def test_measured_autotune_rejected_before_effects(self) -> None:
        with tempfile.TemporaryDirectory(prefix="virtual-target-autotune-") as temp:
            root = Path(temp)
            (root / "workload.py").write_text(_AUTOTUNE_SCRIPT)
            result = self._spawn(root, "cuda:90")
            self.assertEqual(result["load_calls"], 0)
            self.assertEqual(result["rejected"], [False, True])
            self.assertEqual(result["effects"], [])
            self.assertTrue(result["unchanged"])
            self.assertEqual(result["autotune_files"], [])
            self.assertEqual(self._events(root), [])

    def test_compile_only_rejects_a_conflicting_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="virtual-target-mismatch-") as temp:
            root = Path(temp)
            (root / "workload.py").write_text(_MISMATCH_SCRIPT)
            result = self._spawn(root, "cuda:90")
            self.assertTrue(result["rejected"])
            self.assertEqual(result["load_calls"], 0)
            self.assertEqual(self._events(root), [])

    @parameterized.expand(
        [
            ("cuda:100", "sm100", ".target sm_100", 190),
            ("hip:gfx950", "gfx950", "gfx950", 224),
        ]
    )
    def test_tlx_eager_mm(
        self, target: str, key: str, arch_marker: str, elf_machine: int
    ) -> None:
        try:
            tlx_available = (
                importlib.util.find_spec("triton.language.extra.tlx") is not None
            )
        except ModuleNotFoundError:
            tlx_available = False
        if not tlx_available:
            self.skipTest("TLX is not available in this Triton build")
        with tempfile.TemporaryDirectory(prefix="virtual-target-tlx-") as temp:
            root = Path(temp)
            (root / "workload.py").write_text(_TLX_SCRIPT)
            result = self._spawn(root, target)
            self.assertEqual(result["load_calls"], 0)
            self.assertEqual(result["target"], key)
            self.assertEqual(result["unchanged"], [True, True])
            self.assertTrue(result["kernels"])
            if target == "cuda:100":
                self.assertTrue(result["host_device"])
                self.assertEqual(512, result["tmem_columns"])
            for kernel in result["kernels"]:
                self.assertEqual(kernel["binary"]["machine"], elf_machine)
                stage = "ptx" if target.startswith("cuda:") else "llir"
                self.assertIn(arch_marker, kernel["ir"][stage])
            compilations, launches = self._assert_trace(self._events(root), target)
            self.assertEqual(len(launches), 2 * len(compilations))
