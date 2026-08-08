#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Generate the reference trace bundle shipped with the ASPLOS'27 artifact.

The offline checks need a trace that exercises every downstream stage, which means
it must contain:

  * more than one *compilation* of the same kernel   -> `diff --events A,B`
  * kernels with recorded *launches*                 -> `info`, `reproduce`

The repository's own fixtures under ``tests/example_output/`` satisfy this but were
recorded in early 2026 on a different machine and a much older Triton.  This script
re-records an equivalent trace with whatever stack is currently installed, so the
artifact stays self-consistent with ``requirements-pinned.txt``.

Requires a GPU.  The trace shipped with the artifact was produced this way and is
distributed as a GitHub Release asset (see scripts/fetch_reference_trace.sh);
running this script is the alternative for reviewers who prefer to record their own.

    python scripts/make_reference_trace.py --out-dir traces/reference

PUBLISHING NOTE.  A trace records the absolute path of every source file involved, so
whatever directory you record from ends up in the published artifact.  The trace shipped
in ``traces/reference/`` was therefore recorded from a scratch directory with its own
virtualenv, leaving only neutral paths.  If you regenerate it, do the same:

    mkdir /tmp/rec && cp scripts/make_reference_trace.py /tmp/rec/
    python3 -m venv /tmp/rec/venv
    /tmp/rec/venv/bin/pip install -r requirements-pinned.txt
    cd /tmp/rec && ./venv/bin/python make_reference_trace.py --out-dir out

Cache isolation is handled automatically (see below); the directory is not.

Produces:
    <out-dir>/logs/          raw structured log (NDJSON)
    <out-dir>/parsed/        reconstructed archive (*.ndjson.gz + log_file_list.json)
    <out-dir>/PROVENANCE.txt environment the trace was recorded on
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Isolate the Triton compilation cache BEFORE importing triton.
#
# Triton keys its cache on the kernel source, not on the file the source came from.  If
# the same kernel was ever compiled from another directory, Triton serves the cached
# entry and the trace records that *older* path.  We hit exactly this: a trace recorded
# from /tmp still carried 361 references to the author's original working directory.
# A private cache directory forces a real compile and keeps the published trace's
# provenance limited to the directory it was actually recorded in.
if not os.environ.get("TRITONPARSE_AE_KEEP_CACHE"):
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="tritonparse-ae-cache-")

import torch
import triton
import triton.language as tl

import tritonparse.context_manager

sys.path.insert(0, str(Path(__file__).parent))
from ae_platform import neutral_platform  # noqa: E402


# ---------------------------------------------------------------- kernels
# Kernel 1: matmul, compiled once per BLOCK_SIZE triple.  Distinct constexpr values
# give distinct compilations, which is what `diff --events 0,1` compares.
#
# NOTE: an earlier revision used @triton.autotune here.  It works, but autotune
# benchmarks every config on every new shape, producing ~3000 launch records and a
# 15 MB raw log — too large to keep in git.  Driving the constexprs explicitly gives
# the same number of compilations with ~1% of the launch volume.
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# Kernel 2: a plain fused element-wise op.  Compiled twice (one per ACTIVATION value)
# and launched many times, which is what `reproduce` needs.
@triton.jit
def fused_op_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
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


def matmul(a: torch.Tensor, b: torch.Tensor, bm: int, bn: int, bk: int) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
        num_stages=1, num_warps=1,
    )
    return c


def fused_op(a, b, c, scale_factor: float, activation: str | None) -> torch.Tensor:
    n_elements = a.numel()
    out = torch.empty_like(a)
    block = 8
    fused_op_kernel[(triton.cdiv(n_elements, block),)](
        a, b, c, out, n_elements, scale_factor,
        ACTIVATION=activation, BLOCK_SIZE=block,
    )
    return out


def workload(device: str) -> None:
    torch.manual_seed(0)

    # Three block-size triples -> three distinct compilations of matmul_kernel,
    # each launched a handful of times.
    a = torch.randn((64, 64), device=device, dtype=torch.float16)
    b = torch.randn((64, 64), device=device, dtype=torch.float16)
    for bm, bn, bk in [(16, 16, 16), (32, 16, 16), (16, 32, 16)]:
        for _ in range(2):
            matmul(a, b, bm, bn, bk)

    # Two ACTIVATION values -> two compilations; repeated calls -> many launches.
    x = torch.randn(64, device=device, dtype=torch.float32)
    y = torch.randn(64, device=device, dtype=torch.float32)
    z = torch.randn(64, device=device, dtype=torch.float32)
    for activation in (None, "relu"):
        for i in range(4):
            fused_op(x, y, z, 1.0 + 0.5 * i, activation)

    torch.cuda.synchronize()


# ---------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="traces/reference",
                    help="destination directory (created/overwritten)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: a CUDA device is required to record a trace.", file=sys.stderr)
        print(f"       CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
              file=sys.stderr)
        return 1

    out = Path(args.out_dir).resolve()
    if out.exists():
        shutil.rmtree(out)
    logs, parsed = out / "logs", out / "parsed"
    logs.mkdir(parents=True)
    parsed.mkdir(parents=True)

    # Record into a scratch directory first.  TritonParse names raw logs
    # dedicated_log_triton_trace_<user>_rank_<r>_pid_<pid>_host_<host>_.ndjson, which
    # bakes the recording machine's hostname and PID into a file we are about to
    # publish.  Rename to a neutral name and re-parse so the archive name is neutral
    # too.  The `dedicated_log_triton_trace_` prefix is load-bearing: `parse` rejects
    # any other filename.
    scratch = out / ".scratch"
    scratch.mkdir()
    print(f"recording into {out}")
    with tritonparse.context_manager.TritonParseManager(
        enable_trace_launch=True, log_dir=str(scratch), out=str(scratch / "parsed")
    ):
        workload("cuda")

    recorded = sorted(scratch.glob("dedicated_log_triton_trace_*.ndjson"))
    if not recorded:
        print("ERROR: no raw trace recorded", file=sys.stderr)
        return 1
    neutral = logs / "dedicated_log_triton_trace_ae_.ndjson"
    shutil.copy2(recorded[0], neutral)

    from tritonparse.parse.utils import unified_parse

    unified_parse(str(logs), out=str(parsed), overwrite=True)
    shutil.rmtree(scratch)

    archives = sorted(p.name for p in parsed.glob("*.ndjson.gz"))
    raw = sorted(p.name for p in logs.glob("*.ndjson"))
    if not archives:
        print("ERROR: no *.ndjson.gz produced", file=sys.stderr)
        return 1

    raw_bytes = sum(p.stat().st_size for p in logs.glob("*.ndjson"))
    arc_bytes = sum(p.stat().st_size for p in parsed.glob("*.ndjson.gz"))

    provenance = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        # Vendor kernel build tags are stripped here; see scripts/ae_platform.py.
        "platform": neutral_platform(),
        "raw_logs": raw,
        "archives": archives,
        "raw_bytes": raw_bytes,
        "archive_bytes": arc_bytes,
        "reconstruct_ratio": round(raw_bytes / max(arc_bytes, 1), 2),
    }
    (out / "PROVENANCE.txt").write_text(json.dumps(provenance, indent=2) + "\n")

    print("\nrecorded:")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size:,} B)")
    print("\nprovenance:")
    for k, v in provenance.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
