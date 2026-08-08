# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Triton kernels and workloads shared by the ASPLOS'27 artifact scripts.

Two workloads, for two different jobs:

``workload_small``
    Six matmul launches across three block-size triples plus eight fused-op launches.
    Five compilations total, tiny launch volume.  Used for the reference trace that
    ships in git and drives 00_kick_the_tires.sh.

``workload_dense``
    The same matmul under ``triton.autotune``, which benchmarks every config on every
    new shape and therefore emits thousands of launch records.  Used for the C6
    storage measurement, because the reconstruct stage's compression ratio is dominated
    by redundancy across launch records -- a low-launch workload understates it badly
    (3.8x vs 63x on the same machine).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": 16},
                  num_stages=1, num_warps=1)
    for m, n in ((16, 16), (32, 16), (16, 32))
]

matmul_kernel_autotuned = triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])(
    matmul_kernel
)


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


# ---------------------------------------------------------------- host wrappers
def matmul(a, b, bm, bn, bk):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matmul_kernel[(triton.cdiv(M, bm) * triton.cdiv(N, bn),)](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
        num_stages=1, num_warps=1,
    )
    return c


def matmul_autotuned(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)

    matmul_kernel_autotuned[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c


def fused_op(a, b, c, scale_factor, activation):
    n_elements = a.numel()
    out = torch.empty_like(a)
    block = 8
    fused_op_kernel[(triton.cdiv(n_elements, block),)](
        a, b, c, out, n_elements, scale_factor,
        ACTIVATION=activation, BLOCK_SIZE=block,
    )
    return out


# ---------------------------------------------------------------- workloads
def workload_small(device: str = "cuda") -> None:
    """Five compilations, 14 launches.  Keeps the shipped reference trace small."""
    torch.manual_seed(0)

    a = torch.randn((64, 64), device=device, dtype=torch.float16)
    b = torch.randn((64, 64), device=device, dtype=torch.float16)
    for bm, bn, bk in [(16, 16, 16), (32, 16, 16), (16, 32, 16)]:
        for _ in range(2):
            matmul(a, b, bm, bn, bk)

    x = torch.randn(64, device=device, dtype=torch.float32)
    y = torch.randn(64, device=device, dtype=torch.float32)
    z = torch.randn(64, device=device, dtype=torch.float32)
    for activation in (None, "relu"):
        for i in range(4):
            fused_op(x, y, z, 1.0 + 0.5 * i, activation)

    torch.cuda.synchronize()


def workload_dense(device: str = "cuda") -> None:
    """Autotuned matmul over several shapes: thousands of launch records."""
    torch.manual_seed(0)

    for m, n, k in [(32, 32, 32), (64, 32, 32), (32, 64, 32), (64, 64, 64)]:
        a = torch.randn((m, k), device=device, dtype=torch.float16)
        b = torch.randn((k, n), device=device, dtype=torch.float16)
        matmul_autotuned(a, b)

    torch.cuda.synchronize()
