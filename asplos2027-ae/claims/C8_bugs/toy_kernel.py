# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Baseline workload for the C8 bug-reproduction check (paper 4.4).

Deliberately tiny.  The three patches in ``patches/`` inject the same three failure
classes the paper injects into the gpt-oss-20b kernels (paper Listing 1): a device-side
assertion, an illegal memory access, and a hang.

Tracing is initialised here but parsing is NOT: two of the three mutations kill or wedge
the process, so the raw log must be parsed afterwards by run.sh.
"""

import os

import torch
import triton
import triton.language as tl

import tritonparse.structured_logging

tritonparse.structured_logging.init(
    os.environ["AE_LOG_DIR"], enable_trace_launch=True
)


@triton.jit(debug=True)
def scale_kernel(x_ptr, o_ptr, n_elements, stride_ym, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x * 2.0, mask=mask)


def main() -> None:
    n = 256
    x = torch.randn(n, device="cuda")
    o = torch.empty_like(x)
    scale_kernel[(triton.cdiv(n, 64),)](x, o, n, 64, BLOCK=64)
    torch.cuda.synchronize()
    print("workload completed without error")


if __name__ == "__main__":
    main()
