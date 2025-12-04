# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add tritonparse_root to sys.path for module resolution
    tritonparse_root = Path(__file__).resolve().parents[4]
    if str(tritonparse_root) not in sys.path:
        sys.path.insert(0, str(tritonparse_root))

import torch
import triton
import triton.language as tl
from tritonparse.reproducer.tests.artifacts.triton_preprocess import scale_kernel


@triton.jit
def main_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scaled = scale_kernel(x, scale)
    result = scaled * 2.0
    tl.store(output_ptr + offsets, result, mask=mask)


def launch_main_kernel() -> None:
    """Launch and test the main_kernel on GPU."""
    if not torch.cuda.is_available():
        print("CUDA not available - showing call graph only")
        print("  main_kernel -> scale_kernel -> add_values")
        return

    size = 1024
    scale_factor = 3.0
    BLOCK_SIZE = 256

    x = torch.randn(size, device="cuda", dtype=torch.float32)
    output = torch.zeros_like(x)

    main_kernel[(256,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=size,
        scale=scale_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    print("✅ Kernel executed successfully")


if __name__ == "__main__":
    launch_main_kernel()
