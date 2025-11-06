"""
Simple Triton kernel for tensor addition. This file is not included in unit tests.

Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TRITONPARSE_DEBUG=1 python tests/test_add.py
```
"""

import os

import torch
import triton
import triton.language as tl
import tritonparse.structured_logging
import tritonparse.utils

log_path = "./logs"
tritonparse.structured_logging.init(log_path, enable_trace_launch=True)

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"


@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


def tensor_add(a, b):
    n_elements = a.numel()
    c = torch.empty_like(a)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)
    return c


def simple_add(a, b):
    return a + b


def test_tensor_add():
    torch.manual_seed(0)
    size = (1024, 1024)
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)

    # Test Triton kernel
    c_triton = tensor_add(a, b)
    c_triton.sum()
    tensor_add(a, b)
    print("Triton kernel executed successfully")

    # Test torch.compile
    compiled_add = torch.compile(simple_add)
    c_compiled = compiled_add(a, b)
    c_compiled.sum()
    print("Torch compiled function executed successfully")


if __name__ == "__main__":
    test_tensor_add()
    # Use improved unified_parse with explicit output directory
    tritonparse.utils.unified_parse(
        source=log_path, out="./parsed_output", overwrite=True
    )
