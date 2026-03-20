#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Shared test fixtures for the diff module tests.

Provides common constants, helper functions, and pre-built events
used across all diff test files.
"""

from typing import Any


# --- Python Source Fixtures ---

DEFAULT_PYTHON_SOURCE = """\
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
"""

DIFFERENT_PYTHON_SOURCE_MATMUL = """\
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)
"""

SIMILAR_PYTHON_SOURCE = """\
@triton.jit
def add_kernel_v2(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y + 1.0
    tl.store(output_ptr + offsets, output, mask=mask)
"""


# --- IR Fixtures ---

DEFAULT_TTIR = """\
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
    %4 = tt.store %2, %3 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
"""

LONGER_TTIR = """\
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
    %4 = arith.addf %3, %3 : tensor<1024xf32>
    %5 = arith.mulf %4, %3 : tensor<1024xf32>
    %6 = tt.store %2, %5 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
"""


# --- Event Factory Functions ---


def create_compilation_event(
    kernel_name: str = "add_kernel",
    kernel_hash: str = "abc123def456",
    num_stages: int = 3,
    num_warps: int = 4,
    occurrence_id: int = 0,
    python_source: str | None = None,
    ttir: str | None = None,
    source_mappings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a sample compilation event for testing."""
    return {
        "event_type": "compilation",
        "kernel_name": kernel_name,
        "kernel_hash": kernel_hash,
        "occurrence_id": occurrence_id,
        "compilation_metadata": {
            "num_stages": num_stages,
            "num_warps": num_warps,
            "num_ctas": 1,
        },
        "payload": {
            "python_source": {
                "content": python_source or DEFAULT_PYTHON_SOURCE,
                "start_line": 1,
            },
            "ttir": ttir or DEFAULT_TTIR,
            "source_mappings": source_mappings
            or {
                "python": {
                    "3": {"ttir_lines": [3, 4]},
                    "7": {"ttir_lines": [5, 6]},
                }
            },
        },
    }


def create_launch_event(
    kernel_hash: str = "abc123def456",
    extracted_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a sample launch event for testing."""
    return {
        "event_type": "launch",
        "compilation_metadata": {"hash": kernel_hash},
        "extracted_args": extracted_args
        or {
            "x_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
            },
            "y_ptr": {
                "type": "tensor",
                "shape": [1024],
                "dtype": "torch.float32",
            },
            "n_elements": {"type": "int", "value": 1024},
        },
    }


# --- Pre-built Events ---

COMP_EVENT_A = create_compilation_event(
    kernel_hash="abc123def456789",
    num_stages=3,
    num_warps=4,
    occurrence_id=0,
)

COMP_EVENT_B = create_compilation_event(
    kernel_hash="xyz789ghi012345",
    num_stages=5,
    num_warps=8,
    occurrence_id=1,
    ttir=LONGER_TTIR,
    source_mappings={
        "python": {
            "3": {"ttir_lines": [3, 4, 5]},
            "7": {"ttir_lines": [5, 6, 7, 8]},
        }
    },
)
