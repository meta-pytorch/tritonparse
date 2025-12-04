# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import triton
import triton.language as tl
from tritonparse.reproducer.tests.artifacts.triton_utils import add_values


@triton.jit
def scale_kernel(
    x: tl.tensor,
    scale: tl.constexpr,
) -> tl.tensor:
    result = x * scale
    return add_values(result, 0.0)
