# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import triton
import triton.language as tl


@triton.jit
def add_values(
    a: tl.tensor,
    b: tl.tensor,
) -> tl.tensor:
    return a + b
