# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
State management for bisect workflow.

This module provides state persistence for the bisect workflow, enabling
checkpoint/resume functionality. The state is saved as JSON and can be
loaded to continue from where the workflow left off.
"""

from enum import Enum


class BisectPhase(Enum):
    """
    Bisect workflow phases.

    The workflow progresses through these phases sequentially:
    1. TRITON_BISECT: Find culprit Triton commit
    2. TYPE_CHECK: Detect if culprit is an LLVM bump
    3. PAIR_TEST: Test commit pairs to find LLVM range (if LLVM bump)
    4. LLVM_BISECT: Find culprit LLVM commit (if LLVM bump)
    5. COMPLETED: Workflow finished successfully
    6. FAILED: Workflow failed with error
    """

    TRITON_BISECT = "triton_bisect"
    TYPE_CHECK = "type_check"
    PAIR_TEST = "pair_test"
    LLVM_BISECT = "llvm_bisect"
    COMPLETED = "completed"
    FAILED = "failed"
