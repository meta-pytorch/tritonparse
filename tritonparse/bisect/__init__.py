# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Bisect module for tritonparse.

This module provides tools for bisecting Triton and LLVM regressions.
"""

from tritonparse.bisect.base_bisector import BaseBisector, BisectError
from tritonparse.bisect.commit_detector import (
    CommitDetector,
    CommitDetectorError,
    LLVMBumpInfo,
)
from tritonparse.bisect.executor import CommandResult, ShellExecutor
from tritonparse.bisect.llvm_bisector import LLVMBisectError, LLVMBisector
from tritonparse.bisect.logger import BisectLogger
from tritonparse.bisect.pair_tester import (
    CommitPair,
    PairTester,
    PairTesterError,
    PairTestResult,
)
from tritonparse.bisect.state import BisectPhase, BisectState, StateManager
from tritonparse.bisect.triton_bisector import TritonBisectError, TritonBisector
from tritonparse.bisect.ui import BisectProgress

__all__ = [
    "BaseBisector",
    "BisectError",
    "BisectLogger",
    "BisectPhase",
    "BisectProgress",
    "BisectState",
    "CommandResult",
    "CommitDetector",
    "CommitDetectorError",
    "CommitPair",
    "LLVMBisectError",
    "LLVMBisector",
    "LLVMBumpInfo",
    "PairTester",
    "PairTesterError",
    "PairTestResult",
    "ShellExecutor",
    "StateManager",
    "TritonBisectError",
    "TritonBisector",
]
