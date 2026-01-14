# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pair tester for Triton/LLVM commit pairs.

This module provides the PairTester class which implements Phase 3 of the
Triton/LLVM bisect workflow. It tests (Triton, LLVM) commit pairs sequentially
to find the first failing pair.
"""

from dataclasses import dataclass
from typing import Optional


class PairTesterError(Exception):
    """Exception raised for pair testing errors."""

    pass


@dataclass
class CommitPair:
    """
    A (Triton, LLVM) commit pair.

    Attributes:
        triton_commit: The Triton commit hash.
        llvm_commit: The LLVM commit hash.
        index: The pair index (0-based) in the CSV file.
    """

    triton_commit: str
    llvm_commit: str
    index: int


@dataclass
class PairTestResult:
    """
    Result of pair testing.

    Attributes:
        found_failing: Whether a failing pair was found.
        failing_index: Index of the first failing pair (0-based), or -1 if none.
        good_llvm: LLVM commit from the last passing pair (for bisect).
        bad_llvm: LLVM commit from the first failing pair (for bisect).
        triton_commit: Triton commit of the failing pair.
        total_pairs: Total number of pairs tested.
        all_passed: True if all pairs passed (no failing pair found).
        error_message: Error message if testing failed.
    """

    found_failing: bool
    failing_index: int = -1
    good_llvm: Optional[str] = None
    bad_llvm: Optional[str] = None
    triton_commit: Optional[str] = None
    total_pairs: int = 0
    all_passed: bool = False
    error_message: Optional[str] = None
