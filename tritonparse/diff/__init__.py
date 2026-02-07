#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Diff module for comparing different compilation events of the same kernel.

This module provides functionality to compare two Triton compilation events
and identify differences in metadata, Python source, IR statistics, and
source mapping-based line-level comparisons.
"""

from tritonparse.diff.core.diff_types import (
    CompilationDiffResult,
    DiffNote,
    DiffSummary,
    IRStats,
    IRStatsDiff,
    MetadataDiff,
    OperationDiff,
    PythonLineDiff,
    PythonSourceDiff,
)

__all__ = [
    "CompilationDiffResult",
    "DiffNote",
    "DiffSummary",
    "IRStats",
    "IRStatsDiff",
    "MetadataDiff",
    "OperationDiff",
    "PythonLineDiff",
    "PythonSourceDiff",
]
