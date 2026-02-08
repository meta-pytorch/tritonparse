#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Core components for the diff module.
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
from tritonparse.diff.core.ir_stats_analyzer import (
    analyze_ir_stats,
    analyze_operation_diff,
    count_operations,
    get_ir_content,
    IR_TYPES,
    IRStatsAnalyzer,
    OP_PATTERNS,
)

__all__ = [
    # Types
    "CompilationDiffResult",
    "DiffNote",
    "DiffSummary",
    "IRStats",
    "IRStatsDiff",
    "MetadataDiff",
    "OperationDiff",
    "PythonLineDiff",
    "PythonSourceDiff",
    # IR Stats Analyzer
    "analyze_ir_stats",
    "analyze_operation_diff",
    "count_operations",
    "get_ir_content",
    "IR_TYPES",
    "IRStatsAnalyzer",
    "OP_PATTERNS",
]
