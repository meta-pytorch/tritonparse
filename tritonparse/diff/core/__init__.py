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
    KernelMatchResult,
    MatchMethod,
    MetadataDiff,
    OperationDiff,
    PythonLineDiff,
    PythonSourceDiff,
    TensorArgDiff,
    TensorValueDiff,
    TraceDiffResult,
    TraceDiffSummary,
    TraceStats,
)
from tritonparse.diff.core.event_matcher import (
    find_launches_for_compilation,
    group_compilations_by_kernel,
    group_launches_by_kernel,
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
from tritonparse.diff.core.kernel_matcher import KernelMatcher
from tritonparse.diff.core.tensor_value_analyzer import (
    analyze_tensor_values,
    TensorValueAnalyzer,
)
from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine

__all__ = [
    # Types
    "CompilationDiffResult",
    "DiffNote",
    "DiffSummary",
    "IRStats",
    "IRStatsDiff",
    "KernelMatchResult",
    "MatchMethod",
    "MetadataDiff",
    "OperationDiff",
    "PythonLineDiff",
    "PythonSourceDiff",
    "TensorArgDiff",
    "TensorValueDiff",
    "TraceDiffResult",
    "TraceDiffSummary",
    "TraceStats",
    # Kernel Matcher
    "KernelMatcher",
    # Trace Diff Engine
    "TraceDiffEngine",
    # Event matcher
    "find_launches_for_compilation",
    "group_compilations_by_kernel",
    "group_launches_by_kernel",
    # Tensor Value Analyzer
    "analyze_tensor_values",
    "TensorValueAnalyzer",
    # IR Stats Analyzer
    "analyze_ir_stats",
    "analyze_operation_diff",
    "count_operations",
    "get_ir_content",
    "IR_TYPES",
    "IRStatsAnalyzer",
    "OP_PATTERNS",
]
