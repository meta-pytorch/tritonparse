#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Diff module for comparing different compilation events of the same kernel.

This module provides functionality to compare two Triton compilation events
and identify differences in metadata, Python source, IR statistics, and
source mapping-based line-level comparisons.
"""

from tritonparse.diff.cli import _add_diff_args, diff_command
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
from tritonparse.diff.core.kernel_matcher import KernelMatcher
from tritonparse.diff.core.trace_diff_engine import TraceDiffEngine
from tritonparse.diff.output import (
    append_diff_to_file,
    ConsolidatedDiffWriter,
    create_diff_event,
    create_trace_diff_event,
    format_summary,
    format_trace_summary,
    write_consolidated_output,
)

try:
    from tritonparse.diff.fb.ai import (
        AIDiffAnalyzer,
        build_diff_context,
        DIFF_ANALYSIS_SYSTEM_PROMPT,
    )

    _HAS_AI = True
except ImportError:
    _HAS_AI = False

__all__ = [
    # CLI
    "_add_diff_args",
    "diff_command",
    # Data types
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
    # Engines
    "KernelMatcher",
    "TraceDiffEngine",
    # Output functions
    "append_diff_to_file",
    "ConsolidatedDiffWriter",
    "create_diff_event",
    "create_trace_diff_event",
    "format_summary",
    "format_trace_summary",
    "write_consolidated_output",
]

if _HAS_AI:
    __all__ += [
        "AIDiffAnalyzer",
        "build_diff_context",
        "DIFF_ANALYSIS_SYSTEM_PROMPT",
    ]
