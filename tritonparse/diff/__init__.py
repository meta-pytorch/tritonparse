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
    MetadataDiff,
    OperationDiff,
    PythonLineDiff,
    PythonSourceDiff,
    TensorArgDiff,
    TensorValueDiff,
)
from tritonparse.diff.output import (
    append_diff_to_file,
    ConsolidatedDiffWriter,
    create_diff_event,
    format_summary,
    write_consolidated_output,
)

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
    "MetadataDiff",
    "OperationDiff",
    "PythonLineDiff",
    "PythonSourceDiff",
    "TensorArgDiff",
    "TensorValueDiff",
    # Output functions
    "append_diff_to_file",
    "ConsolidatedDiffWriter",
    "create_diff_event",
    "format_summary",
    "write_consolidated_output",
]
