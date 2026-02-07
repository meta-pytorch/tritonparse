#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Data types for the diff module.

This module defines all dataclass structures used for representing
comparison results between two compilation events.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetadataDiff:
    """Level 1: Metadata comparison result.

    Compares compilation configuration parameters like num_stages,
    num_warps, cluster_dims, target arch, shared memory usage, etc.

    Attributes:
        sames: Dictionary of metadata keys with identical values in both compilations.
        diffs: Dictionary of metadata keys with different values.
               Format: {key: {"a": value_a, "b": value_b}}
    """

    sames: dict[str, Any] = field(default_factory=dict)
    diffs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class PythonSourceDiff:
    """Level 2: Python source comparison result.

    Compares the Python source code of two compilations.
    Phase 1 only supports cases where Python source is identical.

    Attributes:
        status: "identical" or "different"
        similarity: Float between 0 and 1 indicating source similarity.
        hunks: List of diff hunks from difflib unified_diff.
    """

    status: str = "identical"
    similarity: float = 1.0
    hunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IRStats:
    """Statistics for a single IR type.

    Attributes:
        lines: Total number of lines in the IR.
        ops: Total number of operations in the IR.
        op_counts: Dictionary mapping operation names to their counts.
    """

    lines: int = 0
    ops: int = 0
    op_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class IRStatsDiff:
    """Level 3a: IR statistics comparison for one IR type.

    Attributes:
        a: IR statistics for compilation A.
        b: IR statistics for compilation B.
        line_diff: Difference in line count (b - a).
        line_diff_pct: Percentage change in line count.
    """

    a: IRStats = field(default_factory=IRStats)
    b: IRStats = field(default_factory=IRStats)
    line_diff: int = 0
    line_diff_pct: float = 0.0


@dataclass
class OperationDiff:
    """Level 3b: Operation-level diff for one IR type.

    Attributes:
        added: List of operation types added in compilation B.
        removed: List of operation types removed from compilation A.
        counts_a: Operation counts for compilation A.
        counts_b: Operation counts for compilation B.
    """

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    counts_a: dict[str, int] = field(default_factory=dict)
    counts_b: dict[str, int] = field(default_factory=dict)


@dataclass
class PythonLineDiff:
    """Level 4: IR diff organized by Python source line.

    Key insight: Organizes IR differences by Python source line,
    letting users see "the same line of Python code compiled to different IR lines".

    Attributes:
        python_line: The Python source line number.
        python_code: The Python source code at this line.
        a: IR line numbers for compilation A. Format: {ir_type: [line_numbers]}
        b: IR line numbers for compilation B. Format: {ir_type: [line_numbers]}
        a_code: IR code lines for compilation A (Phase 2).
        b_code: IR code lines for compilation B (Phase 2).
        expansion: Line count difference by IR type. Format: {ir_type: diff}
        pattern: Detected pattern like "pipelined_load" (Phase 2).
    """

    python_line: int = 0
    python_code: str = ""
    a: dict[str, list[int]] = field(default_factory=dict)
    b: dict[str, list[int]] = field(default_factory=dict)
    a_code: dict[str, list[str]] = field(default_factory=dict)
    b_code: dict[str, list[str]] = field(default_factory=dict)
    expansion: dict[str, int] = field(default_factory=dict)
    pattern: str | None = None


@dataclass
class DiffNote:
    """A note in the summary from rule-based or AI analysis.

    Attributes:
        source: "rule" for rule-based analysis, "ai" for AI-generated.
        category: "performance", "correctness", "analysis", or "warning".
        content: The note content.
        model: AI model name if source == "ai".
        confidence: Confidence score if source == "ai".
    """

    source: str = "rule"
    category: str = "analysis"
    content: str = ""
    model: str | None = None
    confidence: float | None = None


@dataclass
class DiffSummary:
    """Summary of the diff result.

    Attributes:
        status: "identical", "minor_diff", or "significant_diff".
        warning: Optional warning message.
        highlights: List of key differences as human-readable strings.
        stats: Dictionary of statistics (python_lines_compared, lines_with_ir_diff, etc.)
        notes: List of DiffNote objects with detailed analysis.
    """

    status: str = "identical"
    warning: str | None = None
    highlights: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    notes: list[DiffNote] = field(default_factory=list)


@dataclass
class CompilationDiffResult:
    """Complete diff result between two compilation events.

    This is the main output structure containing all comparison results
    organized by level.

    Attributes:
        diff_id: Unique identifier for this diff.
        kernel_name_a: Kernel name from compilation A.
        kernel_name_b: Kernel name from compilation B.
        hash_a: Kernel hash from compilation A.
        hash_b: Kernel hash from compilation B.
        kernel_names_identical: Whether kernel names match.
        source_trace_a: Path to source trace file A.
        source_trace_b: Path to source trace file B.
        event_index_a: Event index in source trace A.
        event_index_b: Event index in source trace B.
        summary: Overall diff summary.
        metadata_diff: Level 1 metadata comparison.
        python_source_diff: Level 2 Python source comparison.
        ir_stats: Level 3a IR statistics by type.
        operation_diff: Level 3b operation-level diff by IR type.
        by_python_line: Level 4 source mapping-based comparison.
    """

    # Identifiers
    diff_id: str = ""
    kernel_name_a: str = ""
    kernel_name_b: str = ""
    kernel_names_identical: bool = False
    hash_a: str = ""
    hash_b: str = ""
    source_trace_a: str = ""
    source_trace_b: str = ""
    event_index_a: int = 0
    event_index_b: int = 0

    # Diff results by level
    summary: DiffSummary = field(default_factory=DiffSummary)
    metadata_diff: MetadataDiff = field(default_factory=MetadataDiff)
    python_source_diff: PythonSourceDiff = field(default_factory=PythonSourceDiff)
    ir_stats: dict[str, IRStatsDiff] = field(default_factory=dict)
    operation_diff: dict[str, OperationDiff] = field(default_factory=dict)
    by_python_line: dict[int, PythonLineDiff] = field(default_factory=dict)
