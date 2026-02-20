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
class TensorArgDiff:
    """Numeric comparison result for a single tensor argument.

    Uses a dict-based metrics field for extensibility — new metrics
    can be added without requiring schema changes.

    Attributes:
        arg_name: Name of the tensor argument (e.g., "x_ptr").
        status: Per-arg comparison status.
        shape_a: Shape of tensor A.
        shape_b: Shape of tensor B.
        dtype_a: Dtype string of tensor A (e.g., "torch.float32").
        dtype_b: Dtype string of tensor B.
        metrics: Dict of computed numeric metrics. Standard keys include:
            comparison_mode: "blob" for full tensor comparison or
                "stats" for inline statistics comparison.
            Blob mode keys: max_abs_error, mean_abs_error, max_rel_error,
                rmse, cosine_similarity, allclose, atol, rtol,
                nan_count_a, nan_count_b, inf_count_a, inf_count_b,
                num_mismatched_elements, mismatch_ratio.
            Stats mode keys: min_a, min_b, min_diff, max_a, max_b, max_diff,
                mean_a, mean_b, mean_diff, std_a, std_b, std_diff, atol.
            New metrics can be added here without schema changes.
        metadata_a: Full tensor metadata from extracted_args for inspection.
        metadata_b: Full tensor metadata from extracted_args for inspection.
    """

    arg_name: str = ""
    status: str = "skipped"
    shape_a: list[int] = field(default_factory=list)
    shape_b: list[int] = field(default_factory=list)
    dtype_a: str = ""
    dtype_b: str = ""
    metrics: dict[str, float | int | bool | str | None] = field(default_factory=dict)
    metadata_a: dict[str, Any] = field(default_factory=dict)
    metadata_b: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorValueDiff:
    """Level 5: Tensor value comparison result.

    Attributes:
        status: Overall status: "identical", "close", "divergent", "skipped".
        args_compared: Number of tensor args compared.
        args_identical: Number of args with status="identical".
        args_close: Number of args with status="close" (allclose=True).
        args_divergent: Number of args that diverge.
        atol: Absolute tolerance used.
        rtol: Relative tolerance used.
        warning: Warning message (e.g., "No tensor blobs found").
        per_arg: Per-argument comparison results keyed by arg name.
    """

    status: str = "skipped"
    args_compared: int = 0
    args_identical: int = 0
    args_close: int = 0
    args_divergent: int = 0
    atol: float = 1e-5
    rtol: float = 1e-3
    warning: str | None = None
    per_arg: dict[str, TensorArgDiff] = field(default_factory=dict)


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
        tensor_value_diff: Level 5 tensor value comparison.
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
    tensor_value_diff: TensorValueDiff = field(default_factory=TensorValueDiff)
