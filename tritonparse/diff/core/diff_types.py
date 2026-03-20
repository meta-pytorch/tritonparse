#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Data types for the diff module.

This module defines all dataclass structures used for representing
comparison results between two compilation events.
"""

from dataclasses import dataclass, field
from enum import Enum
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
class DtypeMismatch:
    """A detected dtype mismatch between tensor args across two traces.

    Used when tensor argument names don't overlap, so we compare
    the set of unique dtypes across all tensor args on each side.

    Attributes:
        dtypes_a: Unique dtypes found in trace A tensor args.
        dtypes_b: Unique dtypes found in trace B tensor args.
        description: Human-readable description of the mismatch.
    """

    dtypes_a: list[str] = field(default_factory=list)
    dtypes_b: list[str] = field(default_factory=list)
    description: str = ""


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
        dtype_inventory_a: Maps arg name -> dtype for ALL tensor args in trace A.
        dtype_inventory_b: Maps arg name -> dtype for ALL tensor args in trace B.
        dtype_mismatches: Detected cross-side dtype mismatches (when names don't match).
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
    dtype_inventory_a: dict[str, str] = field(default_factory=dict)
    dtype_inventory_b: dict[str, str] = field(default_factory=dict)
    dtype_mismatches: list[DtypeMismatch] = field(default_factory=list)


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

    # Trace-mode matching info (None for manual/single-pair diffs)
    match_method: str | None = None
    match_confidence: float | None = None


class MatchMethod(str, Enum):
    """How a kernel pair was matched across traces."""

    HASH = "hash"
    NAME = "name"
    SOURCE = "source"
    FUZZY_NAME = "fuzzy_name"
    CONFIG = "config"


@dataclass
class KernelMatchResult:
    """Result of matching and diffing a single kernel pair across two traces.

    Attributes:
        kernel_name_a: Kernel name from trace A.
        kernel_name_b: Kernel name from trace B.
        hash_a: Kernel hash from trace A.
        hash_b: Kernel hash from trace B.
        event_index_a: Compilation event index in trace A.
        event_index_b: Compilation event index in trace B.
        match_method: Which strategy produced this match.
        match_confidence: Confidence score (1.0 for exact, similarity ratio for fuzzy).
        status: "identical", "similar", or "different".
        compilation_diff: Full per-kernel diff result (filled after matching).
        launch_count_a: Number of launches in trace A.
        launch_count_b: Number of launches in trace B.
        source_similarity: Python source similarity ratio.
        metadata_changes: List of metadata change descriptions.
        ir_stat_highlights: List of IR stat change descriptions.
        tensor_summary: Summary of tensor value comparison, if any.
    """

    kernel_name_a: str = ""
    kernel_name_b: str = ""
    hash_a: str = ""
    hash_b: str = ""
    event_index_a: int = 0
    event_index_b: int = 0
    match_method: MatchMethod = MatchMethod.NAME
    match_confidence: float = 1.0
    status: str = ""
    compilation_diff: CompilationDiffResult | None = None
    launch_count_a: int = 0
    launch_count_b: int = 0
    source_similarity: float = 0.0
    metadata_changes: list[str] = field(default_factory=list)
    ir_stat_highlights: list[str] = field(default_factory=list)
    tensor_summary: str | None = None


@dataclass
class TraceStats:
    """Statistics for a single trace file.

    Attributes:
        trace_path: Path to the trace file.
        total_events: Total number of events in the trace.
        unique_kernels: Number of unique kernel names.
        total_compilations: Total number of compilation events.
        total_launches: Total number of launch events.
        kernel_names: List of unique kernel names.
    """

    trace_path: str = ""
    total_events: int = 0
    unique_kernels: int = 0
    total_compilations: int = 0
    total_launches: int = 0
    kernel_names: list[str] = field(default_factory=list)


@dataclass
class TraceDiffSummary:
    """Summary of the trace-level diff.

    Attributes:
        status: Overall status: "identical", "minor_diff", or "significant_diff".
        total_matched: Number of matched kernel pairs.
        identical: Number of identical kernel pairs.
        similar: Number of similar kernel pairs.
        different: Number of different kernel pairs.
        only_a: Number of kernels only in trace A.
        only_b: Number of kernels only in trace B.
        extra_compilations_a: Unpaired autotuning compilations in trace A.
        extra_compilations_b: Unpaired autotuning compilations in trace B.
        highlights: List of key differences as human-readable strings.
        match_stats: Count of matches per strategy. Format: {method_value: count}.
        tensor_divergent_kernels: Kernel names with divergent tensor values.
    """

    status: str = "identical"
    total_matched: int = 0
    identical: int = 0
    similar: int = 0
    different: int = 0
    only_a: int = 0
    only_b: int = 0
    extra_compilations_a: int = 0
    extra_compilations_b: int = 0
    highlights: list[str] = field(default_factory=list)
    match_stats: dict[str, int] = field(default_factory=dict)
    tensor_divergent_kernels: list[str] = field(default_factory=list)


@dataclass
class TraceDiffResult:
    """Complete diff result between two trace files.

    Attributes:
        diff_id: Unique identifier for this trace diff.
        trace_a: Statistics for trace A.
        trace_b: Statistics for trace B.
        matched_kernels: List of matched kernel pair results.
        only_in_a: Kernel names only present in trace A (truly absent).
        only_in_b: Kernel names only present in trace B (truly absent).
        extra_compilations_a: Unpaired autotuning compilations in trace A.
        extra_compilations_b: Unpaired autotuning compilations in trace B.
        summary: Overall trace diff summary.
    """

    diff_id: str = ""
    trace_a: TraceStats = field(default_factory=TraceStats)
    trace_b: TraceStats = field(default_factory=TraceStats)
    matched_kernels: list[KernelMatchResult] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    extra_compilations_a: int = 0
    extra_compilations_b: int = 0
    summary: TraceDiffSummary = field(default_factory=TraceDiffSummary)
