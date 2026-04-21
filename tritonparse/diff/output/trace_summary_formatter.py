#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Trace-level summary formatter for the diff module.

Formats TraceDiffResult for human-readable CLI output, showing
kernel matching results, per-kernel status, and only-in lists.
"""

from tritonparse.diff.core.diff_types import (
    KernelMatchResult,
    TraceDiffResult,
    TraceDiffSummary,
    TraceStats,
)
from tritonparse.diff.output.summary_formatter import (
    _format_highlights,
    _format_ir_stats,
    _format_python_line_summary,
    _format_tensor_value_diff,
)


def format_trace_summary(result: TraceDiffResult) -> str:
    """Format trace diff result for CLI display.

    Args:
        result: The trace diff result from TraceDiffEngine.run().

    Returns:
        A formatted string suitable for terminal output.
    """
    sections = [
        _format_trace_header(result.trace_a, result.trace_b),
        _format_kernel_matching_summary(result.summary, result.matched_kernels),
        _format_matched_kernels(result.matched_kernels),
        _format_only_in_lists(result.only_in_a, result.only_in_b),
        _format_autotuning_extras(
            result.extra_compilations_a, result.extra_compilations_b
        ),
        _format_trace_notes(result.summary),
    ]

    return "\n".join(s for s in sections if s)


def _format_trace_header(trace_a: TraceStats, trace_b: TraceStats) -> str:
    """Format the trace header with file paths and statistics."""
    path_a = trace_a.trace_path or "trace A"
    path_b = trace_b.trace_path or "trace B"

    lines = [
        "=== Trace Diff Summary ===",
        f"Trace A: {path_a} — {trace_a.unique_kernels} unique kernels, "
        f"{trace_a.total_launches} total launches",
        f"Trace B: {path_b} — {trace_b.unique_kernels} unique kernels, "
        f"{trace_b.total_launches} total launches",
    ]
    return "\n".join(lines)


def _format_kernel_matching_summary(
    summary: TraceDiffSummary,
    matched_kernels: list[KernelMatchResult],
) -> str:
    """Format the kernel matching summary section."""
    lines = [
        "",
        f"Kernel Matching: {summary.total_matched} matched, "
        f"{summary.only_a} only in A, {summary.only_b} only in B",
    ]

    # Match breakdown by strategy
    breakdown = _format_match_breakdown(summary.match_stats)
    if breakdown:
        lines.append(f"  {breakdown}")

    lines.append(f"\nStatus: {summary.status}")

    return "\n".join(lines)


def _format_match_breakdown(match_stats: dict[str, int]) -> str:
    """Format per-strategy match breakdown."""
    if not match_stats:
        return ""

    # Display order with human-readable labels
    label_map = {
        "hash": "By hash",
        "name": "By name",
        "source": "By source similarity",
        "fuzzy_name": "By fuzzy name",
        "config": "By config similarity",
    }

    parts = []
    for method, label in label_map.items():
        count = match_stats.get(method, 0)
        if count > 0:
            parts.append(f"{label}: {count}")

    return " | ".join(parts)


def _format_matched_kernels(matched_kernels: list[KernelMatchResult]) -> str:
    """Format the matched kernels list with status icons."""
    if not matched_kernels:
        return ""

    lines = ["\nMatched Kernels:"]
    for match in matched_kernels:
        lines.append(_format_single_match(match))

    return "\n".join(lines)


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent each line of text with the given prefix."""
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def _format_single_match(match: KernelMatchResult) -> str:
    """Format a single matched kernel entry with per-kernel details."""
    # Status icon
    icon_map = {"identical": "=", "similar": "~", "different": "!"}
    icon = icon_map.get(match.status, "?")

    # Kernel name display
    if match.kernel_name_a == match.kernel_name_b:
        name_display = match.kernel_name_a
    else:
        name_display = f"{match.kernel_name_a} \u2192 {match.kernel_name_b}"

    # Event indices
    event_info = f"events #{match.event_index_a} \u2194 #{match.event_index_b}"

    # Status description
    parts = [f"  [{icon}] {name_display}: {match.status} ({event_info})"]

    # Match method annotation for non-name matches
    method_val = (
        match.match_method.value
        if hasattr(match.match_method, "value")
        else str(match.match_method)
    )
    if method_val != "name":
        confidence_str = ""
        if match.match_confidence < 1.0:
            confidence_str = f" ({match.match_confidence:.0%})"
        parts[0] += f" [matched by {method_val}{confidence_str}]"

    # Launch count info
    if match.launch_count_a > 0 or match.launch_count_b > 0:
        if match.launch_count_a == match.launch_count_b:
            parts[0] += f", {match.launch_count_a} launches each"
        else:
            parts[0] += (
                f", launches {match.launch_count_a} \u2192 {match.launch_count_b}"
            )

    # Per-kernel compilation diff details for non-identical kernels
    if match.status != "identical" and match.compilation_diff:
        diff = match.compilation_diff
        detail_sections = [
            _format_highlights(diff.summary),
            _format_ir_stats(diff.ir_stats),
            _format_python_line_summary(diff.by_python_line),
        ]
        # Include tensor value diff if it has meaningful content
        tv = diff.tensor_value_diff
        if tv.status != "skipped" or tv.dtype_mismatches:
            detail_sections.append(_format_tensor_value_diff(tv))
        elif tv.warning:
            detail_sections.append(f"\nTensor Values: {tv.warning}")
        for section in detail_sections:
            if section:
                parts.append(_indent(section.lstrip("\n"), "    "))

        # Per-kernel AI/rule notes
        if diff.summary.notes:
            for note in diff.summary.notes:
                prefix = "[AI]" if note.source == "ai" else "[Rule]"
                parts.append(f"    {prefix} {note.content}")

    return "\n".join(parts)


def _format_only_in_lists(only_in_a: list[str], only_in_b: list[str]) -> str:
    """Format the only-in-A and only-in-B kernel lists (truly absent kernels)."""
    lines = []

    if only_in_a:
        lines.append(f"\nOnly in Trace A: {', '.join(only_in_a)}")

    if only_in_b:
        lines.append(f"Only in Trace B: {', '.join(only_in_b)}")

    return "\n".join(lines)


def _format_autotuning_extras(extra_a: int, extra_b: int) -> str:
    """Format autotuning extra compilations summary."""
    lines = []

    if extra_a > 0:
        lines.append(
            f"\nAutotuning extras in Trace A: "
            f"{extra_a} additional compilation(s) with no counterpart in B"
        )
    if extra_b > 0:
        lines.append(
            f"Autotuning extras in Trace B: "
            f"{extra_b} additional compilation(s) with no counterpart in A"
        )

    return "\n".join(lines)


def _format_trace_notes(summary: TraceDiffSummary) -> str:
    """Format trace-level notes (AI and rule-based insights)."""
    if not summary.notes:
        return ""

    lines = ["", "Insights:"]
    for note in summary.notes:
        prefix = "[AI]" if note.source == "ai" else "[Rule]"
        lines.append(f"  {prefix} {note.content}")

    return "\n".join(lines)
