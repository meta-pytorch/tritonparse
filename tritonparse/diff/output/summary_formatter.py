#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Summary formatter for the diff module.

This module provides functions to format CompilationDiffResult for
human-readable CLI output.
"""

from tritonparse.diff.core.diff_types import (
    CompilationDiffResult,
    DiffNote,
    DiffSummary,
    IRStatsDiff,
    PythonLineDiff,
)


def format_summary(result: CompilationDiffResult) -> str:
    """Format diff result for CLI display.

    This is the main entry point for formatting a diff result for display
    in the terminal. It combines all sections into a single formatted string.

    Args:
        result: The diff result from DiffEngine.run()

    Returns:
        A formatted string suitable for terminal output.
    """
    sections = [_format_header(result)]

    if result.summary.warning:
        sections.append(f"\nWarning: {result.summary.warning}")

    sections.append(_format_highlights(result.summary))
    sections.append(_format_ir_stats(result.ir_stats))
    sections.append(_format_python_line_summary(result.by_python_line))

    if result.summary.notes:
        sections.append(_format_notes(result.summary.notes))

    return "\n".join(s for s in sections if s)


def _format_header(result: CompilationDiffResult) -> str:
    """Format header with kernel name, status, and hashes.

    Args:
        result: The diff result.

    Returns:
        Formatted header string.
    """
    kernel_display = result.kernel_name_a
    if not result.kernel_names_identical:
        kernel_display = f"{result.kernel_name_a} vs {result.kernel_name_b}"

    # Handle case where hash might be empty or too short
    hash_a_display = (
        result.hash_a[:12] + "..." if len(result.hash_a) > 12 else result.hash_a
    )
    hash_b_display = (
        result.hash_b[:12] + "..." if len(result.hash_b) > 12 else result.hash_b
    )

    lines = [
        f"=== Compilation Diff: {kernel_display} ===",
        f"Status: {result.summary.status}",
        f"Hash A: {hash_a_display} (event #{result.event_index_a})",
        f"Hash B: {hash_b_display} (event #{result.event_index_b})",
    ]
    return "\n".join(lines)


def _format_highlights(summary: DiffSummary) -> str:
    """Format highlights as bullet points.

    Args:
        summary: The diff summary containing highlights.

    Returns:
        Formatted highlights string, or empty string if no highlights.
    """
    if not summary.highlights:
        return ""
    lines = ["\nHighlights:"]
    for h in summary.highlights:
        lines.append(f"  • {h}")
    return "\n".join(lines)


def _format_ir_stats(ir_stats: dict[str, IRStatsDiff]) -> str:
    """Format IR statistics table.

    Shows line count changes for each IR type that has differences.

    Args:
        ir_stats: Dictionary mapping IR type to statistics diff.

    Returns:
        Formatted IR statistics string, or empty string if no changes.
    """
    if not ir_stats:
        return ""

    lines = ["\nIR Statistics:"]
    for ir_type, stats in ir_stats.items():
        if stats.line_diff != 0:
            sign = "+" if stats.line_diff > 0 else ""
            lines.append(
                f"  {ir_type.upper()}: {stats.a.lines} → {stats.b.lines} "
                f"({sign}{stats.line_diff}, {sign}{stats.line_diff_pct:.1f}%)"
            )

    # Only return if we have actual content beyond the header
    return "\n".join(lines) if len(lines) > 1 else ""


def _format_python_line_summary(by_python_line: dict[int, PythonLineDiff]) -> str:
    """Format Python line diff summary.

    Shows the total number of lines with differences and lists the top 5
    lines with the biggest IR expansion (by total absolute change).

    Args:
        by_python_line: Dictionary mapping Python line numbers to diffs.

    Returns:
        Formatted Python line summary string, or empty string if no data.
    """
    if not by_python_line:
        return ""

    total = len(by_python_line)
    with_diff = sum(
        1 for d in by_python_line.values() if any(v != 0 for v in d.expansion.values())
    )

    lines = [f"\nPython Lines with IR Diff: {with_diff} / {total} total"]

    # Show top 5 lines with biggest expansion (by total absolute change)
    lines_with_diff = [
        (k, v)
        for k, v in by_python_line.items()
        if any(x != 0 for x in v.expansion.values())
    ]
    sorted_lines = sorted(
        lines_with_diff,
        key=lambda x: sum(abs(v) for v in x[1].expansion.values()),
        reverse=True,
    )[:5]

    for py_line, diff in sorted_lines:
        exp_parts = []
        for ir_type, change in diff.expansion.items():
            if change != 0:
                sign = "+" if change > 0 else ""
                exp_parts.append(f"{sign}{change} {ir_type}")
        pattern_str = f" ({diff.pattern})" if diff.pattern else ""
        lines.append(f"  Line {py_line}: {', '.join(exp_parts)}{pattern_str}")

    return "\n".join(lines)


def _format_notes(notes: list[DiffNote]) -> str:
    """Format notes section.

    Notes can come from rule-based analysis or AI analysis.

    Args:
        notes: List of DiffNote objects.

    Returns:
        Formatted notes string, or empty string if no notes.
    """
    if not notes:
        return ""
    lines = ["\nInsights:"]
    for note in notes:
        prefix = "[AI]" if note.source == "ai" else "[Rule]"
        lines.append(f"  {prefix} {note.content}")
    return "\n".join(lines)
