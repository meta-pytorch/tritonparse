#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
IR statistics analyzer for the diff module.

This module provides Level 3a comparison of IR statistics
(line counts, operation counts, operation types) between two compilation events.

Per the design document Section 3.3:
- Computes statistics for each IR level (TTIR, TTGIR, LLIR, PTX/AMDGCN)
- Counts occurrences of each operation type
- Identifies added/removed operation types between compilations
"""

import re
from collections import Counter
from typing import Any

from tritonparse.diff.core.diff_types import IRStats, IRStatsDiff, OperationDiff


# IR types to analyze
IR_TYPES = ["ttir", "ttgir", "llir", "ptx", "amdgcn"]

# Operation patterns for different IR types
OP_PATTERNS = {
    "ttir": r"(tt\.\w+|arith\.\w+|scf\.\w+|cf\.\w+)",
    "ttgir": r"(ttg\.\w+|tt\.\w+|arith\.\w+|scf\.\w+|cf\.\w+)",
    "llir": r"(define|call|load|store|alloca|getelementptr|br|ret|icmp|fcmp|add|sub|mul|div|and|or|xor|shl|lshr|ashr)",
    "ptx": r"(ld\.|st\.|mov\.|add\.|sub\.|mul\.|mad\.|fma\.|cvt\.|setp\.|bra|ret|bar\.sync|cp\.async)",
    "amdgcn": r"(v_\w+|s_\w+|ds_\w+|buffer_\w+|global_\w+|flat_\w+)",
}


class IRStatsAnalyzer:
    """Analyzer for Level 3 IR statistics comparison.

    Computes and compares IR statistics between two compilation events,
    including line counts, operation counts, and operation type differences.

    Attributes:
        comp_a: First compilation event.
        comp_b: Second compilation event.
        ir_types: List of IR types to analyze.
    """

    def __init__(
        self,
        comp_a: dict[str, Any],
        comp_b: dict[str, Any],
        ir_types: list[str] | None = None,
    ):
        """Initialize the IR stats analyzer.

        Args:
            comp_a: First compilation event.
            comp_b: Second compilation event.
            ir_types: Optional list of IR types to analyze.
                      Defaults to IR_TYPES if not provided.
        """
        self.comp_a = comp_a
        self.comp_b = comp_b
        self.ir_types = ir_types or IR_TYPES

    def analyze(self) -> tuple[dict[str, IRStatsDiff], dict[str, OperationDiff]]:
        """Analyze IR statistics and operation differences.

        Returns:
            Tuple of (ir_stats, operation_diff) dictionaries.
        """
        ir_stats = self._analyze_ir_stats()
        operation_diff = self._analyze_operation_diff(ir_stats)
        return ir_stats, operation_diff

    def _analyze_ir_stats(self) -> dict[str, IRStatsDiff]:
        """Analyze IR statistics for all IR types."""
        result: dict[str, IRStatsDiff] = {}

        for ir_type in self.ir_types:
            content_a = get_ir_content(self.comp_a, ir_type)
            content_b = get_ir_content(self.comp_b, ir_type)

            if not content_a and not content_b:
                continue

            stats_a = count_operations(content_a, ir_type)
            stats_b = count_operations(content_b, ir_type)

            line_diff = stats_b.lines - stats_a.lines
            line_diff_pct = (
                (line_diff / stats_a.lines * 100) if stats_a.lines > 0 else 0.0
            )

            result[ir_type] = IRStatsDiff(
                a=stats_a,
                b=stats_b,
                line_diff=line_diff,
                line_diff_pct=line_diff_pct,
            )

        return result

    def _analyze_operation_diff(
        self, ir_stats: dict[str, IRStatsDiff]
    ) -> dict[str, OperationDiff]:
        """Analyze operation-level differences for all IR types."""
        result: dict[str, OperationDiff] = {}

        for ir_type, stats_diff in ir_stats.items():
            ops_a = set(stats_diff.a.op_counts.keys())
            ops_b = set(stats_diff.b.op_counts.keys())

            added = sorted(ops_b - ops_a)
            removed = sorted(ops_a - ops_b)

            result[ir_type] = OperationDiff(
                added=added,
                removed=removed,
                counts_a=stats_diff.a.op_counts,
                counts_b=stats_diff.b.op_counts,
            )

        return result


def get_ir_content(comp: dict[str, Any], ir_type: str) -> str:
    """Extract IR content from a compilation event.

    Tries multiple possible locations:
    1. payload.{ir_type} (direct field)
    2. payload.file_content.*.{ir_type} (file content)

    Args:
        comp: Compilation event dictionary.
        ir_type: IR type (ttir, ttgir, llir, ptx, amdgcn).

    Returns:
        IR content string, or empty string if not found.
    """
    payload = comp.get("payload", {})

    # Try direct field first
    content = payload.get(ir_type, "")
    if content:
        return content

    # Try file_content
    file_content = payload.get("file_content", {})
    for key, value in file_content.items():
        if key.endswith(f".{ir_type}"):
            return value

    return ""


def count_operations(content: str, ir_type: str) -> IRStats:
    """Count operations in IR content.

    Parses the IR content and counts occurrences of each operation type
    using the pattern for the specified IR type.

    Args:
        content: IR content string.
        ir_type: IR type (ttir, ttgir, llir, ptx, amdgcn).

    Returns:
        IRStats with line count, operation count, and operation counts.
    """
    if not content:
        return IRStats()

    lines = content.splitlines()
    line_count = len(lines)

    # Count operations
    pattern = OP_PATTERNS.get(ir_type)
    if pattern:
        operations: list[str] = []
        for line in lines:
            operations.extend(re.findall(pattern, line))
        op_counts = dict(Counter(operations))
        ops_count = len(operations)
    else:
        op_counts = {}
        ops_count = 0

    return IRStats(lines=line_count, ops=ops_count, op_counts=op_counts)


def analyze_ir_stats(
    comp_a: dict[str, Any],
    comp_b: dict[str, Any],
    ir_types: list[str] | None = None,
) -> dict[str, IRStatsDiff]:
    """Convenience function to analyze IR statistics differences.

    Args:
        comp_a: First compilation event.
        comp_b: Second compilation event.
        ir_types: Optional list of IR types to analyze.

    Returns:
        Dictionary mapping IR type to IRStatsDiff.
    """
    analyzer = IRStatsAnalyzer(comp_a, comp_b, ir_types)
    ir_stats, _ = analyzer.analyze()
    return ir_stats


def analyze_operation_diff(
    comp_a: dict[str, Any],
    comp_b: dict[str, Any],
    ir_types: list[str] | None = None,
) -> dict[str, OperationDiff]:
    """Convenience function to analyze operation differences.

    Args:
        comp_a: First compilation event.
        comp_b: Second compilation event.
        ir_types: Optional list of IR types to analyze.

    Returns:
        Dictionary mapping IR type to OperationDiff.
    """
    analyzer = IRStatsAnalyzer(comp_a, comp_b, ir_types)
    _, operation_diff = analyzer.analyze()
    return operation_diff
