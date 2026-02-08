#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Main diff engine that coordinates all analyzers.

The DiffEngine is the central orchestrator for comparing two compilation events.
It coordinates multiple analyzers (metadata, source, IR stats, source mapping)
and generates a comprehensive diff result.

Per the design document, this module delegates to specialized analyzer modules
where available:
- Level 1: MetadataAnalyzer (metadata_analyzer.py)
- Level 2: SourceAnalyzer (source_analyzer.py)
- Level 3: IR stats comparison (inline, to be extracted in D5)
- Level 4: Source mapping based comparison (inline)
- Summary: Summary generation (inline, to be extracted in D6)
"""

import re
import uuid
from collections import Counter
from typing import Any

from tritonparse.diff.core.diff_types import (
    CompilationDiffResult,
    DiffSummary,
    IRStats,
    IRStatsDiff,
    OperationDiff,
    PythonLineDiff,
)
from tritonparse.diff.core.event_matcher import (
    ensure_source_mappings,
    get_kernel_hash,
    get_kernel_name,
)
from tritonparse.diff.core.metadata_analyzer import MetadataAnalyzer
from tritonparse.diff.core.source_analyzer import SourceAnalyzer


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


class DiffEngine:
    """Main engine for comparing two compilation events.

    The engine orchestrates multi-level comparison:
    - Level 1: Metadata comparison
    - Level 2: Python source comparison
    - Level 3: IR statistics and operation comparison
    - Level 4: Source mapping-based line-level comparison

    Attributes:
        comp_a: First compilation event (with source_mappings ensured).
        comp_b: Second compilation event (with source_mappings ensured).
        source_trace_a: Path to source trace file for event A.
        source_trace_b: Path to source trace file for event B.
        event_index_a: Index of event A in its source trace.
        event_index_b: Index of event B in its source trace.
        result: The CompilationDiffResult being built.
    """

    def __init__(
        self,
        comp_a: dict[str, Any],
        comp_b: dict[str, Any],
        source_trace_a: str = "",
        source_trace_b: str = "",
        event_index_a: int = 0,
        event_index_b: int = 0,
    ):
        """Initialize the diff engine with two compilation events.

        Args:
            comp_a: First compilation event.
            comp_b: Second compilation event.
            source_trace_a: Path to source trace file for event A.
            source_trace_b: Path to source trace file for event B.
            event_index_a: Index of event A in its source trace.
            event_index_b: Index of event B in its source trace.
        """
        self.comp_a = ensure_source_mappings(comp_a)
        self.comp_b = ensure_source_mappings(comp_b)
        self.source_trace_a = source_trace_a
        self.source_trace_b = source_trace_b
        self.event_index_a = event_index_a
        self.event_index_b = event_index_b
        self.result = CompilationDiffResult()

    def run(self) -> CompilationDiffResult:
        """Execute all diff analysis and return the result.

        This is the main entry point that coordinates all analyzers.

        Returns:
            Complete diff result containing all comparison data.
        """
        self._init_result()
        self._diff_metadata()
        self._diff_python_source()
        self._diff_ir_stats()
        self._diff_by_python_line()
        self._generate_summary()
        return self.result

    def _init_result(self) -> None:
        """Initialize result with basic identifiers."""
        self.result.diff_id = str(uuid.uuid4())
        self.result.kernel_name_a = get_kernel_name(self.comp_a)
        self.result.kernel_name_b = get_kernel_name(self.comp_b)
        self.result.kernel_names_identical = (
            self.result.kernel_name_a == self.result.kernel_name_b
        )
        self.result.hash_a = get_kernel_hash(self.comp_a)
        self.result.hash_b = get_kernel_hash(self.comp_b)
        self.result.source_trace_a = self.source_trace_a
        self.result.source_trace_b = self.source_trace_b
        self.result.event_index_a = self.event_index_a
        self.result.event_index_b = self.event_index_b

    def _diff_metadata(self) -> None:
        """Level 1: Compare compilation metadata."""
        analyzer = MetadataAnalyzer(self.comp_a, self.comp_b)
        self.result.metadata_diff = analyzer.analyze()

    def _diff_python_source(self) -> None:
        """Level 2: Compare Python source code."""
        analyzer = SourceAnalyzer(self.comp_a, self.comp_b)
        self.result.python_source_diff = analyzer.analyze()

    def _diff_ir_stats(self) -> None:
        """Level 3: Compare IR statistics and operations.

        Note: This will be extracted to ir_analyzer.py in D5.
        """
        self.result.ir_stats = self._analyze_ir_stats()
        self.result.operation_diff = self._analyze_operation_diff()

    def _analyze_ir_stats(self) -> dict[str, IRStatsDiff]:
        """Analyze IR statistics for all IR types."""
        result: dict[str, IRStatsDiff] = {}

        for ir_type in IR_TYPES:
            content_a = self._get_ir_content(self.comp_a, ir_type)
            content_b = self._get_ir_content(self.comp_b, ir_type)

            if not content_a and not content_b:
                continue

            stats_a = self._compute_ir_stats(content_a, ir_type)
            stats_b = self._compute_ir_stats(content_b, ir_type)

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

    def _get_ir_content(self, comp: dict[str, Any], ir_type: str) -> str:
        """Extract IR content from a compilation event."""
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

    def _compute_ir_stats(self, content: str, ir_type: str) -> IRStats:
        """Compute statistics for IR content."""
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

    def _analyze_operation_diff(self) -> dict[str, OperationDiff]:
        """Analyze operation-level differences for all IR types."""
        result: dict[str, OperationDiff] = {}

        for ir_type, ir_stats in self.result.ir_stats.items():
            ops_a = set(ir_stats.a.op_counts.keys())
            ops_b = set(ir_stats.b.op_counts.keys())

            added = sorted(ops_b - ops_a)
            removed = sorted(ops_a - ops_b)

            result[ir_type] = OperationDiff(
                added=added,
                removed=removed,
                counts_a=ir_stats.a.op_counts,
                counts_b=ir_stats.b.op_counts,
            )

        return result

    def _diff_by_python_line(self) -> None:
        """Level 4: Compare IR organized by Python source line."""
        # Only perform line-level comparison if Python sources are identical
        if self.result.python_source_diff.status != "identical":
            return

        mappings_a = self.comp_a.get("payload", {}).get("source_mappings", {})
        mappings_b = self.comp_b.get("payload", {}).get("source_mappings", {})

        python_a = mappings_a.get("python", {})
        python_b = mappings_b.get("python", {})

        # Get all Python line numbers from both compilations
        all_lines: set[int] = set()
        all_lines.update(int(k) for k in python_a.keys() if k.isdigit())
        all_lines.update(int(k) for k in python_b.keys() if k.isdigit())

        result: dict[int, PythonLineDiff] = {}

        for py_line in sorted(all_lines):
            a_mapping: dict[str, list[int]] = {}
            b_mapping: dict[str, list[int]] = {}
            expansion: dict[str, int] = {}

            for ir_type in IR_TYPES:
                lines_a = self._get_ir_lines_for_python_line(
                    mappings_a, py_line, ir_type
                )
                lines_b = self._get_ir_lines_for_python_line(
                    mappings_b, py_line, ir_type
                )

                if lines_a or lines_b:
                    a_mapping[ir_type] = lines_a
                    b_mapping[ir_type] = lines_b
                    expansion[ir_type] = len(lines_b) - len(lines_a)

            # Get Python code for this line
            python_code = self._get_python_line_code(py_line)

            result[py_line] = PythonLineDiff(
                python_line=py_line,
                python_code=python_code,
                a=a_mapping,
                b=b_mapping,
                expansion=expansion,
            )

        self.result.by_python_line = result

    def _get_ir_lines_for_python_line(
        self, source_mappings: dict[str, Any], python_line: int, ir_type: str
    ) -> list[int]:
        """Get IR line numbers corresponding to a Python line."""
        python_mappings = source_mappings.get("python", {})
        line_mapping = python_mappings.get(str(python_line), {})
        ir_lines = line_mapping.get(f"{ir_type}_lines", [])
        return [int(ln) if isinstance(ln, str) else ln for ln in ir_lines]

    def _get_python_line_code(self, line_number: int) -> str:
        """Get the Python source code at a specific line number."""
        payload = self.comp_a.get("payload", {})

        python_source = payload.get("python_source", {})
        if isinstance(python_source, dict) and "content" in python_source:
            source = python_source["content"]
            start_line = python_source.get("start_line", 1)
        else:
            source = payload.get("python", "")
            start_line = 1

        if not source:
            return ""

        lines = source.splitlines()
        idx = line_number - start_line
        if 0 <= idx < len(lines):
            return lines[idx]

        return ""

    def _generate_summary(self) -> None:
        """Generate summary from all diff results.

        Note: This will be extracted to summary_generator.py in D6.
        """
        highlights: list[str] = []
        stats: dict[str, int] = {}

        # Add metadata differences to highlights
        for key, diff in self.result.metadata_diff.diffs.items():
            highlights.append(f"{key}: {diff['a']} → {diff['b']}")

        # Add significant IR stat changes to highlights
        for ir_type, ir_diff in self.result.ir_stats.items():
            if abs(ir_diff.line_diff_pct) > 20:
                sign = "+" if ir_diff.line_diff_pct > 0 else ""
                highlights.append(
                    f"{ir_type.upper()}: {sign}{ir_diff.line_diff_pct:.0f}%"
                )

        # Add new operations to highlights
        for ir_type, op_diff in self.result.operation_diff.items():
            if op_diff.added:
                ops_str = ", ".join(op_diff.added[:3])
                if len(op_diff.added) > 3:
                    ops_str += f" (+{len(op_diff.added) - 3} more)"
                highlights.append(f"New in {ir_type.upper()}: {ops_str}")

        # Compute stats
        stats["python_lines_compared"] = len(self.result.by_python_line)
        stats["lines_with_ir_diff"] = sum(
            1
            for diff in self.result.by_python_line.values()
            if any(v != 0 for v in diff.expansion.values())
        )
        stats["total_ir_line_expansion"] = sum(
            sum(diff.expansion.values()) for diff in self.result.by_python_line.values()
        )

        # Determine status
        if not highlights:
            status = "identical"
        elif len(highlights) <= 2:
            status = "minor_diff"
        else:
            status = "significant_diff"

        # Check for warning conditions
        warning = None
        if self.result.python_source_diff.status != "identical":
            warning = "Python sources differ; line-level comparison skipped"

        self.result.summary = DiffSummary(
            status=status,
            warning=warning,
            highlights=highlights,
            stats=stats,
            notes=[],
        )
