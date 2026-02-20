#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Main diff engine that coordinates all analyzers.

The DiffEngine is the central orchestrator for comparing two compilation events.
It coordinates multiple analyzers (metadata, source, IR stats, source mapping,
tensor values) and generates a comprehensive diff result.

Per the design document, this module delegates to specialized analyzer modules:
- Level 1: MetadataAnalyzer (metadata_analyzer.py)
- Level 2: SourceAnalyzer (source_analyzer.py)
- Level 3: IRStatsAnalyzer (ir_stats_analyzer.py)
- Level 4: SourcemapAnalyzer (sourcemap_analyzer.py)
- Level 5: TensorValueAnalyzer (tensor_value_analyzer.py)
- Summary: Summary generation (inline, to be extracted in D7)
"""

import uuid
from typing import Any

from tritonparse.diff.core.diff_types import (
    CompilationDiffResult,
    DiffSummary,
    TensorValueDiff,
)
from tritonparse.diff.core.event_matcher import (
    ensure_source_mappings,
    get_kernel_hash,
    get_kernel_name,
)
from tritonparse.diff.core.ir_stats_analyzer import IRStatsAnalyzer
from tritonparse.diff.core.metadata_analyzer import MetadataAnalyzer
from tritonparse.diff.core.source_analyzer import SourceAnalyzer
from tritonparse.diff.core.sourcemap_analyzer import SourcemapAnalyzer
from tritonparse.diff.core.tensor_value_analyzer import TensorValueAnalyzer


class DiffEngine:
    """Main engine for comparing two compilation events.

    The engine orchestrates multi-level comparison:
    - Level 1: Metadata comparison
    - Level 2: Python source comparison
    - Level 3: IR statistics and operation comparison
    - Level 4: Source mapping-based line-level comparison
    - Level 5: Tensor value comparison (optional, requires launch events)

    Attributes:
        comp_a: First compilation event (with source_mappings ensured).
        comp_b: Second compilation event (with source_mappings ensured).
        source_trace_a: Path to source trace file for event A.
        source_trace_b: Path to source trace file for event B.
        event_index_a: Index of event A in its source trace.
        event_index_b: Index of event B in its source trace.
        launch_a: Launch event for compilation A (optional).
        launch_b: Launch event for compilation B (optional).
        tensor_values: Whether to perform tensor value comparison.
        atol: Absolute tolerance for tensor comparison.
        rtol: Relative tolerance for tensor comparison.
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
        launch_a: dict[str, Any] | None = None,
        launch_b: dict[str, Any] | None = None,
        tensor_values: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-3,
    ):
        """Initialize the diff engine with two compilation events.

        Args:
            comp_a: First compilation event.
            comp_b: Second compilation event.
            source_trace_a: Path to source trace file for event A.
            source_trace_b: Path to source trace file for event B.
            event_index_a: Index of event A in its source trace.
            event_index_b: Index of event B in its source trace.
            launch_a: Launch event for compilation A (optional).
            launch_b: Launch event for compilation B (optional).
            tensor_values: Whether to perform tensor value comparison.
            atol: Absolute tolerance for tensor comparison.
            rtol: Relative tolerance for tensor comparison.
        """
        self.comp_a = ensure_source_mappings(comp_a)
        self.comp_b = ensure_source_mappings(comp_b)
        self.source_trace_a = source_trace_a
        self.source_trace_b = source_trace_b
        self.event_index_a = event_index_a
        self.event_index_b = event_index_b
        self.launch_a = launch_a
        self.launch_b = launch_b
        self.tensor_values = tensor_values
        self.atol = atol
        self.rtol = rtol
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
        self._diff_tensor_values()
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

        Delegates to IRStatsAnalyzer for the analysis.
        """
        analyzer = IRStatsAnalyzer(self.comp_a, self.comp_b)
        self.result.ir_stats, self.result.operation_diff = analyzer.analyze()

    def _diff_by_python_line(self) -> None:
        """Level 4: Compare IR organized by Python source line.

        Delegates to SourcemapAnalyzer for the actual analysis.
        Only performs comparison if Python sources are identical.
        """
        # Only perform line-level comparison if Python sources are identical
        if self.result.python_source_diff.status != "identical":
            return

        analyzer = SourcemapAnalyzer(self.comp_a, self.comp_b)
        self.result.by_python_line = analyzer.analyze()

    def _diff_tensor_values(self) -> None:
        """Level 5: Compare tensor values between launch events.

        Only runs when tensor_values=True. Requires launch events
        to be provided; skips with a warning if they are missing.
        """
        if not self.tensor_values:
            return

        if self.launch_a is None or self.launch_b is None:
            self.result.tensor_value_diff = TensorValueDiff(
                status="skipped",
                warning="No launch events found for tensor value comparison",
            )
            return

        analyzer = TensorValueAnalyzer(
            self.launch_a, self.launch_b, atol=self.atol, rtol=self.rtol
        )
        self.result.tensor_value_diff = analyzer.analyze()

    def _get_tensor_value_highlights(self) -> list[str]:
        """Get highlight strings for tensor value divergences."""
        highlights: list[str] = []
        tv = self.result.tensor_value_diff
        if tv.status == "divergent":
            for arg_name, arg_diff in tv.per_arg.items():
                if arg_diff.status == "divergent":
                    mode = arg_diff.metrics.get("comparison_mode", "blob")
                    if mode == "stats":
                        mean_diff = arg_diff.metrics.get("mean_diff")
                        if mean_diff is not None:
                            highlights.append(
                                f"Tensor '{arg_name}': "
                                f"mean_diff={mean_diff:.2e} (stats-based)"
                            )
                    else:
                        max_err = arg_diff.metrics.get("max_abs_error")
                        if max_err is not None:
                            highlights.append(
                                f"Tensor '{arg_name}': max_abs_error={max_err:.2e}"
                            )
        return highlights

    def _generate_summary(self) -> None:
        """Generate summary from all diff results."""
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

        # Add tensor value divergence highlights
        highlights.extend(self._get_tensor_value_highlights())

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
