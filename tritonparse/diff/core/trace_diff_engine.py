#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Trace-level diff engine.

Orchestrates whole-trace comparison by matching kernels across two traces
using KernelMatcher and then running DiffEngine on each matched pair.
"""

import logging
import uuid
from typing import Any

from tritonparse.diff.core.diff_engine import DiffEngine
from tritonparse.diff.core.diff_types import (
    KernelMatchResult,
    TraceDiffResult,
    TraceDiffSummary,
    TraceStats,
)
from tritonparse.diff.core.event_matcher import (
    find_launch_for_compilation,
    find_launches_for_compilation,
    get_compilation_events,
    get_kernel_hash,
    get_kernel_name,
    group_compilations_by_kernel,
    group_launches_by_kernel,
)
from tritonparse.diff.core.kernel_matcher import KernelMatcher

logger = logging.getLogger(__name__)


class TraceDiffEngine:
    """Engine for comparing all kernels across two trace files.

    Orchestrates:
    1. Compute trace statistics
    2. Match kernels using multi-strategy KernelMatcher
    3. Diff each matched pair using DiffEngine
    4. Generate summary

    Args:
        events_a: All events from trace A.
        events_b: All events from trace B.
        trace_path_a: Path to trace file A.
        trace_path_b: Path to trace file B.
        tensor_values: Whether to compare tensor values.
        atol: Absolute tolerance for tensor comparison.
        rtol: Relative tolerance for tensor comparison.
    """

    def __init__(
        self,
        events_a: list[dict[str, Any]],
        events_b: list[dict[str, Any]],
        trace_path_a: str = "",
        trace_path_b: str = "",
        tensor_values: bool = False,
        atol: float = 1e-5,
        rtol: float = 1e-3,
    ) -> None:
        self.events_a = events_a
        self.events_b = events_b
        self.trace_path_a = trace_path_a
        self.trace_path_b = trace_path_b
        self.tensor_values = tensor_values
        self.atol = atol
        self.rtol = rtol

        self.result = TraceDiffResult()

        # Internal state built during run
        self._comp_groups_a: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self._comp_groups_b: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        self._launch_groups_a: dict[str, list[dict[str, Any]]] = {}
        self._launch_groups_b: dict[str, list[dict[str, Any]]] = {}

    def run(self) -> TraceDiffResult:
        """Execute the full trace diff pipeline.

        Returns:
            TraceDiffResult with all matched/unmatched kernels and summary.
        """
        self.result.diff_id = str(uuid.uuid4())
        self._compute_trace_stats()
        self._match_kernels()
        self._diff_matched_kernels()
        self._generate_summary()
        return self.result

    def _compute_trace_stats(self) -> None:
        """Compute statistics for both traces."""
        self._comp_groups_a = group_compilations_by_kernel(self.events_a)
        self._comp_groups_b = group_compilations_by_kernel(self.events_b)
        self._launch_groups_a = group_launches_by_kernel(self.events_a)
        self._launch_groups_b = group_launches_by_kernel(self.events_b)

        self.result.trace_a = self._build_trace_stats(
            self.trace_path_a,
            self.events_a,
            self._comp_groups_a,
            self._launch_groups_a,
        )
        self.result.trace_b = self._build_trace_stats(
            self.trace_path_b,
            self.events_b,
            self._comp_groups_b,
            self._launch_groups_b,
        )

    @staticmethod
    def _build_trace_stats(
        trace_path: str,
        events: list[dict[str, Any]],
        comp_groups: dict[str, list[tuple[int, dict[str, Any]]]],
        launch_groups: dict[str, list[dict[str, Any]]],
    ) -> TraceStats:
        """Build TraceStats for a single trace."""
        total_compilations = sum(len(v) for v in comp_groups.values())
        total_launches = sum(len(v) for v in launch_groups.values())
        kernel_names = sorted(comp_groups.keys())

        return TraceStats(
            trace_path=trace_path,
            total_events=len(events),
            unique_kernels=len(kernel_names),
            total_compilations=total_compilations,
            total_launches=total_launches,
            kernel_names=kernel_names,
        )

    def _match_kernels(self) -> None:
        """Match kernels across traces using KernelMatcher."""
        comp_events_a = get_compilation_events(self.events_a)
        comp_events_b = get_compilation_events(self.events_b)

        if not comp_events_a or not comp_events_b:
            # Populate only_in lists for non-empty side
            if comp_events_a:
                names = {get_kernel_name(e) for _, e in comp_events_a}
                self.result.only_in_a = sorted(names)
            if comp_events_b:
                names = {get_kernel_name(e) for _, e in comp_events_b}
                self.result.only_in_b = sorted(names)
            return

        matcher = KernelMatcher(comp_events_a, comp_events_b)
        matched, unmatched_a, unmatched_b, extra_a, extra_b = matcher.match()

        self.result.matched_kernels = matched
        # only_in = truly absent kernels (from groups with no cross-trace match)
        self.result.only_in_a = sorted({get_kernel_name(e) for _, e in unmatched_a})
        self.result.only_in_b = sorted({get_kernel_name(e) for _, e in unmatched_b})
        # extra = unpaired autotuning compilations from matched groups
        self.result.extra_compilations_a = len(extra_a)
        self.result.extra_compilations_b = len(extra_b)

    def _select_representative(
        self,
        compilations: list[tuple[int, dict[str, Any]]],
        events: list[dict[str, Any]],
    ) -> tuple[int, dict[str, Any]]:
        """Select the representative compilation for a kernel.

        Picks the compilation whose hash has the most launches
        (the autotuning winner). Falls back to the first compilation.

        Args:
            compilations: List of (index, event) tuples for this kernel.
            events: All events (for finding launches).

        Returns:
            The (index, event) tuple for the representative compilation.
        """
        if len(compilations) == 1:
            return compilations[0]

        best_idx = 0
        best_launch_count = -1

        for i, (_, event) in enumerate(compilations):
            kernel_hash = get_kernel_hash(event)
            launches = find_launches_for_compilation(events, kernel_hash)
            if len(launches) > best_launch_count:
                best_launch_count = len(launches)
                best_idx = i

        return compilations[best_idx]

    def _diff_matched_kernels(self) -> None:
        """Run DiffEngine on each matched kernel pair."""
        for match_result in self.result.matched_kernels:
            self._diff_single_match(match_result)

    def _diff_single_match(self, match_result: KernelMatchResult) -> None:
        """Run DiffEngine for a single matched pair and update the result."""
        # Find the events by index
        comp_a = self._find_event_by_index(self.events_a, match_result.event_index_a)
        comp_b = self._find_event_by_index(self.events_b, match_result.event_index_b)

        if comp_a is None or comp_b is None:
            logger.warning(
                "Could not find compilation events for match: %s / %s",
                match_result.kernel_name_a,
                match_result.kernel_name_b,
            )
            return

        # Find launch events for tensor comparison
        hash_a = get_kernel_hash(comp_a)
        hash_b = get_kernel_hash(comp_b)
        launch_a = find_launch_for_compilation(self.events_a, comp_a, hash_a)
        launch_b = find_launch_for_compilation(self.events_b, comp_b, hash_b)

        # Count launches
        match_result.launch_count_a = len(
            find_launches_for_compilation(self.events_a, hash_a)
        )
        match_result.launch_count_b = len(
            find_launches_for_compilation(self.events_b, hash_b)
        )

        # Run per-kernel diff
        engine = DiffEngine(
            comp_a,
            comp_b,
            source_trace_a=self.trace_path_a,
            source_trace_b=self.trace_path_b,
            event_index_a=match_result.event_index_a,
            event_index_b=match_result.event_index_b,
            launch_a=launch_a,
            launch_b=launch_b,
            tensor_values=self.tensor_values,
            atol=self.atol,
            rtol=self.rtol,
        )
        comp_diff = engine.run()

        # Stamp matching info on the compilation diff result
        comp_diff.match_method = match_result.match_method.value
        comp_diff.match_confidence = match_result.match_confidence

        match_result.compilation_diff = comp_diff
        match_result.source_similarity = comp_diff.python_source_diff.similarity

        # Determine status from the compilation diff
        if comp_diff.summary.status == "identical":
            match_result.status = "identical"
        elif comp_diff.summary.status == "minor_diff":
            match_result.status = "similar"
        else:
            match_result.status = "different"

        # Collect metadata changes
        for key, diff in comp_diff.metadata_diff.diffs.items():
            match_result.metadata_changes.append(f"{key}: {diff['a']} → {diff['b']}")

        # Collect IR stat highlights
        for ir_type, ir_diff in comp_diff.ir_stats.items():
            if ir_diff.line_diff != 0:
                sign = "+" if ir_diff.line_diff > 0 else ""
                match_result.ir_stat_highlights.append(
                    f"{ir_type}: {sign}{ir_diff.line_diff} lines "
                    f"({sign}{ir_diff.line_diff_pct:.0f}%)"
                )

    @staticmethod
    def _find_event_by_index(
        events: list[dict[str, Any]], comp_index: int
    ) -> dict[str, Any] | None:
        """Find a compilation event by its compilation index."""
        comp_events = get_compilation_events(events)
        for idx, event in comp_events:
            if idx == comp_index:
                return event
        return None

    def _generate_summary(self) -> None:
        """Generate the trace diff summary."""
        identical = sum(
            1 for m in self.result.matched_kernels if m.status == "identical"
        )
        similar = sum(1 for m in self.result.matched_kernels if m.status == "similar")
        different = sum(
            1 for m in self.result.matched_kernels if m.status == "different"
        )

        # Count matches per strategy
        match_stats: dict[str, int] = {}
        for m in self.result.matched_kernels:
            method_val = m.match_method.value
            match_stats[method_val] = match_stats.get(method_val, 0) + 1

        # Determine overall status
        if (
            different == 0
            and len(self.result.only_in_a) == 0
            and len(self.result.only_in_b) == 0
            and similar == 0
        ):
            status = "identical"
        elif (
            different == 0
            and len(self.result.only_in_a) == 0
            and len(self.result.only_in_b) == 0
        ):
            status = "minor_diff"
        else:
            status = "significant_diff"

        # Build highlights
        highlights: list[str] = []
        if different > 0:
            highlights.append(f"{different} kernel(s) have significant differences")
        if len(self.result.only_in_a) > 0:
            highlights.append(f"{len(self.result.only_in_a)} kernel(s) only in trace A")
        if len(self.result.only_in_b) > 0:
            highlights.append(f"{len(self.result.only_in_b)} kernel(s) only in trace B")
        if self.result.extra_compilations_a > 0:
            highlights.append(
                f"{self.result.extra_compilations_a} extra autotuning "
                f"compilation(s) in trace A"
            )
        if self.result.extra_compilations_b > 0:
            highlights.append(
                f"{self.result.extra_compilations_b} extra autotuning "
                f"compilation(s) in trace B"
            )

        # Collect tensor divergent kernels
        tensor_divergent: list[str] = []
        for m in self.result.matched_kernels:
            if (
                m.compilation_diff
                and m.compilation_diff.tensor_value_diff.status == "divergent"
            ):
                tensor_divergent.append(m.kernel_name_a)

        self.result.summary = TraceDiffSummary(
            status=status,
            total_matched=len(self.result.matched_kernels),
            identical=identical,
            similar=similar,
            different=different,
            only_a=len(self.result.only_in_a),
            only_b=len(self.result.only_in_b),
            extra_compilations_a=self.result.extra_compilations_a,
            extra_compilations_b=self.result.extra_compilations_b,
            highlights=highlights,
            match_stats=match_stats,
            tensor_divergent_kernels=tensor_divergent,
        )
