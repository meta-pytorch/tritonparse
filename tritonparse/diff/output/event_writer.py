#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Event writer for the diff module.

This module provides functions to serialize CompilationDiffResult to JSON
and write diff events to ndjson files.

Output format: Consolidated ndjson with unique compilation events first,
followed by diff events that reference them by hash.
"""

import math
from dataclasses import asdict
from typing import Any

from tritonparse._json_compat import dumps
from tritonparse.diff.core.diff_types import (
    CompilationDiffResult,
    PythonLineDiff,
    TraceDiffResult,
)
from tritonparse.diff.core.event_matcher import get_kernel_hash


def _sanitize_non_finite_floats(obj: Any) -> Any:
    """Replace NaN/Inf/-Inf floats with None for JSON serialization.

    orjson raises TypeError on non-finite floats, unlike stdlib json which
    outputs non-standard NaN/Infinity tokens. This function ensures safe
    serialization by converting non-finite floats to None.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_non_finite_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_non_finite_floats(v) for v in obj]
    return obj


def _get_event_hash(event: dict[str, Any]) -> str:
    """Extract hash from a compilation event.

    Uses the same logic as get_kernel_hash from event_matcher to find
    the hash in nested metadata structures.

    Args:
        event: A compilation event dictionary.

    Returns:
        The kernel hash, or empty string if not found.
    """
    return get_kernel_hash(event)


def _convert_by_python_line(
    by_python_line: dict[int, PythonLineDiff],
) -> dict[str, dict[str, Any]]:
    """Convert by_python_line dict with int keys to str keys for JSON.

    JSON does not support integer keys, so we must convert them to strings.

    Args:
        by_python_line: Dictionary mapping Python line numbers to PythonLineDiff.

    Returns:
        Dictionary with string keys suitable for JSON serialization.
    """
    return {str(k): asdict(v) for k, v in by_python_line.items()}


def create_diff_event(result: CompilationDiffResult) -> dict[str, Any]:
    """Create a compilation_diff event from the diff result.

    This function converts the CompilationDiffResult dataclass into a
    JSON-serializable dictionary representing a compilation_diff event.

    Args:
        result: The diff result from DiffEngine.run()

    Returns:
        A dictionary representing the compilation_diff event for ndjson.
    """
    return {
        "event_type": "compilation_diff",
        "diff_id": result.diff_id,
        "kernel_name_a": result.kernel_name_a,
        "kernel_name_b": result.kernel_name_b,
        "kernel_names_identical": result.kernel_names_identical,
        "hash_a": result.hash_a,
        "hash_b": result.hash_b,
        "source_trace_a": result.source_trace_a,
        "source_trace_b": result.source_trace_b,
        "event_index_a": result.event_index_a,
        "event_index_b": result.event_index_b,
        "summary": asdict(result.summary),
        "metadata_diff": asdict(result.metadata_diff),
        "python_source_diff": asdict(result.python_source_diff),
        "ir_stats": {k: asdict(v) for k, v in result.ir_stats.items()},
        "operation_diff": {k: asdict(v) for k, v in result.operation_diff.items()},
        "by_python_line": _convert_by_python_line(result.by_python_line),
        "tensor_value_diff": asdict(result.tensor_value_diff),
    }


def create_trace_diff_event(result: TraceDiffResult) -> dict[str, Any]:
    """Create a trace_diff event from the trace diff result.

    Serializes the TraceDiffResult into a JSON-serializable dictionary.
    For each matched kernel, includes the compilation diff without
    by_python_line to control output size.

    Args:
        result: The trace diff result from TraceDiffEngine.run().

    Returns:
        A dictionary representing the trace_diff event for ndjson.
    """
    from dataclasses import asdict

    matched_kernels = []
    for match in result.matched_kernels:
        kernel_data: dict[str, Any] = {
            "kernel_name_a": match.kernel_name_a,
            "kernel_name_b": match.kernel_name_b,
            "hash_a": match.hash_a,
            "hash_b": match.hash_b,
            "event_index_a": match.event_index_a,
            "event_index_b": match.event_index_b,
            "match_method": match.match_method.value
            if hasattr(match.match_method, "value")
            else str(match.match_method),
            "match_confidence": match.match_confidence,
            "status": match.status,
            "launch_count_a": match.launch_count_a,
            "launch_count_b": match.launch_count_b,
            "source_similarity": match.source_similarity,
            "metadata_changes": match.metadata_changes,
            "ir_stat_highlights": match.ir_stat_highlights,
            "tensor_summary": match.tensor_summary,
        }
        if match.compilation_diff:
            # Include compilation diff without by_python_line to save space
            comp_diff = create_diff_event(match.compilation_diff)
            comp_diff.pop("by_python_line", None)
            kernel_data["compilation_diff"] = comp_diff

        matched_kernels.append(kernel_data)

    return {
        "event_type": "trace_diff",
        "diff_id": result.diff_id,
        "trace_a": asdict(result.trace_a),
        "trace_b": asdict(result.trace_b),
        "matched_kernels": matched_kernels,
        "only_in_a": result.only_in_a,
        "only_in_b": result.only_in_b,
        "extra_compilations_a": result.extra_compilations_a,
        "extra_compilations_b": result.extra_compilations_b,
        "summary": asdict(result.summary),
    }


def append_diff_to_file(file_path: str, diff_event: dict[str, Any]) -> None:
    """Append diff event to existing ndjson file.

    This is used for the --in-place mode where the diff event is appended
    to the original input file instead of creating a new output file.

    Args:
        file_path: Path to the ndjson file.
        diff_event: The compilation_diff event to append.
    """
    with open(file_path, "a") as f:
        f.write(dumps(_sanitize_non_finite_floats(diff_event)) + "\n")


class ConsolidatedDiffWriter:
    """Writer for consolidated diff output with unique events first, then diffs.

    This class collects unique compilation events and diff results, then writes
    them to a single ndjson file with the following structure:
    - All unique compilation events (deduplicated by hash)
    - All compilation_diff events (referencing events by hash)

    This format is more efficient than one-file-per-diff when comparing many
    events, as each compilation event is stored only once regardless of how
    many diffs reference it.

    Usage:
        writer = ConsolidatedDiffWriter()
        writer.add_diff(result, comp_a, comp_b)
        writer.add_diff(result2, comp_b, comp_c)
        writer.write("output.ndjson")

    Or as a context manager:
        with ConsolidatedDiffWriter("output.ndjson") as writer:
            writer.add_diff(result, comp_a, comp_b)
    """

    def __init__(self, output_path: str | None = None) -> None:
        """Initialize the writer.

        Args:
            output_path: Optional path for context manager usage. If provided,
                the file will be written automatically on context exit.
        """
        self._output_path = output_path
        self._events: dict[str, dict[str, Any]] = {}  # hash -> event
        self._diffs: list[dict[str, Any]] = []

    def add_event(self, event: dict[str, Any]) -> None:
        """Add a compilation event if not already present.

        Args:
            event: A compilation event dictionary.
        """
        event_hash = _get_event_hash(event)
        if event_hash and event_hash not in self._events:
            self._events[event_hash] = event

    def add_diff(
        self,
        result: CompilationDiffResult,
        comp_a: dict[str, Any],
        comp_b: dict[str, Any],
    ) -> None:
        """Add a diff result and its associated compilation events.

        Args:
            result: The diff result from DiffEngine.run()
            comp_a: Compilation event A.
            comp_b: Compilation event B.
        """
        self.add_event(comp_a)
        self.add_event(comp_b)
        self._diffs.append(create_diff_event(result))

    def add_trace_diff(
        self,
        result: TraceDiffResult,
        events_a: list[dict[str, Any]],
        events_b: list[dict[str, Any]],
    ) -> None:
        """Add a trace diff result and its associated compilation events.

        Adds all unique compilation events from both traces, then adds
        the trace_diff event.

        Args:
            result: The trace diff result from TraceDiffEngine.run().
            events_a: All events from trace A.
            events_b: All events from trace B.
        """
        # Add all unique compilation events
        for event in events_a:
            if event.get("event_type") == "compilation":
                self.add_event(event)
        for event in events_b:
            if event.get("event_type") == "compilation":
                self.add_event(event)

        self._diffs.append(create_trace_diff_event(result))

    def add_diff_event(self, diff_event: dict[str, Any]) -> None:
        """Add a pre-created diff event.

        Use this when you already have a serialized diff event dict.

        Args:
            diff_event: A compilation_diff event dictionary.
        """
        self._diffs.append(diff_event)

    def write(self, output_path: str | None = None) -> None:
        """Write all events and diffs to the output file.

        Events are written first (in insertion order), then diffs.

        Args:
            output_path: Path to write to. Uses the path from __init__ if not
                provided.

        Raises:
            ValueError: If no output path is provided.
        """
        path = output_path or self._output_path
        if not path:
            raise ValueError("No output path provided")

        with open(path, "w") as f:
            # Write all unique compilation events first
            for event in self._events.values():
                f.write(dumps(_sanitize_non_finite_floats(event)) + "\n")
            # Then write all diff events
            for diff in self._diffs:
                f.write(dumps(_sanitize_non_finite_floats(diff)) + "\n")

    @property
    def event_count(self) -> int:
        """Number of unique compilation events collected."""
        return len(self._events)

    @property
    def diff_count(self) -> int:
        """Number of diff events collected."""
        return len(self._diffs)

    def __enter__(self) -> "ConsolidatedDiffWriter":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, writing output if no exception occurred."""
        if exc_type is None and self._output_path:
            self.write()


def write_consolidated_output(
    output_path: str,
    diffs: list[tuple[CompilationDiffResult, dict[str, Any], dict[str, Any]]],
) -> int:
    """Write multiple diffs to a single consolidated file.

    This is a convenience function for batch processing. For streaming usage,
    use ConsolidatedDiffWriter directly.

    Args:
        output_path: Path to the output ndjson file.
        diffs: List of tuples (diff_result, comp_a, comp_b).

    Returns:
        Number of unique compilation events written.
    """
    writer = ConsolidatedDiffWriter()
    for result, comp_a, comp_b in diffs:
        writer.add_diff(result, comp_a, comp_b)
    writer.write(output_path)
    return writer.event_count
