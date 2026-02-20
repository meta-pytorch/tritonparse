#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Event writer for the diff module.

This module provides functions to serialize CompilationDiffResult to JSON
and write diff events to ndjson files.

Output format: Consolidated ndjson with unique compilation events first,
followed by diff events that reference them by hash.
"""

import json
from dataclasses import asdict
from typing import Any

from tritonparse.diff.core.diff_types import CompilationDiffResult, PythonLineDiff
from tritonparse.diff.core.event_matcher import get_kernel_hash


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


def append_diff_to_file(file_path: str, diff_event: dict[str, Any]) -> None:
    """Append diff event to existing ndjson file.

    This is used for the --in-place mode where the diff event is appended
    to the original input file instead of creating a new output file.

    Args:
        file_path: Path to the ndjson file.
        diff_event: The compilation_diff event to append.
    """
    with open(file_path, "a") as f:
        f.write(json.dumps(diff_event) + "\n")


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
                f.write(json.dumps(event) + "\n")
            # Then write all diff events
            for diff in self._diffs:
                f.write(json.dumps(diff) + "\n")

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
