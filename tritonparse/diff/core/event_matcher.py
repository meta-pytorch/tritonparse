#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Event matching utilities for the diff module.

This module provides functionality to load compilation events from ndjson files,
ensure they have source_mappings, and match events by index or kernel name.
"""

import json
from typing import Any

from tritonparse.tools.prettify_ndjson import load_ndjson


def load_events(input_path: str) -> list[dict[str, Any]]:
    """Load events from an ndjson file.

    Supports uncompressed (.ndjson), gzip compressed (.ndjson.gz),
    and gzip member concatenation (.bin.ndjson) formats.

    Args:
        input_path: Path to the ndjson file.

    Returns:
        List of event dictionaries.
    """
    return load_ndjson(input_path)


def is_compilation_event(event: dict[str, Any]) -> bool:
    """Check if an event is a compilation event.

    Args:
        event: Event dictionary.

    Returns:
        True if the event is a compilation event.
    """
    return event.get("event_type") == "compilation"


def get_compilation_events(
    events: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    """Extract all compilation events with their original indices.

    Args:
        events: List of all events.

    Returns:
        List of (original_index, event) tuples for compilation events.
    """
    return [(i, event) for i, event in enumerate(events) if is_compilation_event(event)]


def has_source_mappings(event: dict[str, Any]) -> bool:
    """Check if a compilation event has source_mappings.

    Source mappings are required for the diff module to perform
    line-level IR comparisons.

    Args:
        event: Compilation event dictionary.

    Returns:
        True if the event has non-empty source_mappings.
    """
    payload = event.get("payload", {})
    return bool(payload.get("source_mappings"))


def ensure_source_mappings(event: dict[str, Any]) -> dict[str, Any]:
    """Ensure a compilation event has source_mappings.

    If the event already has source_mappings, returns it unchanged.
    Otherwise, runs the parse module to generate source_mappings.

    Args:
        event: Compilation event dictionary.

    Returns:
        Event dictionary with source_mappings populated.
    """
    if has_source_mappings(event):
        return event

    # Reuse parse module function to generate source_mappings
    from tritonparse.parse.trace_processor import parse_single_trace_content

    event_str = json.dumps(event)
    parsed_str = parse_single_trace_content(event_str)
    return json.loads(parsed_str.strip())


def match_events_by_index(
    events: list[dict[str, Any]], index_a: int, index_b: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Get two compilation events by their compilation indices.

    The indices are relative to compilation events only, not all events.
    For example, if a file has [launch, compilation, launch, compilation],
    index 0 refers to the first compilation and index 1 to the second.

    Args:
        events: List of all events.
        index_a: Index of first compilation event.
        index_b: Index of second compilation event.

    Returns:
        Tuple of (event_a, event_b).

    Raises:
        ValueError: If either index is out of range.
    """
    compilations = get_compilation_events(events)

    if index_a >= len(compilations):
        raise ValueError(
            f"Event index {index_a} out of range. "
            f"Only {len(compilations)} compilation events found."
        )
    if index_b >= len(compilations):
        raise ValueError(
            f"Event index {index_b} out of range. "
            f"Only {len(compilations)} compilation events found."
        )

    _, event_a = compilations[index_a]
    _, event_b = compilations[index_b]
    return event_a, event_b


def match_events_by_kernel(
    events: list[dict[str, Any]], kernel_name: str
) -> list[tuple[int, dict[str, Any]]]:
    """Filter compilation events by kernel name.

    Args:
        events: List of all events.
        kernel_name: Kernel name to filter by.

    Returns:
        List of (compilation_index, event) tuples matching the kernel name.
    """
    compilations = get_compilation_events(events)
    matches = []

    for comp_idx, (_, event) in enumerate(compilations):
        # Try different places where kernel name might be stored
        event_kernel_name = event.get("kernel_name")
        if event_kernel_name is None:
            # Also check in payload.metadata.name
            event_kernel_name = event.get("payload", {}).get("metadata", {}).get("name")

        if event_kernel_name == kernel_name:
            matches.append((comp_idx, event))

    return matches


def list_available_compilations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """List all available compilations with summary info.

    Useful for CLI to show users what compilation events are available
    for comparison.

    Args:
        events: List of all events.

    Returns:
        List of summary dictionaries with keys:
        - index: Compilation index (for use with --events)
        - original_index: Line index in the ndjson file
        - kernel_name: Name of the kernel
        - kernel_hash: First 12 chars of kernel hash
        - num_stages: Pipeline stages (if available)
        - num_warps: Number of warps (if available)
        - has_source_mappings: Whether source_mappings exist
    """
    compilations = get_compilation_events(events)
    result = []

    for comp_idx, (orig_idx, event) in enumerate(compilations):
        payload = event.get("payload", {})
        metadata = payload.get("metadata", {})
        compilation_metadata = event.get("compilation_metadata", {})

        # Get kernel name from various possible locations
        kernel_name = event.get("kernel_name") or metadata.get("name") or "unknown"

        # Get kernel hash
        kernel_hash = (
            event.get("kernel_hash")
            or metadata.get("hash")
            or compilation_metadata.get("hash")
            or ""
        )

        # Get compilation parameters
        num_stages = compilation_metadata.get("num_stages") or metadata.get(
            "num_stages"
        )
        num_warps = compilation_metadata.get("num_warps") or metadata.get("num_warps")

        result.append(
            {
                "index": comp_idx,
                "original_index": orig_idx,
                "kernel_name": kernel_name,
                "kernel_hash": kernel_hash[:12] if kernel_hash else "",
                "num_stages": num_stages,
                "num_warps": num_warps,
                "has_source_mappings": has_source_mappings(event),
            }
        )

    return result


def get_kernel_name(event: dict[str, Any]) -> str:
    """Extract kernel name from a compilation event.

    Args:
        event: Compilation event dictionary.

    Returns:
        Kernel name or "unknown" if not found.
    """
    kernel_name = event.get("kernel_name")
    if kernel_name:
        return kernel_name

    payload = event.get("payload", {})
    metadata = payload.get("metadata", {})
    return metadata.get("name", "unknown")


def get_kernel_hash(event: dict[str, Any]) -> str:
    """Extract kernel hash from a compilation event.

    Args:
        event: Compilation event dictionary.

    Returns:
        Kernel hash or empty string if not found.
    """
    kernel_hash = event.get("kernel_hash")
    if kernel_hash:
        return kernel_hash

    payload = event.get("payload", {})
    metadata = payload.get("metadata", {})
    compilation_metadata = event.get("compilation_metadata", {})

    return metadata.get("hash") or compilation_metadata.get("hash") or ""
