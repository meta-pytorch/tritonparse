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


def find_launches_for_compilation(
    events: list[dict[str, Any]], compilation_hash: str
) -> list[dict[str, Any]]:
    """Find launch events associated with a compilation hash.

    Launch events are linked to compilations via the
    compilation_metadata.hash field.

    Args:
        events: List of all events.
        compilation_hash: Hash of the compilation to find launches for.

    Returns:
        List of launch event dictionaries matching the hash.
    """
    if not compilation_hash:
        return []

    launches = []
    for event in events:
        if event.get("event_type") != "launch":
            continue
        event_hash = event.get("compilation_metadata", {}).get("hash", "")
        if event_hash == compilation_hash:
            launches.append(event)
    return launches


def find_launch_for_compilation(
    events: list[dict[str, Any]],
    compilation_event: dict[str, Any],
    compilation_hash: str,
) -> dict[str, Any] | None:
    """Find the launch event positionally associated with a compilation.

    When multiple compilations share the same hash (e.g., the same kernel
    compiled twice), this function pairs each compilation with the launch
    at the same ordinal position among all launches with that hash.
    For example, the 2nd compilation with hash X gets the 2nd launch
    with hash X.

    Args:
        events: List of all events.
        compilation_event: The specific compilation event to find a launch for.
        compilation_hash: Hash of the compilation.

    Returns:
        The matching launch event, or None if not found.
    """
    if not compilation_hash:
        return None

    # Find the ordinal position of this compilation among all
    # compilations with the same hash
    comp_position = 0
    found = False
    for event in events:
        if not is_compilation_event(event):
            continue
        event_hash = get_kernel_hash(event)
        if event_hash != compilation_hash:
            continue
        if event is compilation_event:
            found = True
            break
        comp_position += 1

    if not found:
        # Compilation not found by identity; fall back to first launch
        comp_position = 0

    # Find the launch at the same ordinal position
    launch_position = 0
    for event in events:
        if event.get("event_type") != "launch":
            continue
        event_hash = event.get("compilation_metadata", {}).get("hash", "")
        if event_hash != compilation_hash:
            continue
        if launch_position == comp_position:
            return event
        launch_position += 1

    return None
