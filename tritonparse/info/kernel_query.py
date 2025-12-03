#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Core query functions for kernel information from NDJSON trace files.

This module provides functions to query kernel launch information from parsed
event lists. It supports both raw log files and parsed ndjson files (with launch_diff events).
"""

import difflib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class KernelSummary:
    """Summary information about a kernel."""

    name: str
    hash: str
    total_launches: int


@dataclass
class LaunchInfo:
    """Information about a specific kernel launch."""

    launch_id: int  # 0-based
    line_index: int  # 0-based (index in events list)
    grid: List[int]


def list_kernels(events: List[Dict[str, Any]]) -> List[KernelSummary]:
    """
    List all kernels with their launch counts.

    Args:
        events: List of parsed event dictionaries from NDJSON file

    Returns:
        List of KernelSummary objects, sorted by kernel name
    """
    # Count launches per kernel
    kernel_counts: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"hash": "", "count": 0}
    )

    for event in events:
        if event.get("event_type") != "launch":
            continue

        comp_meta = event.get("compilation_metadata", {})
        kernel_name = comp_meta.get("name")
        kernel_hash = comp_meta.get("hash", "")

        if kernel_name:
            kernel_counts[kernel_name]["hash"] = kernel_hash
            kernel_counts[kernel_name]["count"] += 1

    # Convert to KernelSummary list
    summaries = [
        KernelSummary(name=name, hash=info["hash"], total_launches=info["count"])
        for name, info in kernel_counts.items()
    ]

    # Sort by kernel name for consistent output
    summaries.sort(key=lambda x: x.name)

    return summaries


def find_launch_index_by_kernel(
    events: List[Dict[str, Any]], kernel_name: str, launch_id: int
) -> int:
    """
    Find the 0-based line index for a kernel's N-th launch.

    Args:
        events: List of parsed event dictionaries
        kernel_name: Exact kernel name to match (case-sensitive)
        launch_id: 0-based launch index for the kernel

    Returns:
        0-based line index (index in events list)

    Raises:
        ValueError: If kernel not found or launch_id out of range
    """
    count = 0
    for i, event in enumerate(events):
        if event.get("event_type") != "launch":
            continue

        comp_meta = event.get("compilation_metadata", {})
        name = comp_meta.get("name")
        if name == kernel_name:
            if count == launch_id:
                return i
            count += 1

    if count == 0:
        raise ValueError(f"Kernel '{kernel_name}' not found")
    else:
        raise ValueError(
            f"Kernel '{kernel_name}' has only {count} launches, "
            f"but --launch-id {launch_id} was requested. Valid range: 0 to {count - 1}"
        )


def list_launches_for_kernel(
    events: List[Dict[str, Any]], kernel_name: str
) -> List[LaunchInfo]:
    """
    List all launches for a specific kernel.

    Args:
        events: List of parsed event dictionaries
        kernel_name: Exact kernel name to match (case-sensitive)

    Returns:
        List of LaunchInfo objects for the kernel, sorted by launch_id

    Raises:
        ValueError: If kernel not found
    """
    launches = []
    launch_id = 0

    for i, event in enumerate(events):
        if event.get("event_type") != "launch":
            continue

        comp_meta = event.get("compilation_metadata", {})
        name = comp_meta.get("name")
        if name == kernel_name:
            # Extract grid information from launch event
            grid = event.get("grid", [])
            launches.append(LaunchInfo(launch_id=launch_id, line_index=i, grid=grid))
            launch_id += 1

    if not launches:
        raise ValueError(f"Kernel '{kernel_name}' not found")

    return launches


def find_similar_kernels(
    events: List[Dict[str, Any]], kernel_name: str, n: int = 3
) -> List[str]:
    """
    Find similar kernel names using fuzzy matching.

    Args:
        events: List of parsed event dictionaries
        kernel_name: Kernel name to find similar matches for
        n: Maximum number of matches to return

    Returns:
        List of similar kernel names (may be empty if no matches found)
    """
    all_kernels = list_kernels(events)
    all_names = [k.name for k in all_kernels]
    return difflib.get_close_matches(kernel_name, all_names, n=n, cutoff=0.6)


def list_kernels_fast(events: List[Dict[str, Any]]) -> List[KernelSummary]:
    """
    Fast kernel listing using launch_diff events when available.

    If launch_diff events are present, uses them for fast listing.
    Otherwise, falls back to list_kernels().

    Args:
        events: List of parsed event dictionaries

    Returns:
        List of KernelSummary objects, sorted by kernel name
    """
    # Check if launch_diff events are available
    launch_diff_events = [e for e in events if e.get("event_type") == "launch_diff"]

    if launch_diff_events:
        # Use launch_diff events for fast listing
        # Merge kernels with the same name (sum up launches)
        kernel_dict: Dict[str, KernelSummary] = {}
        for event in launch_diff_events:
            name = event.get("name", "")
            if not name:
                continue
            hash_val = event.get("hash", "")
            launches = event.get("total_launches", 0)

            if name in kernel_dict:
                # Merge: sum up launches, keep first hash
                kernel_dict[name].total_launches += launches
            else:
                kernel_dict[name] = KernelSummary(
                    name=name,
                    hash=hash_val,
                    total_launches=launches,
                )

        summaries = list(kernel_dict.values())
        summaries.sort(key=lambda x: x.name)
        return summaries
    else:
        # Fall back to full traversal
        return list_kernels(events)
