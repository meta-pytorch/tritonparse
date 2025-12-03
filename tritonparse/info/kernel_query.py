#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Core query functions for kernel information from NDJSON trace files.

This module provides functions to query kernel launch information from parsed
event lists. It supports both raw log files and parsed ndjson files (with launch_diff events).
"""

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
