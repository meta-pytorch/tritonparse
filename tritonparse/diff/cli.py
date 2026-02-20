#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the diff subcommand.

This module provides command-line interface for comparing different
compilation events of the same kernel.
"""

import argparse
import os
from typing import Optional

from tritonparse.diff.core.diff_engine import DiffEngine
from tritonparse.diff.core.event_matcher import (
    find_launch_for_compilation,
    get_compilation_events,
    get_kernel_hash,
    load_events,
    match_events_by_index,
    match_events_by_kernel,
)
from tritonparse.diff.output import (
    append_diff_to_file,
    ConsolidatedDiffWriter,
    create_diff_event,
    format_summary,
)
from tritonparse.info.kernel_query import list_compilations
from tritonparse.shared_vars import is_fbcode


def _add_diff_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the diff subcommand."""
    parser.add_argument(
        "input",
        nargs="+",
        help="Path(s) to ndjson file(s). One file for single-file mode, two for dual-file mode.",
    )
    parser.add_argument(
        "--events",
        "-e",
        type=str,
        default="0,1",
        help="Event indices to compare, comma-separated (default: 0,1)",
    )
    parser.add_argument(
        "--kernel",
        "-k",
        type=str,
        default=None,
        help="Filter by kernel name before selecting events",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: {input}_diff.ndjson)",
    )
    parser.add_argument(
        "--in-place",
        "-i",
        action="store_true",
        help="Append diff event to input file instead of creating new file",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        dest="list_compilations",
        help="List available compilations and exit",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress CLI summary output (only write to file)",
    )
    parser.add_argument(
        "--tensor-values",
        action="store_true",
        help=(
            "Compare tensor values between launch events. "
            "Best with TRITONPARSE_SAVE_TENSOR_BLOBS=1 (full comparison); "
            "also supports TRITONPARSE_MORE_TENSOR_INFORMATION=1 (stats comparison)"
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for tensor comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for tensor comparison (default: 1e-3)",
    )


def _parse_event_indices(events_str: str) -> tuple[int, int]:
    """Parse event indices from comma-separated string.

    Args:
        events_str: String like "0,1" or "2,5"

    Returns:
        Tuple of (index_a, index_b)

    Raises:
        ValueError: If format is invalid
    """
    parts = events_str.split(",")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --events format: '{events_str}'. Expected 'N,M' (e.g., '0,1')"
        )
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError as e:
        raise ValueError(
            f"Invalid --events format: '{events_str}'. Indices must be integers."
        ) from e


def _generate_output_path(input_path: str) -> str:
    """Generate default output path from input path.

    Args:
        input_path: Path to input file

    Returns:
        Output path with _diff suffix before extension
    """
    base, ext = os.path.splitext(input_path)
    if ext == ".gz":
        # Handle .ndjson.gz
        base2, ext2 = os.path.splitext(base)
        return f"{base2}_diff{ext2}{ext}"
    return f"{base}_diff{ext}"


def diff_command(
    input_paths: list[str],
    events: str = "0,1",
    kernel: Optional[str] = None,
    output: Optional[str] = None,
    in_place: bool = False,
    list_compilations_flag: bool = False,
    quiet: bool = False,
    skip_logger: bool = False,
    tensor_values: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> None:
    """
    Main function for the diff command.

    Args:
        input_paths: List of input ndjson file paths (1 or 2 files)
        events: Comma-separated event indices to compare (e.g., "0,1")
        kernel: Optional kernel name to filter by
        output: Optional output file path
        in_place: If True, append diff to input file
        list_compilations_flag: If True, list compilations and exit
        quiet: If True, suppress CLI output
        skip_logger: Whether to skip usage logging
        tensor_values: If True, compare tensor values between launch events
        atol: Absolute tolerance for tensor comparison
        rtol: Relative tolerance for tensor comparison
    """
    if not skip_logger and is_fbcode():
        from tritonparse.fb.utils import usage_report_logger

        usage_report_logger()

    # Validate input paths
    if len(input_paths) > 2:
        raise ValueError(
            "At most 2 input files allowed (single-file or dual-file mode)"
        )

    # Load events from first file
    input_path_a = input_paths[0]
    all_events_a = load_events(input_path_a)

    # Handle dual-file mode
    if len(input_paths) == 2:
        input_path_b = input_paths[1]
        all_events_b = load_events(input_path_b)
    else:
        input_path_b = input_path_a
        all_events_b = all_events_a

    # Filter by kernel name if specified
    if kernel:
        compilations_a = match_events_by_kernel(all_events_a, kernel)
        if len(input_paths) == 2:
            compilations_b = match_events_by_kernel(all_events_b, kernel)
        else:
            compilations_b = compilations_a

        if not compilations_a:
            raise ValueError(
                f"No compilations found for kernel '{kernel}' in {input_path_a}"
            )
        if len(input_paths) == 2 and not compilations_b:
            raise ValueError(
                f"No compilations found for kernel '{kernel}' in {input_path_b}"
            )

        # Use filtered events for listing
        events_for_listing = [comp for _, comp in compilations_a]
    else:
        events_for_listing = all_events_a

    # List compilations mode
    if list_compilations_flag:
        compilations = list_compilations(events_for_listing)
        if not compilations:
            print(f"No compilation events found in {input_path_a}")
            return

        print(f"\nCompilations in {input_path_a}:")
        if kernel:
            print(f"(Filtered by kernel: {kernel})")
        print("-" * 80)
        any_tensor_data = False
        for comp in compilations:
            stages = comp.num_stages if comp.num_stages is not None else "?"
            warps = comp.num_warps if comp.num_warps is not None else "?"
            has_map = "✓" if comp.has_source_mappings else "✗"
            launches_str = f"launches={comp.num_launches}"
            tensor_str = ""
            if comp.tensor_data:
                tensor_str = f" tensor={comp.tensor_data}"
                any_tensor_data = True
            print(
                f"  [{comp.index:2d}] {comp.kernel_name[:30]:30s} "
                f"hash={comp.kernel_hash} "
                f"stages={stages} warps={warps} mapped={has_map} "
                f"{launches_str}{tensor_str}"
            )
        print("-" * 80)
        print(f"Total: {len(compilations)} compilation(s)")
        print("\nUse --events N,M to compare two compilations (e.g., --events 0,1)")
        if any_tensor_data:
            print("Add --tensor-values to compare tensor data between launch events")
        return

    # Parse event indices
    index_a, index_b = _parse_event_indices(events)

    # Get the two compilation events to compare
    # When kernel filter is specified, use the filtered compilations
    if kernel:
        # Use filtered compilations (compilations_a/b were set earlier)
        if index_a >= len(compilations_a):
            raise ValueError(
                f"Event index {index_a} out of range for kernel '{kernel}' in {input_path_a} "
                f"(has {len(compilations_a)} matching compilations)"
            )
        if index_b >= len(compilations_b):
            raise ValueError(
                f"Event index {index_b} out of range for kernel '{kernel}' in {input_path_b} "
                f"(has {len(compilations_b)} matching compilations)"
            )

        _, comp_a = compilations_a[index_a]
        _, comp_b = compilations_b[index_b]
    elif len(input_paths) == 2:
        # Dual-file mode: index_a from file A, index_b from file B
        comps_a = get_compilation_events(all_events_a)
        comps_b = get_compilation_events(all_events_b)

        if index_a >= len(comps_a):
            raise ValueError(
                f"Event index {index_a} out of range for {input_path_a} "
                f"(has {len(comps_a)} compilations)"
            )
        if index_b >= len(comps_b):
            raise ValueError(
                f"Event index {index_b} out of range for {input_path_b} "
                f"(has {len(comps_b)} compilations)"
            )

        _, comp_a = comps_a[index_a]
        _, comp_b = comps_b[index_b]
    else:
        # Single-file mode: both indices from same file
        comp_a, comp_b = match_events_by_index(all_events_a, index_a, index_b)

    # Find launch events if tensor value comparison requested
    launch_a, launch_b = None, None
    if tensor_values:
        hash_a = get_kernel_hash(comp_a)
        hash_b = get_kernel_hash(comp_b)
        launch_a = find_launch_for_compilation(all_events_a, comp_a, hash_a)
        launch_b = find_launch_for_compilation(all_events_b, comp_b, hash_b)

    # Run the diff engine
    engine = DiffEngine(
        comp_a,
        comp_b,
        source_trace_a=input_path_a,
        source_trace_b=input_path_b,
        event_index_a=index_a,
        event_index_b=index_b,
        launch_a=launch_a,
        launch_b=launch_b,
        tensor_values=tensor_values,
        atol=atol,
        rtol=rtol,
    )
    result = engine.run()

    # Print CLI summary
    if not quiet:
        print(format_summary(result))
        print()

    # Create diff event
    diff_event = create_diff_event(result)

    # Write output
    if in_place:
        append_diff_to_file(input_path_a, diff_event)
        if not quiet:
            print(f"Appended diff event to: {input_path_a}")
    else:
        output_path = output or _generate_output_path(input_path_a)
        writer = ConsolidatedDiffWriter()
        writer.add_diff(result, comp_a, comp_b)
        writer.write(output_path)
        if not quiet:
            print(f"Output written to: {output_path}")
