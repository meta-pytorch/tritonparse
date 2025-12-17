#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the info subcommand.

This module provides command-line interface for querying kernel information
from NDJSON trace files.
"""

import argparse
import tempfile
from typing import Any, Dict, Optional

from tritonparse.common import is_fbcode
from tritonparse.info.kernel_query import (
    find_similar_kernels,
    list_kernels_fast,
    list_launches_for_kernel,
)
from tritonparse.info.parse_helper import parse_and_compress_raw_log
from tritonparse.tools.prettify_ndjson import load_ndjson


def _add_info_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the info subcommand."""
    parser.add_argument(
        "input",
        help="Path to ndjson/ndjson.gz/.bin.ndjson file",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help="Kernel name to list launches for",
    )
    parser.add_argument(
        "--args-list",
        type=str,
        default=None,
        help="Filter by num_stages=3,num_warps=4,...",
    )


def info_command(
    input_path: str,
    kernel_name: Optional[str] = None,
    skip_logger: bool = False,
    args_list: Optional[str] = None,
) -> None:
    """
    Main function for the info command.

    Args:
        input_path: Path to ndjson file
        kernel_name: Optional kernel name to list launches for
        skip_logger: Whether to skip usage logging (default: False).
        args_list: Optional filter string like "num_stages=3,num_warps=4,BLOCK_K=64"
    """
    if not skip_logger and is_fbcode():
        from tritonparse.fb.utils import usage_report_logger

        usage_report_logger()

    # 1. Load and detect type
    events = load_ndjson(input_path)
    has_launch_diff = any(e.get("event_type") == "launch_diff" for e in events)

    # 2. If no launch_diff, auto-parse
    if not has_launch_diff:
        print(
            f"Input file '{input_path}' appears to be raw log (no launch_diff events)."
        )
        print("Parsing automatically to generate launch_diff events...")

        temp_dir = tempfile.mkdtemp(prefix="tritonparse_info_")

        try:
            # Parse and compress (reuses parse module's functions)
            parsed_file = parse_and_compress_raw_log(
                input_path,
                output_dir=temp_dir,
                split_inductor_compilations=False,
                verbose=False,
            )

            # Load compressed file (load_ndjson supports .ndjson.gz)
            events = load_ndjson(parsed_file)

            print(f"✓ Parsed and compressed file: {parsed_file}")
            print(f"  (Temporary directory: {temp_dir})")
        except Exception as e:
            raise RuntimeError(f"Failed to parse input file '{input_path}': {e}") from e
    else:
        print(f"Using parsed trace file: {input_path}")

    # 3. Process query
    if kernel_name:
        # List launches for specific kernel
        try:
            launches = list_launches_for_kernel(events, kernel_name)
            total_launches = len(launches)
            if args_list:
                launches = _filter_launches(launches, events, args_list)

            print(f"\nLaunches for '{kernel_name}':")
            print("-" * 60)
            for launch in launches:
                grid_str = str(launch.grid) if launch.grid else "N/A"
                print(
                    f"  id={launch.launch_id:3d}  line {launch.line_index:5d}  grid={grid_str}"
                )
            print("-" * 60)
            print(f"Total: {len(launches)} of {total_launches} launches.")
            if args_list:
                print(f"Filtered by: {args_list}")
        except ValueError as e:
            error_msg = str(e)
            print(f"\nError: {error_msg}")
            # Try to suggest similar kernels
            try:
                similar = find_similar_kernels(events, kernel_name, n=3)
                if similar:
                    print("\nDid you mean one of these?")
                    all_kernels = list_kernels_fast(
                        events
                    )  # Use fast path for consistency
                    kernel_dict = {k.name: k for k in all_kernels}
                    for name in similar:
                        count = kernel_dict[name].total_launches
                        print(f"  - {name} ({count} launches)")
                    print("\nUse 'tritonparseoss info <file>' to list all kernels.")
            except Exception:
                pass  # Ignore errors in suggestion
            raise
    else:
        # List all kernels
        kernels = list_kernels_fast(events)
        print(f"\nKernels in {input_path}:")
        print("-" * 60)
        for kernel in kernels:
            if kernel.total_launches > 0:
                max_id = kernel.total_launches - 1
                print(
                    f"  {kernel.name:30s} {kernel.total_launches:3d} launches "
                    f"(id: 0-{max_id})"
                )
            else:
                print(f"  {kernel.name:30s} {kernel.total_launches:3d} launches")


def _parse_args_list(args_list: str) -> Dict[str, Any]:
    """
    Parse the args-list filter string into a dictionary.

    Args:
        args_list: Comma-separated key=value pairs, e.g., "num_stages=3,num_warps=4"

    Returns:
        Dictionary mapping field names to their expected values.
        Values are converted to int, bool, or kept as strings.
    """
    filters = {}
    for pair in args_list.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid filter format: '{pair}'. Expected 'key=value'.")
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert value to appropriate type
        # Check for boolean first
        if value.lower() == "true":
            filters[key] = True
        elif value.lower() == "false":
            filters[key] = False
        else:
            # Try to convert to int
            try:
                filters[key] = int(value)
            except ValueError:
                filters[key] = value

    return filters


def _filter_launches(launches, events, args_list):
    """
    Filter launches based on args_list criteria.

    Args:
        launches: List of LaunchInfo objects
        events: List of all event dictionaries
        args_list: Optional filter string like "num_stages=3,num_warps=4"

    Returns:
        Tuple of (filtered_info string, filtered launches list)
    """
    if not args_list:
        return launches

    filters = _parse_args_list(args_list)
    filtered_launches = []
    for launch in launches:
        # Get the original event to check filter criteria
        event = events[launch.line_index]
        if _launch_matches_filter(event, filters):
            filtered_launches.append(launch)

    total_kernel_launches = len(launches)
    filtered_out_count = total_kernel_launches - len(filtered_launches)
    filtered_info = (
        f" ({filtered_out_count} filtered out of {total_kernel_launches})"
        if filtered_out_count > 0
        else ""
    )

    print(f"\nFiltered launches{filtered_info}:")
    return filtered_launches


def _launch_matches_filter(event: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Check if a launch event matches the provided filters.

    Args:
        event: A launch event dictionary
        filters: Dictionary of field names to expected values

    Returns:
        True if the event matches all filters, False otherwise
    """
    if not filters:
        return True

    comp_meta = event.get("compilation_metadata", {})
    extracted_args = event.get("extracted_args", {})
    extracted_inductor_args = event.get("extracted_inductor_args", {})

    for key, expected_value in filters.items():
        # First check compilation_metadata
        if key in comp_meta:
            actual_value = comp_meta[key]
        # Then check extracted_args
        elif key in extracted_args:
            actual_value = extracted_args[key]
        # Then check extracted_inductor_args (values are nested as {'type': ..., 'value': ...})
        elif key in extracted_inductor_args:
            inductor_arg = extracted_inductor_args[key]
            if isinstance(inductor_arg, dict) and "value" in inductor_arg:
                actual_value = inductor_arg["value"]
            else:
                actual_value = inductor_arg
        else:
            actual_value = None

        if actual_value != expected_value:
            return False

    return True
