#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the info subcommand.

This module provides command-line interface for querying kernel information
from NDJSON trace files.
"""

import argparse
import tempfile
from typing import Any, Dict, Optional

from tritonparse.info.kernel_query import (
    find_similar_kernels,
    list_kernels_fast,
    list_launches_for_kernel,
)
from tritonparse.info.parse_helper import parse_and_compress_raw_log
from tritonparse.shared_vars import is_fbcode
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
        help=(
            "Filter by argument values. Supports simple and advanced syntax:\n"
            "  Simple: num_stages=3,num_warps=4\n"
            "  Nested: C_ptr.shape=[3024, 10752],C_ptr.dtype=torch.bfloat16\n"
            "  Array index: C_ptr.shape[0]=3024"
        ),
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


def _parse_value(value: str) -> Any:
    """
    Parse a string value into the appropriate Python type.

    Handles booleans, integers, floats, lists, and strings.
    """
    value = value.strip()

    # Check for boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Check for list (e.g., "[3024, 10752]")
    if value.startswith("[") and value.endswith("]"):
        import json

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # Try to convert to int
    try:
        return int(value)
    except ValueError:
        pass

    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass

    return value


def _parse_args_list(args_list: str) -> Dict[str, Any]:
    """
    Parse the args-list filter string into a dictionary.

    Supports simple and advanced filter syntax:
    - Simple: "num_stages=3,num_warps=4"
    - Nested: "C_ptr.shape=[3024, 10752],C_ptr.dtype=torch.bfloat16"
    - Array index: "C_ptr.shape[0]=3024"

    Args:
        args_list: Comma-separated key=value pairs

    Returns:
        Dictionary mapping field names (with optional path) to expected values.
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
        filters[key] = _parse_value(value)

    return filters


def _get_nested_value(obj: Any, path: str) -> Any:
    """
    Get a nested value from an object using dot notation and array indexing.

    Examples:
        _get_nested_value({"a": {"b": 1}}, "a.b") -> 1
        _get_nested_value({"a": [1, 2, 3]}, "a[1]") -> 2
        _get_nested_value({"a": {"b": [1, 2]}}, "a.b[0]") -> 1

    Args:
        obj: The object to traverse
        path: Dot-separated path with optional array indices

    Returns:
        The value at the path, or None if not found
    """
    import re

    if obj is None:
        return None

    current = obj
    # Split by dots, but keep array indices attached to their keys
    # e.g., "a.b[0].c" -> ["a", "b[0]", "c"]
    parts = path.split(".")

    for part in parts:
        if current is None:
            return None

        # Check for array index: key[index]
        match = re.match(r"^(\w+)\[(\d+)\]$", part)
        if match:
            key, index = match.groups()
            index = int(index)
            if isinstance(current, dict) and key in current:
                current = current[key]
                if isinstance(current, (list, tuple)) and 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                return None
        else:
            # Simple key access
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

    return current


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


def _find_value_in_sources(
    key: str,
    comp_meta: Dict[str, Any],
    extracted_args: Dict[str, Any],
    extracted_inductor_args: Dict[str, Any],
) -> Any:
    """
    Find a value by key in one of the data sources.

    Args:
        key: The key to look up
        comp_meta: compilation_metadata dictionary
        extracted_args: extracted_args dictionary
        extracted_inductor_args: extracted_inductor_args dictionary

    Returns:
        The value if found, None otherwise
    """
    if key in comp_meta:
        return comp_meta[key]
    if key in extracted_args:
        return extracted_args[key]
    if key in extracted_inductor_args:
        return extracted_inductor_args[key]
    return None


def _get_filter_value(
    key: str,
    comp_meta: Dict[str, Any],
    extracted_args: Dict[str, Any],
    extracted_inductor_args: Dict[str, Any],
) -> Any:
    """
    Get the value for a filter key, handling both simple and nested paths.

    Args:
        key: Filter key, can be simple ("num_stages") or nested ("C_ptr.shape[0]")
        comp_meta: compilation_metadata dictionary
        extracted_args: extracted_args dictionary
        extracted_inductor_args: extracted_inductor_args dictionary

    Returns:
        The actual value for comparison, or None if not found
    """
    import re

    # Check if this is a nested path (contains . or [)
    if "." not in key and "[" not in key:
        # Simple key lookup
        return _find_value_in_sources(
            key, comp_meta, extracted_args, extracted_inductor_args
        )

    # Advanced filter: parse path and get nested value
    match = re.match(r"^(\w+)", key)
    if not match:
        return None

    arg_name = match.group(1)
    rest_path = key[len(arg_name) :]
    if rest_path.startswith("."):
        rest_path = rest_path[1:]

    arg_obj = _find_value_in_sources(
        arg_name, comp_meta, extracted_args, extracted_inductor_args
    )
    if arg_obj is None:
        return None

    return _get_nested_value(arg_obj, rest_path) if rest_path else arg_obj


def _launch_matches_filter(event: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """
    Check if a launch event matches the provided filters.

    Supports both simple and advanced filter syntax:
    - Simple: "num_stages" matches top-level keys
    - Advanced: "C_ptr.shape" or "C_ptr.shape[0]" matches nested properties

    Args:
        event: A launch event dictionary
        filters: Dictionary of field names (with optional path) to expected values

    Returns:
        True if the event matches all filters, False otherwise
    """
    if not filters:
        return True

    comp_meta = event.get("compilation_metadata", {})
    extracted_args = event.get("extracted_args", {})
    extracted_inductor_args = event.get("extracted_inductor_args", {})

    for key, expected_value in filters.items():
        actual_value = _get_filter_value(
            key, comp_meta, extracted_args, extracted_inductor_args
        )

        # Unwrap nested dict values to get the actual value
        if actual_value and isinstance(actual_value, dict) and "value" in actual_value:
            actual_value = actual_value["value"]

        if actual_value != expected_value:
            return False

    return True
