#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the info subcommand.

This module provides command-line interface for querying kernel information
from NDJSON trace files.
"""

import argparse
import tempfile
from typing import Optional

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


def info_command(
    input_path: str, kernel_name: Optional[str] = None, skip_logger: bool = False
) -> None:
    """
    Main function for the info command.

    Args:
        input_path: Path to ndjson file
        kernel_name: Optional kernel name to list launches for
        skip_logger: Whether to skip usage logging (default: False).
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
            print(f"\nLaunches for '{kernel_name}':")
            print("-" * 60)
            for launch in launches:
                grid_str = str(launch.grid) if launch.grid else "N/A"
                print(
                    f"  id={launch.launch_id:3d}  line {launch.line_index:5d}  grid={grid_str}"
                )
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
