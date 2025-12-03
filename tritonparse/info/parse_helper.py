#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Helper functions for parsing raw log files in the info module.

This module provides utilities to parse and compress raw log files,
reusing functionality from the parse module.
"""

from pathlib import Path

from tritonparse.common import gzip_single_file
from tritonparse.trace_processor import parse_single_file


def parse_and_compress_raw_log(
    input_path: str,
    output_dir: str,
    split_inductor_compilations: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Parse a raw log file, compress it, and return the path to the compressed parsed file.

    This function reuses the parse module's functionality:
    - parse_single_file: Parse the file
    - gzip_single_file: Compress the parsed file

    Args:
        input_path: Path to raw log file
        output_dir: Directory to save parsed file
        split_inductor_compilations: Whether to split by inductor compilations
        verbose: Whether to print verbose information

    Returns:
        Path to the generated compressed parsed file (.ndjson.gz)

    Raises:
        RuntimeError: If parsing fails or parsed file not found
    """
    # 1. Parse the file (generates uncompressed .ndjson)
    parse_single_file(
        input_path,
        output_dir=output_dir,
        split_inductor_compilations=split_inductor_compilations,
    )

    # 2. Calculate expected output filename
    input_path_obj = Path(input_path)
    file_name = input_path_obj.name

    if input_path.endswith(".bin.ndjson"):
        file_name_without_ext = file_name[:-11]  # Remove ".bin.ndjson"
    else:
        file_name_without_ext = input_path_obj.stem  # Remove all extensions
        # If there's still a .ndjson extension, remove it
        if file_name_without_ext.endswith(".ndjson"):
            file_name_without_ext = file_name_without_ext[:-7]

    uncompressed_file = Path(output_dir) / f"{file_name_without_ext}_mapped.ndjson"

    if not uncompressed_file.exists():
        raise RuntimeError(
            f"Failed to generate parsed file. Expected: {uncompressed_file}"
        )

    # 3. Compress the file (reusing parse module's function)
    compressed_file = gzip_single_file(str(uncompressed_file), verbose=verbose)

    return Path(compressed_file)  # Returns .ndjson.gz path
