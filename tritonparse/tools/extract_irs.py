#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Extract IR files from NDJSON trace logs.

This script extracts intermediate representation (IR) files from a Triton trace NDJSON file.
For compilation events, it extracts the IR files (ttir, ttgir, llir, ptx, etc.) contained in
the file_content field and saves them as individual files.

Example:
    Extract IRs from line 0 (first line) of the NDJSON file:
        python extract_irs.py -i logs.ndjson --line 0 -o output_folder

    Extract from line 5:
        python extract_irs.py -i logs.ndjson --line 5 -o ./irs

Usage:
    python extract_irs.py -i <input.ndjson> --line <line_number> -o <output_folder>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def read_ndjson_line(file_path: Path, line_number: int) -> Optional[Dict[str, Any]]:
    """
    Read a specific line from an NDJSON file (0-based indexing).

    Args:
        file_path: Path to the NDJSON file
        line_number: Line number to read (0-based, where 0 = first line)

    Returns:
        Parsed JSON object from the specified line, or None if line doesn't exist

    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If the line contains invalid JSON
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for current_line_num, line in enumerate(f):
                if current_line_num == line_number:
                    line = line.strip()
                    if not line:
                        print(f"Warning: Line {line_number} is empty", file=sys.stderr)
                        return None
                    return json.loads(line)

        print(
            f"Error: Line {line_number} not found in file (file has fewer lines)",
            file=sys.stderr,
        )
        return None

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON on line {line_number}: {e}", file=sys.stderr)
        raise


def extract_irs(
    json_obj: Dict[str, Any], output_dir: Path, kernel_name: Optional[str] = None
) -> int:
    """
    Extract IR files from a JSON object and save them to the output directory.

    Args:
        json_obj: Parsed JSON object from the NDJSON file
        output_dir: Directory to save the extracted IR files
        kernel_name: Optional kernel name to use for file naming (overrides metadata.name)

    Returns:
        Number of files extracted

    Raises:
        ValueError: If the JSON object is not a compilation event or missing required fields
    """
    # Validate that this is a compilation event
    event_type = json_obj.get("event_type")
    if event_type != "compilation":
        raise ValueError(f"Not a compilation event (event_type: {event_type})")

    payload = json_obj.get("payload")
    if not payload:
        raise ValueError("Missing 'payload' field in JSON object")

    # Get file_content
    file_content = payload.get("file_content")
    if not file_content:
        raise ValueError("Missing 'file_content' field in payload")

    # Determine kernel name
    if kernel_name is None:
        metadata = payload.get("metadata", {})
        kernel_name = metadata.get("name", "kernel")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract each IR file
    files_extracted = 0
    for file_key, content in file_content.items():
        # Determine file extension from the key
        # file_key is typically like "embedding_forward_kernel.ttir"
        # We want to extract just the extension
        if "." in file_key:
            extension = file_key.split(".")[-1]
        else:
            extension = "txt"

        # Create output filename
        output_filename = f"{kernel_name}.{extension}"
        output_path = output_dir / output_filename

        # Write content to file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Extracted: {output_path}")
            files_extracted += 1
        except OSError as e:
            print(f"Error writing file {output_path}: {e}", file=sys.stderr)

    # Optionally extract Python source code
    python_source = payload.get("python_source")
    if python_source and isinstance(python_source, dict):
        source_code = python_source.get("code")
        if source_code:
            output_path = output_dir / f"{kernel_name}_source.py"
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    # Add header comment with file path and line range
                    file_path_info = python_source.get("file_path", "unknown")
                    start_line = python_source.get("start_line", "?")
                    end_line = python_source.get("end_line", "?")
                    f.write(f"# Source: {file_path_info}\n")
                    f.write(f"# Lines: {start_line}-{end_line}\n\n")
                    f.write(source_code)
                print(f"Extracted Python source: {output_path}")
                files_extracted += 1
            except OSError as e:
                print(
                    f"Error writing Python source file {output_path}: {e}",
                    file=sys.stderr,
                )

    return files_extracted


def main():
    """Main function to handle command line arguments and orchestrate IR extraction."""
    parser = argparse.ArgumentParser(
        description="Extract IR files from Triton trace NDJSON logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract IRs from line 0 (first line):
    python extract_irs.py -i logs.ndjson --line 0 -o output_folder
  
  Extract from line 5:
    python extract_irs.py -i logs.ndjson --line 5 -o ./irs
  
  Specify custom kernel name:
    python extract_irs.py -i logs.ndjson --line 0 -o ./irs --kernel-name my_kernel
        """,
    )

    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the input NDJSON file"
    )

    parser.add_argument(
        "--line",
        type=int,
        required=True,
        help="Line number to extract (0-based indexing, where 0 = first line)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory to save extracted IR files",
    )

    parser.add_argument(
        "--kernel-name",
        type=str,
        help="Custom kernel name for output files (default: use metadata.name from JSON)",
    )

    args = parser.parse_args()

    # Validate line number
    if args.line < 0:
        print(
            f"Error: Line number must be non-negative (got {args.line})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Convert to Path objects
    input_path = Path(args.input)
    output_dir = Path(args.output)

    try:
        # Read the specified line
        print(f"Reading line {args.line} from {input_path}...")
        json_obj = read_ndjson_line(input_path, args.line)

        if json_obj is None:
            print("Error: Failed to read JSON from specified line", file=sys.stderr)
            sys.exit(1)

        # Extract IRs
        print(f"Extracting IRs to {output_dir}...")
        num_files = extract_irs(json_obj, output_dir, args.kernel_name)

        print(f"\nSuccess! Extracted {num_files} file(s) to {output_dir}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
