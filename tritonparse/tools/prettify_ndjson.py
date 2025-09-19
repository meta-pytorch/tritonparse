#!/usr/bin/env python3
"""
Convert an NDJSON file to a prettified JSON file.

This script takes an NDJSON (newline-delimited JSON) file and converts it to a
standard human-readable JSON file where each line becomes an element in a JSON array, with
pretty formatting applied.

Example:
    Input NDJSON file (data.ndjson):
        {"name": "Alice", "age": 30}
        {"name": "Bob", "age": 25}
        {"name": "Charlie", "age": 35}

    Output JSON file (data_prettified.json):
        [
          {
            "age": 30,
            "name": "Alice"
          },
          {
            "age": 25,
            "name": "Bob"
          },
          {
            "age": 35,
            "name": "Charlie"
          }
        ]

Usage:
    python prettify_ndjson.py data.ndjson
    python prettify_ndjson.py --lines 1,3 data.ndjson  # Only process lines 1 and 3
    python prettify_ndjson.py --save-irs logs.ndjson   # Keep all fields for compilation events


"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Union


def parse_line_ranges(lines_arg: str) -> set[int]:
    """
    Parse line ranges from string like "1,2,3,5-10" into a set of line numbers.

    Line numbers use 1-based indexing (first line is line 1, not 0).

    Args:
        lines_arg: String containing comma-separated line numbers and ranges
                  Examples: "1", "1,2,3", "5-10", "1,3,5-10,15"

    Returns:
        Set of line numbers (1-based indexing, where 1 = first line)

    Raises:
        ValueError: If the format is invalid or contains non-positive numbers
    """
    line_numbers = set()

    if not lines_arg.strip():
        return line_numbers

    parts = lines_arg.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            # Handle range like "5-10"
            try:
                start, end = part.split("-", 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                if start_num <= 0 or end_num <= 0:
                    raise ValueError("Line numbers must be positive")
                if start_num > end_num:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                line_numbers.update(range(start_num, end_num + 1))
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid range format: {part}")
                raise
        else:
            # Handle single number like "1"
            try:
                line_num = int(part)
                if line_num <= 0:
                    raise ValueError("Line numbers must be positive")
                line_numbers.add(line_num)
            except ValueError:
                raise ValueError(f"Invalid line number: {part}")

    return line_numbers


def load_ndjson(
    file_path: Path, save_irs: bool = False, line_filter: set[int] = None
) -> List[Any]:
    """
    Load NDJSON file and return list of JSON objects.

    Args:
        file_path: Path to the NDJSON file
        save_irs: Whether to save file_content and python_source for compilation events
        line_filter: Set of line numbers to include (1-based indexing), None means include all

    Returns:
        List of parsed JSON objects

    Raises:
        FileNotFoundError: If the input file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
    """
    json_objects = []
    filtered_compilation_events = 0
    total_lines_processed = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # enumerate(f, 1) starts line numbering from 1 (1-based indexing)
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                # Skip line if line filtering is enabled and this line is not in the filter
                # line_num is 1-based (first line = 1, second line = 2, etc.)
                if line_filter is not None and line_num not in line_filter:
                    continue

                total_lines_processed += 1

                try:
                    json_obj = json.loads(line)

                    # Filter out file_content and python_source for compilation events if save_irs is False
                    if not save_irs and isinstance(json_obj, dict):
                        event_type = json_obj.get("event_type")
                        if event_type == "compilation":
                            # Remove file_content and python_source from payload if they exist
                            payload = json_obj.get("payload")
                            if isinstance(payload, dict):
                                fields_to_remove = []
                                if "file_content" in payload:
                                    fields_to_remove.append("file_content")
                                if "python_source" in payload:
                                    fields_to_remove.append("python_source")

                                if fields_to_remove:
                                    payload = (
                                        payload.copy()
                                    )  # Create a copy to avoid modifying original
                                    for field in fields_to_remove:
                                        del payload[field]
                                    json_obj = (
                                        json_obj.copy()
                                    )  # Create a copy of the main object
                                    json_obj["payload"] = payload
                                    filtered_compilation_events += 1

                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(
                        f"Error parsing JSON on line {line_num}: {e}", file=sys.stderr
                    )
                    print(f"Problematic line: {line[:100]}...", file=sys.stderr)
                    raise

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        raise
    except (OSError, UnicodeDecodeError) as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        raise

    # Print informational messages
    if line_filter is not None:
        if line_filter:
            print(
                f"Line filtering: processed {total_lines_processed} out of {len(line_filter)} specified lines"
            )
        else:
            print("Line filtering: no valid lines specified")

    # Print warning if compilation events were filtered
    if not save_irs and filtered_compilation_events > 0:
        print(
            f"WARNING: Removed 'file_content' and 'python_source' fields from {filtered_compilation_events} compilation events to reduce file size.",
            file=sys.stderr,
        )
        print(
            "         Use --save-irs flag to preserve these fields if needed.",
            file=sys.stderr,
        )

    return json_objects


def save_prettified_json(
    json_objects: Union[List[Any], Any], output_path: Path
) -> None:
    """
    Save JSON data to a prettified JSON file.

    Args:
        json_objects: Either a list of JSON objects or a single JSON-serializable object
        output_path: Path where to save the prettified JSON file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_objects, f, indent=2, ensure_ascii=False, sort_keys=True)
        print(f"Successfully converted to prettified JSON: {output_path}")
    except OSError as e:
        print(f"Error writing to file '{output_path}': {e}", file=sys.stderr)
        raise


def main():
    """Main function to handle command line arguments and orchestrate the conversion."""
    parser = argparse.ArgumentParser(
        description="Convert NDJSON file to prettified JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prettify_ndjson.py data.ndjson
  python prettify_ndjson.py /path/to/logs.ndjson
        """,
    )

    parser.add_argument(
        "ndjson_file", type=str, help="Path to the NDJSON file to convert"
    )

    parser.add_argument(
        "--save-irs",
        action="store_true",
        default=False,
        help="Save file_content and python_source for compilation events (default: False, removes these fields to reduce size)",
    )

    parser.add_argument(
        "--lines",
        type=str,
        help="Specify line numbers to include using 1-based indexing (e.g., '1,2,3,5-10'). "
        "Line 1 is the first line of the file. Only these lines from the original NDJSON will be processed. "
        "Supports individual lines (1,2,3) and ranges (5-10).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Specify output file path (default: {input_stem}_prettified.json in the same directory as input)",
    )

    args = parser.parse_args()

    # Convert to Path object and validate
    input_path = Path(args.ndjson_file)

    if not input_path.exists():
        print(f"Error: File '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"Error: '{input_path}' is not a file.", file=sys.stderr)
        sys.exit(1)

    # Generate output filename
    if args.output:
        output_path = Path(args.output)
    else:
        # Default: original_prettified.json in same directory as input
        output_path = input_path.parent / f"{input_path.stem}_prettified.json"

    try:
        # Parse line filter if provided
        line_filter = None
        if args.lines:
            try:
                line_filter = parse_line_ranges(args.lines)
                print(
                    f"Line filtering enabled: will process {len(line_filter)} specified lines"
                )
            except ValueError as e:
                print(f"Error parsing --lines argument: {e}", file=sys.stderr)
                sys.exit(1)

        # Load NDJSON file
        print(f"Loading NDJSON file: {input_path}")
        if not args.save_irs:
            print(
                "Filtering out file_content and python_source from compilation events to reduce size"
            )
        json_objects = load_ndjson(
            input_path, save_irs=args.save_irs, line_filter=line_filter
        )
        print(f"Loaded {len(json_objects)} JSON objects")

        # Save as prettified JSON
        print(f"Saving prettified JSON to: {output_path}")
        save_prettified_json(json_objects, output_path)

        print("Conversion completed successfully!")

    except Exception as e:
        print(f"Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
