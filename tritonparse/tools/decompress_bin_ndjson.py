#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Script to decompress .bin.ndjson files back to regular .ndjson format.

The .bin.ndjson format stores each JSON record as a separate gzip member,
concatenated in sequence within a single binary file. This script uses
gzip.open() which automatically handles member concatenation to read
the compressed file and write out the original NDJSON format.

Usage:
    python decompress_bin_ndjson.py trace.bin.ndjson
"""

import argparse
import gzip
import sys
from pathlib import Path


def decompress_bin_ndjson(input_file: str, output_file: str = None) -> None:
    """
    Decompress a .bin.ndjson file to regular .ndjson format.

    Args:
        input_file: Path to the .bin.ndjson file
        output_file: Path for the output .ndjson file (optional)
    """
    input_path = Path(input_file)

    # Validate input file
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist", file=sys.stderr)
        return

    if not input_path.suffix.endswith(".bin.ndjson"):
        print(f"Warning: Input file '{input_file}' doesn't have .bin.ndjson extension")

    # Determine output file path
    if output_file is None:
        if input_path.name.endswith(".bin.ndjson"):
            # Replace .bin.ndjson with .ndjson
            output_file = str(input_path.with_suffix("").with_suffix(".ndjson"))
        else:
            # Add .decompressed.ndjson suffix
            output_file = str(input_path.with_suffix(".decompressed.ndjson"))

    output_path = Path(output_file)

    try:
        line_count = 0
        # Because we use NDJSON format, each line is a complete JSON record.
        # It is guruanteed here https://github.com/meta-pytorch/tritonparse/blob/
        # c8dcc2a94ac10ede4342dba7456f6ebd8409b95d/tritonparse/structured_logging.py#L320
        with gzip.open(input_path, "rt", encoding="utf-8") as compressed_file:
            with open(output_path, "w", encoding="utf-8") as output:
                for line in compressed_file:
                    # gzip.open automatically handles member concatenation
                    # Each line is already a complete JSON record with newline
                    output.write(line)
                    line_count += 1

        # Get file sizes for comparison
        input_size = input_path.stat().st_size
        output_size = output_path.stat().st_size
        compression_ratio = (
            (1 - input_size / output_size) * 100 if output_size > 0 else 0
        )

        print(f"Successfully decompressed '{input_file}' to '{output_file}'")
        print(f"  Input size:  {input_size:,} bytes")
        print(f"  Output size: {output_size:,} bytes")
        print(f"  Compression ratio: {compression_ratio:.1f}%")
        print(f"  Records processed: {line_count:,}")

    except gzip.BadGzipFile as e:
        print(f"Error: Invalid gzip format in '{input_file}': {e}", file=sys.stderr)
    except UnicodeDecodeError as e:
        print(f"Error: Unicode decode error in '{input_file}': {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error: Failed to decompress '{input_file}': {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Decompress .bin.ndjson files to regular .ndjson format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trace.bin.ndjson
  %(prog)s trace.bin.ndjson -o output.ndjson
  %(prog)s /logs/dedicated_log_triton_trace_user_.bin.ndjson
        """,
    )

    parser.add_argument("input_file", help="Input .bin.ndjson file to decompress")

    parser.add_argument(
        "-o",
        "--output",
        help="Output .ndjson file path (default: replace .bin.ndjson with .ndjson)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Decompressing: {args.input_file}")
        if args.output:
            print(f"Output file: {args.output}")

    decompress_bin_ndjson(args.input_file, args.output)


if __name__ == "__main__":
    main()
