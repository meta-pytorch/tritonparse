#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Extract source code mappings from Triton trace files and update the original JSON.
This script reads a JSON trace file containing Triton IR (TTIR, TTGIR) and PTX(AMDGCN),
and extracts bidirectional mappings between:
- Python ↔ TTIR
- Python ↔ TTGIR
- Python ↔ PTX(AMDGCN)
- TTIR ↔ TTGIR
- TTIR ↔ PTX(AMDGCN)
- TTGIR ↔ PTX(AMDGCN)
"""

import argparse
import logging

from .trace_processor import parse_single_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SourceMapping")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract source code mappings from Triton trace files."
    )
    parser.add_argument("-i", "--input", help="Path to the Triton trace NDJSON file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the output files. If not specified, the input file's directory will be used.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output NDJSON path. If it is None, the default output file name will be set to {input}_mapped.ndjson in the parse function.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.input:
        parse_single_file(args.input, args.output_dir)
    else:
        logger.error("No input file specified.")
