#  Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

from tritonparse.reproducer.types import KernelImportMode


def _add_reproducer_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for the reproducer to a parser."""
    parser.add_argument("input", help="Path to the ndjson/ndjson.gz log file")
    parser.add_argument(
        "--line",
        type=int,
        default=0,
        help=(
            "The line index (0-based) of the launch event in the input file to reproduce. "
            "Defaults to 0 (first launch event). Mutually exclusive with --kernel/--launch-id."
        ),
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default=None,
        help=(
            "Kernel name (exact match, case-sensitive) to reproduce. "
            "Use with --launch-id to specify which launch of the kernel. "
            "Mutually exclusive with --line."
        ),
    )
    parser.add_argument(
        "--launch-id",
        type=int,
        default=0,
        help=(
            "0-based launch index for the kernel specified by --kernel. "
            "Defaults to 0 (first launch). Only used when --kernel is provided."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="repro_output",
        help=(
            "Directory to save the reproducer script and context JSON. Defaults to "
            "'repro_output/<kernel_name>/' if not provided."
        ),
    )
    parser.add_argument(
        "--template",
        default="example",
        help=(
            "Template name (builtin, without .py) or a filesystem path to a .py file. "
            "Defaults to 'example'."
        ),
    )
    parser.add_argument(
        "--kernel-import",
        type=KernelImportMode,
        choices=list(KernelImportMode),
        default=KernelImportMode.DEFAULT,
        help=(
            "Kernel import strategy:\n"
            "  default: Import kernel from original file (current behavior)\n"
            "  copy: Embed kernel source code directly in reproducer\n"
            "  override-ttir: Use TTIR from compilation event (bypass Python frontend)\n"
            "Defaults to 'default'."
        ),
    )
    parser.add_argument(
        "--use-fbcode",
        action="store_true",
        help=("Use fbcode to setup repro environment."),
    )
