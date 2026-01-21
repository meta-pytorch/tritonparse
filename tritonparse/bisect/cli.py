# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CLI implementation for the bisect subcommand.

This module provides the command-line interface for bisecting Triton and LLVM
commits to find regression-causing changes.

Usage Examples:
    # Default: Triton bisect only
    tritonparseoss bisect --triton-dir /path/to/triton --test-script test.py \\
        --good v2.0.0 --bad HEAD

    # Full workflow (Triton -> detect LLVM bump -> LLVM bisect if needed)
    tritonparseoss bisect --triton-dir /path/to/triton --test-script test.py \\
        --good v2.0.0 --bad HEAD --commits-csv pairs.csv

    # LLVM-only bisect (uses current HEAD of triton repo if --triton-commit not specified)
    tritonparseoss bisect --llvm-only --triton-dir /path/to/triton \\
        --test-script test.py --good-llvm def456 --bad-llvm 789abc

    # LLVM-only bisect with explicit triton commit
    tritonparseoss bisect --llvm-only --triton-dir /path/to/triton \\
        --test-script test.py --triton-commit abc123 \\
        --good-llvm def456 --bad-llvm 789abc

    # Resume from saved state
    tritonparseoss bisect --resume --state ./bisect_logs/state.json

    # Check status
    tritonparseoss bisect --status
"""

import argparse


def _add_bisect_args(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the bisect subcommand.

    This implements smart defaults with switches:
    - Default (no special flags) = Triton bisect only
    - --commits-csv = Full workflow (Triton -> detect LLVM bump -> LLVM bisect if needed)
    - --llvm-only = LLVM bisect only
    - --resume = Resume from saved state
    - --status = Show bisect status
    """
    # Mode switches (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--llvm-only",
        action="store_true",
        help="Only bisect LLVM commits (requires --triton-commit, --good-llvm, --bad-llvm)",
    )
    mode_group.add_argument(
        "--pair-test",
        action="store_true",
        help="Test (Triton, LLVM) commit pairs from CSV to find LLVM bisect range "
        "(requires --commits-csv, --good-llvm, --bad-llvm)",
    )
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume bisect from saved state file",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show current bisect status",
    )

    # Common arguments
    parser.add_argument(
        "--triton-dir",
        type=str,
        help="Path to Triton repository",
    )
    parser.add_argument(
        "--test-script",
        type=str,
        help="Path to test script that determines pass/fail",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="triton_bisect",
        help="Conda environment name (default: triton_bisect)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./bisect_logs",
        help="Directory for log files (default: ./bisect_logs)",
    )
    parser.add_argument(
        "--build-command",
        type=str,
        default=None,
        help="Custom build command (default: 'pip install -e .' for Triton)",
    )

    # Triton bisect arguments
    parser.add_argument(
        "--good",
        type=str,
        help="Known good commit (test passes)",
    )
    parser.add_argument(
        "--bad",
        type=str,
        help="Known bad commit (test fails)",
    )

    # Full workflow argument
    parser.add_argument(
        "--commits-csv",
        type=str,
        help="CSV file with (triton_commit, llvm_commit) pairs for full workflow",
    )

    # LLVM-only arguments
    parser.add_argument(
        "--triton-commit",
        type=str,
        help="Fixed Triton commit for LLVM bisect (default: current HEAD of triton-dir)",
    )
    parser.add_argument(
        "--good-llvm",
        type=str,
        help="Known good LLVM commit (required with --llvm-only)",
    )
    parser.add_argument(
        "--bad-llvm",
        type=str,
        help="Known bad LLVM commit (required with --llvm-only)",
    )

    # Resume/status arguments
    parser.add_argument(
        "--state",
        type=str,
        help="Path to state file (for --resume or --status)",
    )

    # TUI control
    parser.add_argument(
        "--tui",
        action="store_true",
        default=True,
        dest="tui",
        help="Enable Rich TUI interface (default: enabled if available)",
    )
    parser.add_argument(
        "--no-tui",
        action="store_false",
        dest="tui",
        help="Disable Rich TUI, use plain text output",
    )
