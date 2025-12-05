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


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Validate argument combinations based on the selected mode.

    Args:
        args: Parsed arguments.
        parser: ArgumentParser for error reporting.
    """
    if args.status:
        # Status mode: no required args (will use default state path if not specified)
        return

    if args.resume:
        # Resume mode: --state is optional (will use default if not specified)
        return

    if args.llvm_only:
        # LLVM-only mode: requires specific arguments
        missing = []
        if not args.triton_dir:
            missing.append("--triton-dir")
        if not args.test_script:
            missing.append("--test-script")
        # --triton-commit is optional, defaults to current HEAD of triton_dir
        if not args.good_llvm:
            missing.append("--good-llvm")
        if not args.bad_llvm:
            missing.append("--bad-llvm")

        if missing:
            parser.error(
                f"--llvm-only requires the following arguments: {', '.join(missing)}"
            )
        return

    # Default mode (Triton bisect) or full workflow (with --commits-csv)
    missing = []
    if not args.triton_dir:
        missing.append("--triton-dir")
    if not args.test_script:
        missing.append("--test-script")
    if not args.good:
        missing.append("--good")
    if not args.bad:
        missing.append("--bad")

    if missing:
        parser.error(f"The following arguments are required: {', '.join(missing)}")


def bisect_command(args: argparse.Namespace) -> int:
    """
    Execute the bisect command based on parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """

    # Handle --status mode
    if args.status:
        return _handle_status(args)

    # Handle --resume mode
    if args.resume:
        return _handle_resume(args)

    # Handle --llvm-only mode
    if args.llvm_only:
        return _handle_llvm_only(args)

    # Default mode: Triton bisect (or full workflow if --commits-csv provided)
    return _handle_triton_bisect(args)


def _handle_status(args: argparse.Namespace) -> int:
    """Handle --status mode: show current bisect status."""
    state_path = args.state or "./bisect_logs/state.json"

    try:
        from tritonparse.bisect.state import StateManager

        state = StateManager.load(state_path)
        StateManager.print_status(state)
        return 0
    except FileNotFoundError:
        print(f"No state file found at: {state_path}")
        print("No bisect in progress.")
        return 0
    except ImportError:
        # state.py not yet implemented (PR7)
        print("Status feature not yet implemented (requires PR7)")
        return 1
    except Exception as e:
        print(f"Error loading state: {e}")
        return 1


def _handle_resume(args: argparse.Namespace) -> int:
    """Handle --resume mode: resume from saved state."""
    state_path = args.state or "./bisect_logs/state.json"

    try:
        from tritonparse.bisect.workflow import BisectWorkflow

        workflow = BisectWorkflow.resume(state_path)
        result = workflow.run()
        _print_result(result)
        return 0
    except FileNotFoundError:
        print(f"State file not found: {state_path}")
        return 1
    except ImportError:
        # workflow.py not yet implemented (PR7)
        print("Resume feature not yet implemented (requires PR7)")
        return 1
    except Exception as e:
        print(f"Error resuming bisect: {e}")
        return 1


def _handle_llvm_only(args: argparse.Namespace) -> int:
    """Handle --llvm-only mode: bisect only LLVM commits."""
    try:
        from tritonparse.bisect.llvm_bisector import LLVMBisector

        logger = _create_logger(args.log_dir)

        bisector = LLVMBisector(
            triton_dir=args.triton_dir,
            test_script=args.test_script,
            conda_env=args.conda_env,
            logger=logger,
            build_command=args.build_command,
        )

        culprit = bisector.run(
            triton_commit=args.triton_commit,
            good_llvm=args.good_llvm,
            bad_llvm=args.bad_llvm,
        )

        print(f"\n{'=' * 60}")
        print("LLVM Bisect Result")
        print(f"{'=' * 60}")
        print(f"Culprit LLVM commit: {culprit}")
        print(f"{'=' * 60}")
        return 0

    except ImportError:
        # llvm_bisector.py not yet implemented (PR5)
        print("LLVM-only bisect not yet implemented (requires PR5)")
        return 1
    except Exception as e:
        print(f"LLVM bisect failed: {e}")
        return 1


def _handle_triton_bisect(args: argparse.Namespace) -> int:
    """Handle default mode: Triton bisect (or full workflow with --commits-csv)."""
    from tritonparse.bisect.triton_bisector import TritonBisectError, TritonBisector

    logger = _create_logger(args.log_dir)

    # Check if this is full workflow mode
    if args.commits_csv:
        return _handle_full_workflow(args, logger)

    # Triton-only bisect
    try:
        bisector = TritonBisector(
            triton_dir=args.triton_dir,
            test_script=args.test_script,
            conda_env=args.conda_env,
            logger=logger,
            build_command=args.build_command,
        )

        culprit = bisector.run(
            good_commit=args.good,
            bad_commit=args.bad,
        )

        print(f"\n{'=' * 60}")
        print("Triton Bisect Result")
        print(f"{'=' * 60}")
        print(f"Culprit commit: {culprit}")
        print(f"Log directory: {args.log_dir}")
        print(f"{'=' * 60}")
        return 0

    except TritonBisectError as e:
        print(f"\nTriton bisect failed: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


def _handle_full_workflow(args: argparse.Namespace, logger) -> int:  # noqa: ARG001
    """Handle full workflow mode (with --commits-csv)."""
    try:
        from tritonparse.bisect.workflow import BisectWorkflow

        workflow = BisectWorkflow(
            triton_dir=args.triton_dir,
            test_script=args.test_script,
            good_commit=args.good,
            bad_commit=args.bad,
            commits_csv=args.commits_csv,
            conda_env=args.conda_env,
            log_dir=args.log_dir,
            build_command=args.build_command,
        )

        result = workflow.run()
        _print_result(result)
        return 0

    except ImportError:
        # workflow.py not yet implemented (PR7)
        print("Full workflow not yet implemented (requires PR7)")
        print("Use default mode (without --commits-csv) for Triton-only bisect.")
        return 1
    except Exception as e:
        print(f"Workflow failed: {e}")
        return 1


def _create_logger(log_dir: str):
    """Create a BisectLogger instance."""
    from tritonparse.bisect.logger import BisectLogger

    return BisectLogger(log_dir)


def _print_result(result: dict) -> None:
    """Print the bisect result in a formatted way."""
    print(f"\n{'=' * 60}")
    print("Bisect Result")
    print(f"{'=' * 60}")

    print(f"Phase: {result.get('phase', 'unknown')}")
    print(f"Triton culprit: {result.get('triton_culprit', 'N/A')}")

    if result.get("is_llvm_bump"):
        print("Is LLVM bump: Yes")
        print(f"LLVM culprit: {result.get('llvm_culprit', 'N/A')}")
        llvm_range = result.get("llvm_range", {})
        print(
            f"LLVM range: {llvm_range.get('good', 'N/A')} -> {llvm_range.get('bad', 'N/A')}"
        )
    else:
        print("Is LLVM bump: No")

    print(f"{'=' * 60}")
