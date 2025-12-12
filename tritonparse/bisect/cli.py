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

    if args.pair_test:
        # Pair test mode: requires CSV and LLVM range
        missing = []
        if not args.triton_dir:
            missing.append("--triton-dir")
        if not args.test_script:
            missing.append("--test-script")
        if not args.commits_csv:
            missing.append("--commits-csv")
        if not args.good_llvm:
            missing.append("--good-llvm")
        if not args.bad_llvm:
            missing.append("--bad-llvm")

        if missing:
            parser.error(
                f"--pair-test requires the following arguments: {', '.join(missing)}"
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

    # Handle --pair-test mode
    if args.pair_test:
        return _handle_pair_test(args)

    # Default mode: Triton bisect (or full workflow if --commits-csv provided)
    return _handle_triton_bisect(args)


def _handle_status(args: argparse.Namespace) -> int:
    """Handle --status mode: show current bisect status."""
    from tritonparse.bisect.state import StateManager

    state_path = args.state or "./bisect_logs/state.json"

    try:
        state = StateManager.load(state_path)
        StateManager.print_status(state)
        return 0
    except FileNotFoundError:
        print(f"No state file found at: {state_path}")
        print("No bisect in progress.")
        return 0
    except Exception as e:
        print(f"Error loading state: {e}")
        return 1


def _handle_resume(args: argparse.Namespace) -> int:
    """Handle --resume mode: resume from saved state."""
    from tritonparse.bisect.commit_detector import LLVMBumpInfo
    from tritonparse.bisect.ui import SummaryMode, print_final_summary
    from tritonparse.bisect.workflow import BisectWorkflow

    state_path = args.state or "./bisect_logs/state.json"

    try:
        workflow = BisectWorkflow.resume(state_path)
        result = workflow.run()

        # Build culprits dictionary from result
        culprits = {}
        if result.get("triton_culprit"):
            culprits["triton"] = result["triton_culprit"]
        if result.get("llvm_culprit"):
            culprits["llvm"] = result["llvm_culprit"]

        # Determine mode based on result
        if result.get("llvm_culprit"):
            mode = SummaryMode.FULL_WORKFLOW
        else:
            mode = SummaryMode.TRITON_BISECT

        # Build LLVMBumpInfo from result using correct fields
        llvm_bump_info = None
        if result.get("is_llvm_bump"):
            # Use llvm_bump (original bump info from Type Check), not llvm_range
            llvm_bump = result.get("llvm_bump", {})
            llvm_bump_info = LLVMBumpInfo(
                is_llvm_bump=True,
                old_hash=llvm_bump.get("old"),
                new_hash=llvm_bump.get("new"),
                triton_commit=result.get("triton_culprit"),
            )

        print_final_summary(
            mode=mode,
            culprits=culprits if culprits else None,
            llvm_bump_info=llvm_bump_info,
            log_dir=workflow.state.log_dir,
        )
        return 0
    except FileNotFoundError:
        print(f"State file not found: {state_path}")
        return 1
    except Exception as e:
        print(f"Error resuming bisect: {e}")
        return 1


def _handle_llvm_only(args: argparse.Namespace) -> int:
    """Handle --llvm-only mode: bisect only LLVM commits."""
    from tritonparse.bisect.llvm_bisector import LLVMBisectError, LLVMBisector
    from tritonparse.bisect.ui import BisectUI, SummaryMode, print_final_summary

    # Initialize TUI first
    ui = BisectUI(enabled=args.tui)

    # Variables to store results for summary after TUI exits
    culprit = None
    error_msg = None
    logger = None

    # Use context manager to start/stop TUI - entire workflow inside
    with ui:
        try:
            # Create logger inside TUI context
            logger = _create_logger(args.log_dir)

            # Configure logger for TUI mode (redirect output to TUI)
            if ui.is_tui_enabled:
                logger.configure_for_tui(ui.create_output_callback())

            ui.append_output(ui.get_tui_status_message())

            # Set initial progress: LLVM only mode has 1 phase
            ui.update_progress(
                phase="LLVM Bisect",
                phase_number=1,
                total_phases=1,
                log_dir=str(logger.log_dir),
                log_file=logger.module_log_path.name,
                command_log=logger.command_log_path.name,
            )

            # Create bisector (its logs will go to TUI)
            bisector = LLVMBisector(
                triton_dir=args.triton_dir,
                test_script=args.test_script,
                conda_env=args.conda_env,
                logger=logger,
                build_command=args.build_command,
            )

            # Run bisect
            culprit = bisector.run(
                triton_commit=args.triton_commit,
                good_llvm=args.good_llvm,
                bad_llvm=args.bad_llvm,
                output_callback=ui.create_output_callback(),
            )

            # Show result in TUI
            ui.append_output("")
            ui.append_output("=" * 60)
            ui.append_output("LLVM Bisect Result")
            ui.append_output("=" * 60)
            ui.append_output(f"Culprit LLVM commit: {culprit}")
            ui.append_output(f"Log directory: {args.log_dir}")
            ui.append_output("=" * 60)

        except LLVMBisectError as e:
            error_msg = str(e)
            ui.append_output(f"\nLLVM bisect failed: {e}")
        except Exception as e:
            error_msg = str(e)
            ui.append_output(f"\nUnexpected error: {e}")

    # TUI has exited, print final summary
    print_final_summary(
        mode=SummaryMode.LLVM_BISECT,
        culprits={"llvm": culprit} if culprit else None,
        llvm_bump_info=None,
        error_msg=error_msg,
        log_dir=args.log_dir,
        log_file=str(logger.module_log_path) if logger else None,
        command_log=str(logger.command_log_path) if logger else None,
    )

    return 0 if culprit else 1


def _handle_triton_bisect(args: argparse.Namespace) -> int:
    """Handle default mode: Triton bisect (or full workflow with --commits-csv)."""
    from tritonparse.bisect.commit_detector import CommitDetector
    from tritonparse.bisect.executor import ShellExecutor
    from tritonparse.bisect.triton_bisector import TritonBisectError, TritonBisector
    from tritonparse.bisect.ui import BisectUI, SummaryMode, print_final_summary

    # Check if this is full workflow mode
    if args.commits_csv:
        return _handle_full_workflow(args)

    # Initialize TUI first
    ui = BisectUI(enabled=args.tui)

    # Variables to store results for summary after TUI exits
    culprit = None
    llvm_bump_info = None
    error_msg = None
    logger = None

    # Use context manager to start/stop TUI - entire workflow inside
    with ui:
        try:
            # Create logger inside TUI context
            logger = _create_logger(args.log_dir)

            # Configure logger for TUI mode (redirect output to TUI)
            if ui.is_tui_enabled:
                logger.configure_for_tui(ui.create_output_callback())

            ui.append_output(ui.get_tui_status_message())

            # Set initial progress: Triton only mode has 2 phases
            ui.update_progress(
                phase="Triton Bisect",
                phase_number=1,
                total_phases=2,
                log_dir=str(logger.log_dir),
                log_file=logger.module_log_path.name,
                command_log=logger.command_log_path.name,
            )

            # Create bisector (its logs will go to TUI)
            bisector = TritonBisector(
                triton_dir=args.triton_dir,
                test_script=args.test_script,
                logger=logger,
                conda_env=args.conda_env,
                build_command=args.build_command,
            )

            # Run bisect
            culprit = bisector.run(
                good_commit=args.good,
                bad_commit=args.bad,
                output_callback=ui.create_output_callback(),
            )

            # Detect if culprit is an LLVM bump (Phase 2)
            ui.update_progress(
                phase="LLVM Bump Check",
                phase_number=2,
            )
            executor = ShellExecutor(logger)
            detector = CommitDetector(
                triton_dir=args.triton_dir,
                executor=executor,
                logger=logger,
            )
            llvm_bump_info = detector.detect(culprit)

            # Show result in TUI
            ui.append_output("")
            ui.append_output("=" * 60)
            ui.append_output("Triton Bisect Result")
            ui.append_output("=" * 60)
            ui.append_output(f"Culprit commit: {culprit}")

            # Show LLVM bump info if applicable
            if llvm_bump_info.is_llvm_bump:
                ui.append_output("")
                ui.append_output("⚠️  This commit is an LLVM bump!")
                ui.append_output(
                    f"  LLVM version: {llvm_bump_info.old_hash} -> {llvm_bump_info.new_hash}"
                )

            ui.append_output(f"Log directory: {args.log_dir}")
            ui.append_output("=" * 60)

        except TritonBisectError as e:
            error_msg = str(e)
            ui.append_output(f"\nTriton bisect failed: {e}")
        except Exception as e:
            error_msg = str(e)
            ui.append_output(f"\nUnexpected error: {e}")

    # TUI has exited, print final summary
    print_final_summary(
        mode=SummaryMode.TRITON_BISECT,
        culprits={"triton": culprit} if culprit else None,
        llvm_bump_info=llvm_bump_info,
        error_msg=error_msg,
        log_dir=args.log_dir,
        log_file=str(logger.module_log_path) if logger else None,
        command_log=str(logger.command_log_path) if logger else None,
    )

    return 0 if culprit else 1


def _handle_full_workflow(args: argparse.Namespace) -> int:
    """Handle full workflow mode (with --commits-csv)."""
    from tritonparse.bisect.commit_detector import LLVMBumpInfo
    from tritonparse.bisect.ui import SummaryMode, print_final_summary
    from tritonparse.bisect.workflow import BisectWorkflow, BisectWorkflowError

    try:
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

        # Build culprits dictionary from result
        culprits = {}
        if result.get("triton_culprit"):
            culprits["triton"] = result["triton_culprit"]
        if result.get("llvm_culprit"):
            culprits["llvm"] = result["llvm_culprit"]

        # Build LLVMBumpInfo from result using correct fields
        llvm_bump_info = None
        if result.get("is_llvm_bump"):
            # Use llvm_bump (original bump info from Type Check), not llvm_range
            llvm_bump = result.get("llvm_bump", {})
            llvm_bump_info = LLVMBumpInfo(
                is_llvm_bump=True,
                old_hash=llvm_bump.get("old"),
                new_hash=llvm_bump.get("new"),
                triton_commit=result.get("triton_culprit"),
            )

        print_final_summary(
            mode=SummaryMode.FULL_WORKFLOW,
            culprits=culprits if culprits else None,
            llvm_bump_info=llvm_bump_info,
            log_dir=args.log_dir,
        )
        return 0

    except BisectWorkflowError as e:
        print_final_summary(
            mode=SummaryMode.FULL_WORKFLOW,
            culprits=None,
            error_msg=str(e),
            log_dir=args.log_dir,
        )
        return 1
    except Exception as e:
        print_final_summary(
            mode=SummaryMode.FULL_WORKFLOW,
            culprits=None,
            error_msg=str(e),
            log_dir=args.log_dir,
        )
        return 1


def _handle_pair_test(args: argparse.Namespace) -> int:
    """Handle --pair-test mode: test commit pairs to find LLVM bisect range."""
    from pathlib import Path

    from tritonparse.bisect.executor import ShellExecutor
    from tritonparse.bisect.pair_tester import PairTester, PairTesterError
    from tritonparse.bisect.ui import BisectUI, print_final_summary

    # Initialize TUI
    ui = BisectUI(enabled=args.tui)

    # Variables to store results for summary after TUI exits
    result = None
    error_msg = None
    logger = None

    with ui:
        try:
            # Create logger inside TUI context
            logger = _create_logger(args.log_dir)

            # Configure logger for TUI mode
            if ui.is_tui_enabled:
                logger.configure_for_tui(ui.create_output_callback())

            ui.append_output(ui.get_tui_status_message())

            # Set initial progress
            ui.update_progress(
                phase="Pair Test",
                phase_number=1,
                total_phases=1,
                log_dir=str(logger.log_dir),
                log_file=logger.module_log_path.name,
                command_log=logger.command_log_path.name,
            )

            ui.append_output("")
            ui.append_output("=" * 60)
            ui.append_output("Pair Test Mode")
            ui.append_output("=" * 60)
            ui.append_output(f"CSV file: {args.commits_csv}")
            ui.append_output(f"LLVM range: {args.good_llvm} -> {args.bad_llvm}")
            ui.append_output("")

            # Create pair tester
            executor = ShellExecutor(logger)
            tester = PairTester(
                triton_dir=Path(args.triton_dir),
                test_script=Path(args.test_script),
                executor=executor,
                logger=logger,
                conda_env=args.conda_env,
                build_command=args.build_command,
            )

            # Run pair test with LLVM range filtering
            result = tester.test_from_csv(
                csv_path=Path(args.commits_csv),
                good_llvm=args.good_llvm,
                bad_llvm=args.bad_llvm,
                output_callback=ui.create_output_callback(),
            )

            # Show result in TUI
            ui.append_output("")
            ui.append_output("=" * 60)
            ui.append_output("Pair Test Result")
            ui.append_output("=" * 60)

            if result.all_passed:
                ui.append_output("✅ All pairs passed - no failing pair found")
            elif result.found_failing:
                ui.append_output(
                    f"📍 First failing pair: #{result.failing_index + 1} "
                    f"of {result.total_pairs}"
                )
                ui.append_output(f"   Triton commit: {result.triton_commit}")
                ui.append_output(
                    f"   LLVM range: {result.good_llvm} -> {result.bad_llvm}"
                )
            elif result.error_message:
                ui.append_output(f"❌ Error: {result.error_message}")

            ui.append_output(f"Log directory: {args.log_dir}")
            ui.append_output("=" * 60)

        except PairTesterError as e:
            error_msg = str(e)
            ui.append_output(f"\nPair test failed: {e}")
        except Exception as e:
            error_msg = str(e)
            ui.append_output(f"\nUnexpected error: {e}")

    # TUI has exited, print final summary
    from tritonparse.bisect.ui import SummaryMode

    print_final_summary(
        mode=SummaryMode.PAIR_TEST,
        pair_test_result=result,
        error_msg=error_msg,
        log_dir=args.log_dir,
        log_file=str(logger.module_log_path) if logger else None,
        command_log=str(logger.command_log_path) if logger else None,
    )

    # Return success if we found a failing pair or all passed
    if result:
        return 0 if (result.found_failing or result.all_passed) else 1
    return 1


def _create_logger(log_dir: str):
    """Create a BisectLogger instance."""
    from tritonparse.bisect.logger import BisectLogger

    return BisectLogger(log_dir)
