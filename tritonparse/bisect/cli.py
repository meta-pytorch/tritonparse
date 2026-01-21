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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logger import BisectLogger
    from .state import BisectState
    from .ui import BisectUI


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


def _handle_status(args: argparse.Namespace) -> int:
    """
    Handle --status mode: show current bisect status.

    If --state is not provided, searches for the most recent state file
    in the log directory (default: ./bisect_logs).

    Args:
        args: Parsed arguments with optional 'state' and 'log_dir'.

    Returns:
        0 on success or no state found, 1 on error.
    """
    from .state import StateManager

    # Determine state file path
    state_path = args.state
    if state_path is None:
        # Search for latest state in log directory
        log_dir = getattr(args, "log_dir", "./bisect_logs")
        found_path = StateManager.find_latest_state(log_dir)
        if found_path is None:
            print(f"No state file found in: {log_dir}")
            print("No bisect in progress.")
            return 0
        state_path = str(found_path)

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


def _create_logger(log_dir: str) -> "BisectLogger":
    """
    Create a BisectLogger instance.

    Args:
        log_dir: Directory for log files.

    Returns:
        Configured BisectLogger instance.
    """
    from .logger import BisectLogger

    return BisectLogger(log_dir=log_dir)


def _handle_full_workflow(args: argparse.Namespace) -> int:
    """
    Handle full workflow mode (with --commits-csv).

    Orchestrates all 4 phases with TUI support:
    1. Triton Bisect - Find culprit Triton commit
    2. Type Check - Detect if it's an LLVM bump
    3. Pair Test - Find LLVM bisect range (if LLVM bump)
    4. LLVM Bisect - Find culprit LLVM commit (if LLVM bump)

    This function is the entry point for full workflow mode. It initializes
    the TUI, logger, and state, then delegates to _orchestrate_workflow()
    for the actual 4-phase execution.

    Args:
        args: Parsed arguments including triton_dir, test_script, good, bad,
              commits_csv, conda_env, log_dir, build_command, tui, etc.

    Returns:
        0 on success (all phases completed), 1 on failure.
    """
    from pathlib import Path

    from .state import BisectState
    from .ui import BisectUI

    # Initialize TUI
    ui = BisectUI(enabled=args.tui)

    # Create logger
    logger = _create_logger(args.log_dir)

    # Create state with all configuration
    # Paths are resolved to absolute paths for consistency
    state = BisectState(
        triton_dir=str(Path(args.triton_dir).resolve()),
        test_script=str(Path(args.test_script).resolve()),
        good_commit=args.good,
        bad_commit=args.bad,
        commits_csv=str(Path(args.commits_csv).resolve()),
        conda_env=args.conda_env,
        log_dir=str(Path(args.log_dir).resolve()),
        build_command=args.build_command,
        session_name=logger.session_name,  # Links state file to log files
    )

    # Run orchestration
    return _orchestrate_workflow(state, ui, logger)


def _orchestrate_workflow(
    state: "BisectState",
    ui: "BisectUI",
    logger: "BisectLogger",
) -> int:
    """
    Core orchestration logic for the 4-phase bisect workflow.

    This function is shared by both _handle_full_workflow() (new runs)
    and _handle_resume() (resumed runs).

    Phases:
    1. Triton Bisect - Find culprit Triton commit
    2. Type Check - Detect if it's an LLVM bump
    3. Pair Test - Find LLVM bisect range (if LLVM bump)
    4. LLVM Bisect - Find culprit LLVM commit (if LLVM bump)

    The function uses 'if' (not 'elif') for phase checks to allow
    resumed runs to continue through subsequent phases.

    Args:
        state: BisectState containing configuration and progress.
        ui: BisectUI instance for TUI display.
        logger: BisectLogger for logging.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    from pathlib import Path

    from .commit_detector import CommitDetector, LLVMBumpInfo
    from .executor import ShellExecutor
    from .llvm_bisector import LLVMBisector
    from .pair_tester import PairTester
    from .state import BisectPhase
    from .triton_bisector import TritonBisector
    from .ui import print_final_summary, SummaryMode

    # Variables for final summary
    culprits: dict[str, str] = {}
    llvm_bump_info = None
    error_msg = None

    with ui:
        try:
            # Configure logger for TUI mode
            if ui.is_tui_enabled:
                logger.configure_for_tui(ui.create_output_callback())

            ui.append_output(ui.get_tui_status_message())

            executor = ShellExecutor(logger)

            # Determine total phases (4 for full workflow)
            total_phases = 4

            # ========== Phase 1: Triton Bisect ==========
            if state.phase == BisectPhase.TRITON_BISECT:
                ui.update_progress(
                    phase="Triton Bisect",
                    phase_number=1,
                    total_phases=total_phases,
                    log_dir=str(logger.log_dir),
                    log_file=logger.module_log_path.name,
                    command_log=logger.command_log_path.name,
                )

                bisector = TritonBisector(
                    triton_dir=state.triton_dir,
                    test_script=state.test_script,
                    conda_env=state.conda_env,
                    logger=logger,
                    build_command=state.build_command,
                )

                state.triton_culprit = bisector.run(
                    good_commit=state.good_commit,
                    bad_commit=state.bad_commit,
                    output_callback=ui.create_output_callback(),
                )

                state.phase = BisectPhase.TYPE_CHECK
                state.save(session_name=logger.session_name)

            # ========== Phase 2: Type Check ==========
            if state.phase == BisectPhase.TYPE_CHECK:
                ui.update_progress(
                    phase="LLVM Bump Check",
                    phase_number=2,
                    total_phases=total_phases,
                )

                detector = CommitDetector(
                    triton_dir=state.triton_dir,
                    executor=executor,
                    logger=logger,
                )
                bump_info = detector.detect(state.triton_culprit)

                state.is_llvm_bump = bump_info.is_llvm_bump
                state.old_llvm_hash = bump_info.old_hash
                state.new_llvm_hash = bump_info.new_hash

                if not bump_info.is_llvm_bump:
                    ui.append_output("")
                    ui.append_output("Commit is NOT an LLVM bump. Workflow complete.")
                    state.phase = BisectPhase.COMPLETED
                else:
                    ui.append_output("")
                    ui.append_output("⚠️  This commit is an LLVM bump!")
                    ui.append_output(
                        f"  LLVM version: {bump_info.old_hash[:12]} -> "
                        f"{bump_info.new_hash[:12]}"
                    )
                    ui.append_output("Proceeding to pair testing...")
                    state.phase = BisectPhase.PAIR_TEST

                state.save(session_name=logger.session_name)

            # ========== Phase 3: Pair Test ==========
            if state.phase == BisectPhase.PAIR_TEST:
                ui.update_progress(
                    phase="Pair Test",
                    phase_number=3,
                    total_phases=total_phases,
                )

                tester = PairTester(
                    triton_dir=Path(state.triton_dir),
                    test_script=Path(state.test_script),
                    executor=executor,
                    logger=logger,
                    conda_env=state.conda_env,
                    build_command=state.build_command,
                )

                # Use Type Check results to filter pairs
                result = tester.test_from_csv(
                    csv_path=Path(state.commits_csv),
                    good_llvm=state.old_llvm_hash,
                    bad_llvm=state.new_llvm_hash,
                    output_callback=ui.create_output_callback(),
                )

                if result.all_passed:
                    ui.append_output("")
                    ui.append_output("All pairs passed. No LLVM regression found.")
                    state.phase = BisectPhase.COMPLETED
                elif result.found_failing:
                    state.failing_pair_index = result.failing_index
                    state.good_llvm = result.good_llvm
                    state.bad_llvm = result.bad_llvm
                    state.triton_commit_for_llvm = result.triton_commit

                    ui.append_output("")
                    ui.append_output(
                        f"Found first failing pair at index {result.failing_index}"
                    )
                    ui.append_output(
                        f"LLVM bisect range: {result.good_llvm[:12]} -> "
                        f"{result.bad_llvm[:12]}"
                    )
                    state.phase = BisectPhase.LLVM_BISECT
                else:
                    raise RuntimeError(f"Pair test failed: {result.error_message}")

                state.save(session_name=logger.session_name)

            # ========== Phase 4: LLVM Bisect ==========
            if state.phase == BisectPhase.LLVM_BISECT:
                ui.update_progress(
                    phase="LLVM Bisect",
                    phase_number=4,
                    total_phases=total_phases,
                )

                bisector = LLVMBisector(
                    triton_dir=state.triton_dir,
                    test_script=state.test_script,
                    conda_env=state.conda_env,
                    logger=logger,
                )

                # Use the Triton commit from pair testing
                triton_commit = state.triton_commit_for_llvm or state.triton_culprit

                state.llvm_culprit = bisector.run(
                    triton_commit=triton_commit,
                    good_llvm=state.good_llvm,
                    bad_llvm=state.bad_llvm,
                    output_callback=ui.create_output_callback(),
                )

                state.phase = BisectPhase.COMPLETED
                state.save(session_name=logger.session_name)

            # ========== Show Final Result in TUI ==========
            ui.append_output("")
            ui.append_output("=" * 60)
            ui.append_output("Full Workflow Complete!")
            ui.append_output("=" * 60)
            ui.append_output(f"Triton culprit: {state.triton_culprit}")
            if state.is_llvm_bump and state.llvm_culprit:
                ui.append_output(f"LLVM culprit: {state.llvm_culprit}")
            ui.append_output(f"Log directory: {state.log_dir}")

            # Prepare data for final summary
            if state.triton_culprit:
                culprits["triton"] = state.triton_culprit
            if state.llvm_culprit:
                culprits["llvm"] = state.llvm_culprit

            if state.is_llvm_bump:
                llvm_bump_info = LLVMBumpInfo(
                    is_llvm_bump=True,
                    old_hash=state.old_llvm_hash,
                    new_hash=state.new_llvm_hash,
                    triton_commit=state.triton_culprit,
                )

        except Exception as e:
            error_msg = str(e)
            ui.append_output(f"\nWorkflow failed: {e}")
            if state:
                state.phase = BisectPhase.FAILED
                state.error_message = error_msg
                state.save(session_name=logger.session_name)
            # Cleanup git bisect state on failure
            _cleanup_bisect_state(state, logger)

    # TUI has exited, print final summary
    print_final_summary(
        mode=SummaryMode.FULL_WORKFLOW,
        culprits=culprits if culprits else None,
        llvm_bump_info=llvm_bump_info,
        error_msg=error_msg,
        log_dir=state.log_dir,
        log_file=str(logger.module_log_path) if logger else None,
        command_log=str(logger.command_log_path) if logger else None,
        elapsed_time=ui.progress.elapsed_seconds,
        logger=logger,
    )

    return 0 if state and state.phase == BisectPhase.COMPLETED else 1


def _cleanup_bisect_state(state: "BisectState", logger: "BisectLogger") -> None:
    """
    Clean up git bisect state on failure.

    Resets the git bisect state in both Triton and LLVM repositories
    to avoid leaving the repos in an inconsistent state.

    Args:
        state: BisectState containing repository paths.
        logger: BisectLogger for logging.
    """
    from pathlib import Path

    from .executor import ShellExecutor

    executor = ShellExecutor(logger)

    # Reset Triton repo bisect state
    executor.run_command(
        ["git", "bisect", "reset"],
        cwd=state.triton_dir,
    )

    # Reset LLVM repo bisect state (if it exists)
    llvm_dir = Path(state.triton_dir) / "llvm-project"
    if llvm_dir.exists():
        executor.run_command(
            ["git", "bisect", "reset"],
            cwd=str(llvm_dir),
        )


def _handle_resume(args: argparse.Namespace) -> int:
    """
    Handle --resume mode: resume from saved state with TUI support.

    This function loads a previously saved bisect state and continues
    the workflow from where it left off. It's useful when a bisect
    was interrupted or needs to be resumed after a system restart.

    Args:
        args: Parsed arguments including state (path to state file),
              tui (whether to use Rich TUI), and optional log_dir.

    Returns:
        0 on success, 1 on failure.
    """
    from .state import StateManager
    from .ui import BisectUI

    # Determine state file path
    state_path = args.state
    if state_path is None:
        # Search for latest state in log directory
        log_dir = getattr(args, "log_dir", "./bisect_logs")
        found_path = StateManager.find_latest_state(log_dir)
        if found_path is None:
            print(f"No state file found in: {log_dir}")
            print("Use --state to specify a state file path.")
            return 1
        state_path = str(found_path)

    try:
        # Load existing state
        state = StateManager.load(state_path)

        # Create logger using the log_dir from state
        logger = _create_logger(state.log_dir)

        # Initialize TUI (supports --tui/--no-tui flags)
        ui = BisectUI(enabled=args.tui)

        # Log resume information
        logger.info(f"Resuming from state: {state_path}")
        logger.info(f"Current phase: {state.phase.value}")

        # Run orchestration with TUI support
        return _orchestrate_workflow(state, ui, logger)

    except FileNotFoundError:
        print(f"State file not found: {state_path}")
        return 1
    except Exception as e:
        print(f"Error resuming bisect: {e}")
        return 1


def _handle_triton_bisect(args: argparse.Namespace) -> int:
    """
    Handle default mode: Triton bisect (or full workflow with --commits-csv).

    This function performs a two-phase operation:
    1. Phase 1: Triton Bisect - Find the culprit Triton commit
    2. Phase 2: LLVM Bump Check - Detect if the culprit is an LLVM version bump

    If --commits-csv is provided, delegates to _handle_full_workflow() instead.

    Args:
        args: Parsed arguments including triton_dir, test_script, good, bad, etc.

    Returns:
        0 on success, 1 on failure.
    """
    # Check if this is full workflow mode
    if args.commits_csv:
        return _handle_full_workflow(args)

    from .commit_detector import CommitDetector
    from .executor import ShellExecutor
    from .triton_bisector import TritonBisectError, TritonBisector
    from .ui import BisectUI, print_final_summary, SummaryMode

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
                    f"  LLVM version: {llvm_bump_info.old_hash} -> "
                    f"{llvm_bump_info.new_hash}"
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
        elapsed_time=ui.progress.elapsed_seconds,
        logger=logger,
    )

    return 0 if culprit else 1


def bisect_command(args: argparse.Namespace) -> int:
    """
    Execute the bisect command based on parsed arguments.

    This is the main entry point for the bisect subcommand. It routes
    to the appropriate handler based on the mode flags.

    Mode priority (checked in order):
    1. --status: Show current bisect status
    2. --resume: Resume from saved state
    3. --llvm-only: LLVM-only bisect
    4. --pair-test: Test commit pairs from CSV
    5. Default: Triton bisect (or full workflow with --commits-csv)

    The mode flags are mutually exclusive (enforced by argparse).

    Args:
        args: Parsed command-line arguments from _add_bisect_args().

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


def _handle_pair_test(args: argparse.Namespace) -> int:
    """
    Handle --pair-test mode: test commit pairs to find LLVM bisect range.

    This function tests (Triton, LLVM) commit pairs from a CSV file to find
    the first failing combination. The result can be used to determine the
    LLVM bisect range for subsequent LLVM bisect.

    Args:
        args: Parsed arguments including triton_dir, test_script, commits_csv,
              good_llvm, bad_llvm, etc.

    Returns:
        0 on success (found failing pair or all passed), 1 on failure.
    """
    from pathlib import Path

    from .executor import ShellExecutor
    from .pair_tester import PairTester, PairTesterError
    from .ui import BisectUI, print_final_summary, SummaryMode

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

            # Show mode info
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
    print_final_summary(
        mode=SummaryMode.PAIR_TEST,
        pair_test_result=result,
        error_msg=error_msg,
        log_dir=args.log_dir,
        log_file=str(logger.module_log_path) if logger else None,
        command_log=str(logger.command_log_path) if logger else None,
        elapsed_time=ui.progress.elapsed_seconds,
        logger=logger,
    )

    # Return success if we found a failing pair or all passed
    if result:
        return 0 if (result.found_failing or result.all_passed) else 1
    return 1


def _handle_llvm_only(args: argparse.Namespace) -> int:
    """
    Handle --llvm-only mode: bisect only LLVM commits.

    This function performs LLVM bisect without first running Triton bisect.
    It's useful when you already know the Triton commit to use and want to
    find the culprit LLVM commit directly.

    Args:
        args: Parsed arguments including triton_dir, test_script,
              triton_commit (optional), good_llvm, bad_llvm, etc.

    Returns:
        0 on success, 1 on failure.
    """
    from .llvm_bisector import LLVMBisectError, LLVMBisector
    from .ui import BisectUI, print_final_summary, SummaryMode

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
        elapsed_time=ui.progress.elapsed_seconds,
        logger=logger,
    )

    return 0 if culprit else 1
