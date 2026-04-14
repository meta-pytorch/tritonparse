# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
CLI for the compat-build subcommand.

Provides command-line access to the compat_builder workflow for building
LLVM compatibility maps. Supports four modes:

- Default (build): Build a commits.csv for a single LLVM bump
- ``--resume``: Resume after providing a manual fix commit
- ``--verify``: Validate an existing CSV file
- ``--status``: Show current build status
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger
from tritonparse.compat_builder.ai_fixer import AICompatFixer
from tritonparse.compat_builder.builder import CompatBuilder, WaitingForFixError
from tritonparse.compat_builder.csv_manager import CSVManager
from tritonparse.compat_builder.state import CompatBuildState, CompatStateManager

logger: logging.Logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Argument definitions
# ------------------------------------------------------------------


def _add_compat_build_args(parser: argparse.ArgumentParser) -> None:
    """Register compat-build subcommand arguments."""

    # Mode switches (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a saved state after a manual fix",
    )
    mode_group.add_argument(
        "--verify",
        action="store_true",
        help="Validate an existing commits.csv file",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show current compat-build status",
    )

    # Common arguments
    parser.add_argument(
        "--triton-dir",
        type=str,
        help="Path to the Triton repository",
    )
    parser.add_argument(
        "--llvm-bump-commit",
        type=str,
        help="The Triton commit that bumped LLVM",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./commits.csv",
        help="Output CSV path (default: ./commits.csv)",
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
        default="./compat_build_logs",
        help="Log directory (default: ./compat_build_logs)",
    )

    # AI control
    parser.add_argument(
        "--ai",
        action="store_true",
        default=True,
        dest="use_ai",
        help="Enable AI automatic fix (default: enabled)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_false",
        dest="use_ai",
        help="Disable AI, manual fix only",
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default=None,
        help="LLM model for AI fix (default: auto-select)",
    )

    # Worktree control
    parser.add_argument(
        "--worktree-root",
        type=str,
        default=None,
        help="Root directory for auto-created worktrees",
    )
    parser.add_argument(
        "--worktree-path",
        type=str,
        default=None,
        help="Explicit path to an existing compat worktree",
    )

    # Resume arguments
    parser.add_argument(
        "--state",
        type=str,
        help="Path to state JSON file (for --resume / --status)",
    )
    parser.add_argument(
        "--fix-commit",
        type=str,
        help="Fix commit hash to apply when resuming",
    )

    # Verify arguments
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to CSV file to verify (for --verify)",
    )

    # TUI control
    parser.add_argument(
        "--tui",
        action="store_true",
        default=True,
        dest="tui",
        help="Enable TUI output (default)",
    )
    parser.add_argument(
        "--no-tui",
        action="store_false",
        dest="tui",
        help="Disable TUI output",
    )


# ------------------------------------------------------------------
# Argument validation
# ------------------------------------------------------------------


def _validate_compat_build_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Validate argument combinations per mode."""
    if args.status:
        return

    if args.verify:
        if not args.csv:
            parser.error("--verify requires --csv")
        return

    if args.resume:
        if not args.state:
            parser.error("--resume requires --state")
        if not args.fix_commit:
            parser.error("--resume requires --fix-commit")
        return

    # Default build mode
    if not args.triton_dir:
        parser.error("--triton-dir is required for build mode")
    if not args.llvm_bump_commit:
        parser.error("--llvm-bump-commit is required for build mode")


# ------------------------------------------------------------------
# Command entry point
# ------------------------------------------------------------------


def compat_build_command(args: argparse.Namespace) -> int:
    """Main entry point for the compat-build subcommand.

    Routes to the appropriate handler based on mode flags.

    Returns:
        Exit code: 0 = success, 1 = manual fix needed or error.
    """
    if args.status:
        return _handle_status(args)
    if args.verify:
        return _handle_verify(args)
    if args.resume:
        return _handle_resume(args)
    return _handle_build(args)


# ------------------------------------------------------------------
# Mode handlers
# ------------------------------------------------------------------


def _handle_build(args: argparse.Namespace) -> int:
    """Default mode: build commits.csv for a single LLVM bump."""
    build_logger = BisectLogger(args.log_dir)

    ai_fixer_factory = (
        _create_ai_fixer_factory(args, build_logger) if args.use_ai else None
    )

    builder = CompatBuilder(
        triton_dir=args.triton_dir,
        llvm_bump_commit=args.llvm_bump_commit,
        output_csv=args.output_csv,
        logger=build_logger,
        conda_env=args.conda_env,
        worktree_root=args.worktree_root,
        worktree_path=args.worktree_path,
        ai_fixer_factory=ai_fixer_factory,
    )

    builder.initialize()
    return _run_build_loop(builder)


def _handle_resume(args: argparse.Namespace) -> int:
    """Resume mode: apply a fix commit and continue the build loop."""
    state = CompatBuildState.load(Path(args.state))
    build_logger = BisectLogger(state.log_dir)

    ai_fixer_factory = (
        _create_ai_fixer_factory(args, build_logger) if args.use_ai else None
    )

    builder = CompatBuilder(
        triton_dir=state.triton_dir,
        llvm_bump_commit=state.llvm_bump_commit,
        output_csv=state.output_csv,
        logger=build_logger,
        conda_env=state.conda_env,
        worktree_root=state.worktree_root,
        worktree_path=state.worktree_path,
        ai_fixer_factory=ai_fixer_factory,
    )
    builder.state = state
    builder.conda_prefix = builder._resolve_conda_prefix()
    builder._validate_conda_env()
    builder.apply_fix(args.fix_commit)
    return _run_build_loop(builder)


def _handle_status(args: argparse.Namespace) -> int:
    """Status mode: display current compat-build state."""
    state_path: Path | None = None
    if args.state:
        state_path = Path(args.state)
    else:
        state_path = CompatStateManager.find_latest_state(args.log_dir)

    if state_path is None or not state_path.exists():
        print("No compat-build state found.", file=sys.stderr)
        return 1

    state = CompatBuildState.load(state_path)
    old_llvm = state.old_llvm or "?"
    new_llvm = state.new_llvm or "?"
    print(f"Phase:     {state.phase.value}")
    print(f"Bump:      {state.llvm_bump_commit}")
    print(f"LLVM:      {old_llvm[:12]} -> {new_llvm[:12]}")
    print(f"Pairs:     {len(state.pairs)}")
    print(f"Triton:    {state.current_triton}")
    print(f"Worktree:  {state.worktree_path}")
    if state.last_incompatible_llvm:
        print(f"Blocked:   {state.last_incompatible_llvm[:12]}")
    return 0


def _handle_verify(args: argparse.Namespace) -> int:
    """Verify mode: validate an existing commits.csv."""
    csv_path = Path(args.csv)
    try:
        mgr = CSVManager(csv_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    errors = mgr.validate_monotonic_pairs() + mgr.validate_terminal_boundary()
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    meta = mgr.load_metadata()
    pairs = mgr.load_pairs()
    print(f"CSV valid: {len(pairs)} pairs, bump={meta.llvm_bump_commit[:12]}")
    return 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _run_build_loop(builder: CompatBuilder) -> int:
    """Run the find-record-fix loop until completion or manual fix needed.

    Returns:
        0 on successful CSV generation, 1 if manual fix is required.
    """
    while not builder.is_complete():
        incompatible = builder.find_next_incompatible()
        if incompatible is None:
            break
        builder.record_pair()
        try:
            builder.fix_incompatibility(incompatible)
        except WaitingForFixError as e:
            print(f"\nManual fix required. State saved to: {e.state_path}")
            print(
                "Resume with: tritonparseoss compat-build --resume "
                f"--state {e.state_path} --fix-commit <hash>"
            )
            return 1

    csv_path = builder.generate_csv()
    print(f"Generated: {csv_path}")
    return 0


def _create_ai_fixer_factory(
    args: argparse.Namespace,
    build_logger: BisectLogger,
) -> Callable[[str], AICompatFixer]:
    """Return a factory that creates an AICompatFixer for a given worktree path.

    The factory delays creation until the worktree path is known (after
    initialize()), so the fixer's triton_dir and llvm_dir point to the
    correct worktree rather than the original Triton repository.
    """

    def factory(worktree_path: str) -> AICompatFixer:
        return AICompatFixer(
            triton_dir=worktree_path,
            executor=ShellExecutor(build_logger),
            bisect_logger=build_logger,
            model=args.ai_model,
        )

    return factory
