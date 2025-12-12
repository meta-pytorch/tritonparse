#  Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

from .common import (
    copy_local_to_tmpdir,
    is_fbcode,
    parse_logs,
    print_parsed_files_summary,
    RankConfig,
    save_logs,
)
from .source_type import Source, SourceType


def _add_parse_args(parser: argparse.ArgumentParser) -> None:
    """Add common 'parse' subcommand arguments to a parser."""
    parser.add_argument(
        "source",
        help=(
            "Source of torch logs to be analyzed. It is expected to path to a local "
            "directory or log"
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Output directory.",
        type=str,
    )
    parser.add_argument(
        "--overwrite",
        help=(
            "Delete out directory if it already exists. Only does something if --out is set"
        ),
        action="store_true",
    )
    parser.add_argument("-r", "--rank", help="Rank of logs to be analyzed", type=int)
    parser.add_argument(
        "--all-ranks",
        help="Analyze all ranks",
        action="store_true",
    )
    parser.add_argument("-v", "--verbose", help="Verbose logging", action="store_true")
    if is_fbcode():
        from tritonparse.fb.utils import append_parser

        append_parser(parser)


def oss_run(
    source: str,
    out: Optional[str] = None,
    overwrite: Optional[bool] = False,
    rank: Optional[int] = None,
    all_ranks: bool = False,
    verbose: bool = False,
    split_inductor_compilations: bool = True,
    skip_logger: bool = True,
):
    """
    Main function for tritonparse. It is for OSS only.

    Args:
        source: Source of torch logs to be analyzed (required)
        out: Output directory
        overwrite: Delete out directory if it already exists
        rank: Rank of logs to be analyzed
        all_ranks: Analyze all ranks
        verbose: Verbose logging
        skip_logger: Unused in OSS, kept for API compatibility.
    """
    source = Source(source, verbose)
    rank_config = RankConfig.from_cli_args(rank, all_ranks, source.type)

    # Check output directory early if specified
    if out is not None:
        out_dir = Path(out)
        if out_dir.exists():
            if not overwrite:
                raise RuntimeError(
                    f"{out_dir} already exists, pass --overwrite to overwrite"
                )
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # For signpost logging (not implemented in Python version)

    if source.type == SourceType.LOCAL:
        local_path = source.value
        # Copy the results to a temp directory, then parse them
        logs = copy_local_to_tmpdir(local_path, verbose)

    elif source.type == SourceType.LOCAL_FILE:
        local_path = source.value
        # Copy the single file to a temp directory, then parse it
        logs = copy_local_to_tmpdir(local_path, verbose)

    parsed_log_dir, _ = parse_logs(
        logs,
        rank_config,
        verbose,
        split_inductor_compilations=split_inductor_compilations,
    )
    if out is not None:
        save_logs(Path(out), parsed_log_dir, overwrite, verbose)
    # Print beautiful summary of all parsed files
    if out is not None:
        out_dir = str(Path(out).absolute())
    else:
        out_dir = str(Path(parsed_log_dir).absolute())
    print_parsed_files_summary(out_dir)
    return None


def unified_parse(
    source: str,
    out: Optional[str] = None,
    overwrite: Optional[bool] = False,
    rank: Optional[int] = None,
    all_ranks: bool = False,
    verbose: bool = False,
    split_inductor_compilations: bool = True,
    skip_logger: bool = False,
    **kwargs,
):
    """
    Unified parse function that provides a flexible interface for parsing triton logs.

    Args:
        source: Input directory containing logs to parse.
        out: Output directory for parsed results. By default, parsed logs will be saved to a temporary directory.
        overwrite: Whether to overwrite existing output directory
        rank: Specific rank to analyze
        all_ranks: Whether to analyze all ranks
        verbose: Whether to enable verbose logging
        skip_logger: Whether to skip usage logging (default: False).
    """
    # Log usage for API invocations
    if not skip_logger and is_fbcode():
        from tritonparse.fb.utils import usage_report_logger

        usage_report_logger()

    # Choose the appropriate parse function
    if is_fbcode():
        from tritonparse.fb.utils import fb_run as parse
    else:
        parse = oss_run

    output = parse(
        source=source,
        out=out,
        overwrite=overwrite,
        rank=rank,
        all_ranks=all_ranks,
        verbose=verbose,
        split_inductor_compilations=split_inductor_compilations,
        skip_logger=skip_logger,
        **kwargs,
    )
    return output
