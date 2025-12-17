#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
from importlib.metadata import PackageNotFoundError, version

from .common import is_fbcode
from .info.cli import _add_info_args, info_command
from .reproducer.cli import _add_reproducer_args
from .reproducer.orchestrator import reproduce
from .utils import _add_parse_args, unified_parse


def _get_package_version() -> str:
    try:
        return version("tritonparse")
    except PackageNotFoundError:
        return "0+unknown"


def main():
    pkg_version = _get_package_version()

    # Use different command name for fbcode vs OSS
    prog_name = "tritonparse" if is_fbcode() else "tritonparseoss"

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description=(
            "TritonParse: parse structured logs and generate minimal reproducers"
        ),
        epilog=(
            "Examples:\n"
            f"  {prog_name} parse /path/to/logs --out parsed_output\n"
            f"  {prog_name} reproduce /path/to/trace.ndjson --line 1 --out-dir repro_output\n"
            f"  {prog_name} info /path/to/trace.ndjson\n"
            f"  {prog_name} info /path/to/trace.ndjson --kernel matmul_kernel\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {pkg_version}",
        help="Show program's version number and exit",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse subcommand
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse triton structured logs",
        conflict_handler="resolve",
    )
    _add_parse_args(parse_parser)
    parse_parser.set_defaults(func="parse")

    # reproduce subcommand
    repro_parser = subparsers.add_parser(
        "reproduce",
        help="Build reproducer from trace file",
    )
    _add_reproducer_args(repro_parser)
    repro_parser.set_defaults(func="reproduce")

    # info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Query kernel information from trace file",
    )
    _add_info_args(info_parser)
    info_parser.set_defaults(func="info")

    args = parser.parse_args()

    # Log usage for CLI invocations
    if is_fbcode():
        from tritonparse.fb.utils import usage_report_logger

        usage_report_logger()

    if args.func == "parse":
        parse_args = {
            k: v for k, v in vars(args).items() if k not in ["command", "func"]
        }
        # skip_logger=True because we already logged above
        unified_parse(**parse_args, skip_logger=True)
    elif args.func == "reproduce":
        # Check mutual exclusivity between --line and --kernel/--launch-id
        if args.kernel and args.line != 0:
            repro_parser.error("--line and --kernel/--launch-id are mutually exclusive")

        replacer = None
        if args.use_fbcode:
            from tritonparse.fb.reproducer.replacer import FBCodePlaceholderReplacer

            replacer = FBCodePlaceholderReplacer()
            print(f"Using FBCode placeholder replacer for template: {args.template}")

        reproduce(
            input_path=args.input,
            line_index=args.line if not args.kernel else 0,
            out_dir=args.out_dir,
            template=args.template,
            kernel_name=args.kernel,
            launch_id=args.launch_id if args.kernel else 0,
            kernel_import=args.kernel_import,
            replacer=replacer,
            skip_logger=True,  # Already logged above
        )
    elif args.func == "info":
        info_command(
            input_path=args.input,
            kernel_name=args.kernel,
            skip_logger=True,
            args_list=args.args_list,
        )  # Already logged above
    else:
        raise RuntimeError(f"Unknown command: {args.func}")


if __name__ == "__main__":
    main()  # pragma: no cover
