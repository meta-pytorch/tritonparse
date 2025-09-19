#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

from tritonparse.reproducer.cli import _add_reproducer_args
from tritonparse.reproducer.orchestrator import reproducer as run_reproducer

from tritonparse.utils import _add_parse_args, unified_parse


# We need this as an entrace for fbpkg
def main():
    parser = argparse.ArgumentParser(description="tritonparse CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse triton structured logs")
    _add_parse_args(parse_parser)
    parse_parser.set_defaults(func="parse")

    repro_parser = subparsers.add_parser(
        "reproduce", help="Build reproducer from trace file"
    )
    _add_reproducer_args(repro_parser)
    repro_parser.set_defaults(func="reproduce")

    args = parser.parse_args()

    if args.func == "parse":
        parse_kwargs = {
            "source": args.source,
            "out": getattr(args, "out", None),
            "overwrite": getattr(args, "overwrite", False),
            "rank": getattr(args, "rank", None),
            "all_ranks": getattr(args, "all_ranks", False),
            "verbose": getattr(args, "verbose", False),
        }
        unified_parse(**parse_kwargs)
    elif args.func == "reproduce":
        run_reproducer(
            input_path=args.ndjson_file,
            line_index=args.line_index,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    # Do not add code here, it won't be run. Add them to the function called below.
    main()  # pragma: no cover
