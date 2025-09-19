import argparse


def _add_reproducer_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for the reproducer to a parser."""
    parser.add_argument("input", help="Path to the ndjson/ndjson.gz log file")
    parser.add_argument(
        "--line-index",
        type=int,
        help="The line number of the launch event in the input file to reproduce.",
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
