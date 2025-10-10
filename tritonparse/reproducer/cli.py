import argparse

from tritonparse.reproducer.types import KernelImportMode


def _add_reproducer_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments for the reproducer to a parser."""
    parser.add_argument("input", help="Path to the ndjson/ndjson.gz log file")
    parser.add_argument(
        "--line",
        type=int,
        default=1,
        help=(
            "The line number (1-based) of the launch event in the input file to reproduce. "
            "Defaults to 1."
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
            "Defaults to 'default'."
        ),
    )
