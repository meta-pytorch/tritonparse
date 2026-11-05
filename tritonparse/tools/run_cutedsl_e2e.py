# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Launch the CuTeDSL E2E workload with its external Python/CUDA environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tritonparse-root", type=Path, required=True)
    args, workload_args = parser.parse_known_args()
    # absolute(), not resolve(): a virtualenv's python is a symlink, and
    # following it would start the base interpreter without the venv.
    interpreter = args.python.absolute()
    source_root = args.tritonparse_root.resolve()
    if not interpreter.is_file():
        parser.error(f"Python interpreter does not exist: {interpreter}")
    if not (source_root / "tritonparse/tools/cutedsl_hopper_e2e.py").is_file():
        parser.error(f"Tritonparse workload is missing under {source_root}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    # Keep the working directory off sys.path: buck runs from the repository
    # root, whose top-level directories (e.g. dill) shadow installed packages.
    environment["PYTHONSAFEPATH"] = "1"
    os.execve(
        interpreter,
        [
            str(interpreter),
            "-m",
            "tritonparse.tools.cutedsl_hopper_e2e",
            *workload_args,
        ],
        environment,
    )
