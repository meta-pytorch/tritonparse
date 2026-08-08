#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""The environment a run happened in, printed at the top of every check.

A reviewer reporting "C6 gave me a different number" is telling us something we can only
act on if we know what they ran.  Most of the differences we expect -- and the one we
have already seen in practice, where a different machine autotunes a different number of
launches and so records a different number of trace records -- are explained entirely by
the lines below.  So every check prints them, and every log file therefore carries them.

Everything here has to be safe for a reviewer to paste to us.  Artifact-evaluation
reviewers are anonymous to authors, and a home directory or a hostname deanonymises one
instantly -- ``/home/jsmith`` and ``gpu3.cs.example.edu`` say who ran this and answer no
question we would ask.  So no hostname, no absolute paths, and environments are described
by kind rather than by location.  ``ae_logging.sh`` sweeps the same things out of the log
file as a backstop, since the checks print paths of their own.

Nothing here is allowed to fail the caller: a missing torch prints as unknown rather
than raising.  ``neutral_platform`` keeps the vendor build tag out of the kernel string.

    python scripts/ae_env.py                  # the block
    python scripts/ae_env.py --full DIR       # the block, plus DIR/environment.txt
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ae_platform import neutral_platform  # noqa: E402


def _version(module: str) -> str:
    try:
        mod = __import__(module)
    except Exception as exc:                                     # noqa: BLE001
        return f"not importable ({type(exc).__name__})"
    version = getattr(mod, "__version__", None)
    if version:
        return str(version)
    # setuptools-scm projects do not always expose __version__ on the package.
    try:
        from importlib.metadata import version as _dist_version
        return str(_dist_version(module))
    except Exception:                                            # noqa: BLE001
        return "unknown"


ARTIFACT = Path(__file__).resolve().parent.parent


def _conda_version() -> str | None:
    """conda's own version, which is worth knowing and gives nothing away."""
    candidates = ["conda"]
    root = ARTIFACT / ".conda-root"
    if root.is_file():
        try:
            candidates.insert(0, str(Path(root.read_text().strip()) / "bin" / "conda"))
        except OSError:
            pass
    for exe in candidates:
        got = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             check=False)
        if got.returncode == 0 and got.stdout.strip():
            return got.stdout.strip().replace("conda ", "")
    return None


def _conda_prefix() -> str | None:
    """The conda environment the *running interpreter* belongs to, if any.

    Not ``$CONDA_PREFIX``: a shell can carry a stale one from an environment that has
    nothing to do with the python actually executing this, and reporting that would send
    a reviewer -- or us, reading their log -- after the wrong environment entirely.
    """
    return sys.prefix if (Path(sys.prefix) / "conda-meta").is_dir() else None


def _environment_kind() -> str:
    """Which *kind* of environment this is, never where it lives.

    The distinction that matters when reading a report is whether the checks ran in the
    environment setup_env.sh built or in one the reviewer already had; the path adds
    nothing to that and takes their username with it.
    """
    conda = _conda_prefix()
    if conda:
        try:
            Path(conda).relative_to(ARTIFACT / ".conda")
            kind = "the artifact's conda environment"
        except ValueError:
            kind = "a conda environment outside the artifact"
        version = _conda_version()
        if version:
            kind += f" (conda {version})"
    elif os.environ.get("VIRTUAL_ENV") == sys.prefix or (Path(sys.prefix) / "pyvenv.cfg").is_file():
        kind = "a virtualenv"
    else:
        kind = "the system python"

    stale = os.environ.get("CONDA_PREFIX")
    if stale and stale != conda:
        # Worth flagging -- it means an activation did not take -- but the path it names
        # is as identifying as any other, so say only that the two disagree.
        kind += "  (note: CONDA_PREFIX names a different environment than this one)"
    return kind


def _gpu_lines() -> list[tuple[str, str]]:
    try:
        import torch
    except Exception as exc:                                     # noqa: BLE001
        return [("gpu", f"torch not importable ({type(exc).__name__})")]

    rows = [("torch", f"{torch.__version__} (cuda {torch.version.cuda or 'none'})")]
    if not torch.cuda.is_available():
        rows.append(("gpu", "no CUDA device visible"))
        return rows

    name = torch.cuda.get_device_name(0)
    cap = "".join(map(str, torch.cuda.get_device_capability(0)))
    gib = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    count = torch.cuda.device_count()
    suffix = f", {count} visible" if count > 1 else ""
    rows.append(("gpu", f"{name} (sm_{cap}, {gib:.0f} GiB{suffix})"))

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    if driver.returncode == 0 and driver.stdout.strip():
        rows.append(("driver", driver.stdout.strip().splitlines()[0]))
    return rows


def block() -> str:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    # No hostname.  It identifies the reviewer's institution and answers nothing.
    rows: list[tuple[str, str]] = [
        ("date", now),
        ("os", neutral_platform()),
    ]
    tree = os.environ.get("AE_TREE")
    if tree:
        rows.append(("tree", tree))
    rows += [
        ("python", platform.python_version()),
        ("env", _environment_kind()),
        ("tritonparse", _version("tritonparse")),
        ("triton", _version("triton")),
    ]
    rows += _gpu_lines()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        rows.append(("CUDA_VISIBLE_DEVICES", visible))

    width = max(len(k) for k, _ in rows)
    return "\n".join(f"  {k:<{width}} : {v}" for k, v in rows)


def redact(text: str) -> str:
    """Take the reviewer's identity out of text we ask them to send us."""
    text = text.replace(str(ARTIFACT), "<artifact>")
    try:
        return text.replace(str(Path.home()), "~")
    except (RuntimeError, OSError):
        return text


def write_full(directory: Path) -> Path | None:
    """The exhaustive version list, which is too long to print but worth keeping.

    A reviewer's report is far easier to act on with this attached than without it, and
    it costs one file per run.
    """
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "environment.txt"
    parts = [block(), ""]

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=False)
    parts += ["---- pip freeze ----", freeze.stdout.strip() or "(pip freeze failed)", ""]

    prefix = _conda_prefix()
    if prefix:
        conda = subprocess.run(["conda", "list", "-p", prefix],
                               capture_output=True, text=True, check=False)
        if conda.returncode == 0:
            parts += ["---- conda list ----", conda.stdout.strip(), ""]

    try:
        # pip records editable installs as file:///home/..., and conda list heads its
        # output with the environment path.
        out.write_text(redact("\n".join(parts)) + "\n")
    except OSError:
        return None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", metavar="DIR",
                    help="also write DIR/environment.txt with the full version list")
    args = ap.parse_args()

    text = block()
    print(text)
    if args.full:
        written = write_full(Path(args.full))
        if written:
            width = max(len(line.split(" : ")[0].strip()) for line in text.splitlines())
            print(f"  {'full versions':<{width}} : {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
