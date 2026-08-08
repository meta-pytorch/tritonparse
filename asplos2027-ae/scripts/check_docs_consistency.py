#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Keep the numbers in the artifact's documents honest.

Every measured value in this artifact appears in several places: README.md, STATUS.md,
the LaTeX appendix, and the header comments of the scripts.  The appendix in particular
*has* to repeat them, because it is submitted with the paper and must stand on its own.
That duplication is where mistakes come from -- during development this artifact
accumulated a wrong repository size (off by 65x), four stale timings, a claim that both
scripts write CSVs when only one does, and two cross-references to a section that had
been renumbered.  Each was a single edit that did not propagate.

This script is the guard.  It checks three things:

  1. every path the documents mention actually exists;
  2. no retired wording has crept back in (values and file names we have already
     corrected once);
  3. the numbers quoted in prose still agree with data/reference/, which is measured
     rather than hand-maintained -- so re-measuring tells you which prose to update
     instead of leaving the two to drift apart.

Run it after editing any document:

    python scripts/check_docs_consistency.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Prose files whose numbers must agree with each other and with the measured data.
DOCS = ["README.md", "STATUS.md", "run_all.sh", "setup_env.sh",
        "00_kick_the_tires.sh", "01_gpu_core.sh", "02_review_ui.sh",
        "claims/C8_bugs/README.md", "claims/C8_bugs/run.sh",
        "claims/C8_bugs/toy_kernel.py", "claims/C9_rope/README.md",
        "data/reference/PROVENANCE.txt", "traces/reference/PROVENANCE.txt",
        "requirements-pinned.txt", "environment.yml"]

# The Artifact Appendix lives in the paper repository, not here: it is compiled into
# the paper and uses \autoref against the paper's labels, so it cannot sit in this
# tree.  It still quotes every number below, though, so point --extra-doc at it when
# editing either side:
#     python scripts/check_docs_consistency.py --extra-doc ../paper/sections/ae_appendix.tex

# Wording we have already corrected once and do not want back.  Keyed by what went wrong,
# so a failure explains itself instead of just naming a regex.
RETIRED = {
    r"\b300\s*MB\b(?!.*npm)": "repository size: it is ~5 MB; 300 MB was the working tree with node_modules",
    r"\bTier[- ][012]\b": "tier vocabulary was dropped when the artifact went to two scripts",
    r"DOWNLOAD\.md": "the trace ships in git; there is no download stub",
    r"requirements-tier0-cpu": "the CPU-only requirements file was removed",
    r"fetch_reference_trace|make_release_asset": "the Release-asset machinery was removed",
    r"REQUIREMENTS\.md": "REQUIREMENTS.md was merged into README.md",
    r"02_paper_scale": "that script never existed",
    r"No H100/MI300X": "reads as 'those GPUs are excluded'; say 'not required' instead",
    r"\b(151|156|196)\s*\\?,?\s*s(econds)?\b": "superseded timing; use the measured ranges",
    r"\b(24|25)\s*[-–]{1,2}\s*(45|90)\b": "timings are single budgets now, not ranges: "
                                            "the basic test is ~2 min",
    r"\b1?7[08]\s*[-–]{1,2}\s*450\b": "timings are single budgets now, not ranges: "
                                        "the main check is ~8 min",
    r"\b8\s*[-–]{1,2}\s*20\s*s\b": "the interface build is quoted as ~1 min now",
    r"numpy==2\.5": "numpy 2.5 requires Python >= 3.12; the pin is 2.4.1",
    r"Python\s*(?:≥|>=)\s*3\.10": "numpy 2.4.1 sets the floor at 3.11, not 3.10",
    r"(?:paper|§)\s*4\.5\b": "the paper dropped its autotune subsection, so the "
                              "reproduction experiment renumbered from 4.5 to 4.4",
    r"autotune reuse": "the paper no longer has an autotune experiment",
    r"python3 -m venv \.venv &&": "the quick start installs from environment.yml; the "
                                  "venv path is the documented alternative further down",
    r"4\.7\s*\\?,?\s*GB": "measured 5.1 GB once the conda environment supplies Node",
    # There are no release tags any more: reviewers track the branch, and the fixed
    # point is the archival deposit.  A document naming a tag is pointing at something
    # that does not exist.
    r"asplos2027-ae-v[0-9]+\b": "release tags were removed; reviewers track the branch "
                                "and the fixed point is the DOI",
    # A DOI placeholder that escapes into a submitted appendix claims an archive that
    # does not exist.  The reserved DOI is 10.5281/zenodo.21853863.
    r"zenodo\.X+": "DOI placeholder; the reserved DOI is 10.5281/zenodo.21853863",
    r"10\.5281/zenodo\.(?!21853863)\d+": "wrong Zenodo DOI; ours is 10.5281/zenodo.21853863",
    r"(?i)\b(TODO|FIXME|TBD)\b": "unresolved placeholder in a reviewer-facing document",
}

# Wall-time budgets, in minutes.  Prose quotes a single upper bound per step rather than
# a measured range, and the total has to be the sum -- an artifact that advertises "about
# 15 minutes" and then spends twenty is worse than one that never promised.
BUDGETS = {"setup": 5, "basic": 2, "main": 8}   # the required path
OPTIONAL_BUDGETS = {"ui": 1}                    # the interface build, which may be skipped
TOTAL_BUDGET = 15                               # must equal sum(BUDGETS)

# Figures we quote as *observed* rather than as budgets -- an H100 with warm caches.
# Listed so the check below does not mistake them for an undeclared budget.
MEASURED_MINUTES = {3}


# Wording that is correct while the artifact is under evaluation and wrong once it is
# published.  The Artifact Appendix is the only one of these documents that gets typeset
# into the proceedings: README.md and STATUS.md are working documents we can rewrite any
# day, but the appendix is read years later by someone who was never part of the
# evaluation.  Two kinds of text do not survive that transition -- text addressed to
# reviewers, and text about the evaluation as an ongoing process -- and the branch
# instruction and the deferred deposit will be outright false by then.
#
# Not checked by default, because all of it is right for now.  Run
#
#     python scripts/check_docs_consistency.py --camera-ready --extra-doc ...
#
# once before the camera-ready deadline and resolve every hit.  Relying on remembering
# is what left a zenodo.XXXXXXX placeholder and a superseded tag in this file before.
CAMERA_READY = {
    r"(?i)\bwe claim\b": "badge claims belong to the submission form; by publication the "
                          "badges are awarded and printed on the paper",
    r"(?i)\breviewer(s|'s)?\b": "addressed to the evaluation committee; rewrite for a "
                               "reader who was not part of it",
    r"(?i)(during|once|for the) evaluation": "describes the evaluation as ongoing",
    r"(?i)\bsend us\b": "addressed to the evaluation committee",
    r"(?i)(--branch asplos2027-ae|track a \\emph\{branch\}|rather than a tag)":
        "the branch is the evaluation-time pointer; the published appendix should cite "
        "the archival DOI",
    r"(?i)kick-the-tires": "an evaluation-phase term",
    r"(?i)deposit is made once": "by publication the deposit exists; cite the DOI",
}


def read(path: Path) -> str:
    text = path.read_text(errors="replace")
    if path.suffix == ".tex":
        # LaTeX escapes underscores and needs explicit break points inside long
        # identifiers, since \texttt does not hyphenate and a 30-character path
        # otherwise runs out of the column.  Undo both so paths match the filesystem.
        text = text.replace("\\_", "_").replace("\\allowbreak", "")
    return text


def load_reference() -> dict:
    out = {}
    ref = ROOT / "data" / "reference"
    for f in sorted(ref.glob("*.csv")):
        for row in csv.DictReader(f.open()):
            if row.get("metric"):
                out[row["metric"]] = row["value"]
    return out


def check_paths(problems: list) -> None:
    """Every artifact-relative path mentioned in prose should resolve."""
    pattern = re.compile(
        r"(?<![\w/.])((?:scripts|claims|data|traces)/[A-Za-z0-9_][A-Za-z0-9_./-]*)"
    )
    for name in DOCS:
        doc = Path(name) if Path(name).is_absolute() else ROOT / name
        if not doc.is_file():
            problems.append(f"{name}: listed in DOCS but missing")
            continue
        for m in pattern.finditer(read(doc)):
            rel = m.group(1).rstrip(".,;:)")
            if rel.endswith("/"):
                rel = rel[:-1]
            if "*" in rel or rel.endswith(("/**",)):
                continue
            if not (ROOT / rel).exists():
                problems.append(f"{name}: references {rel}, which does not exist")


# Longest run of characters with no line-break opportunity that a \texttt token may
# contain.  \texttt does not hyphenate, so a long identifier is one unbreakable box; when
# it lands at the end of a line TeX will accept an overfull line rather than the gaping
# one that moving it would leave, and the token pokes out of the column.  Two rounds of
# this were found by eye, both times after I had written a scanner that missed them --
# once because the regex could not match the \texttt{-{}-flag} form at all, once because
# it did not know that TeX breaks after an explicit hyphen or slash.  Hence this.
MAX_UNBREAKABLE = 13

# What TeX will break after inside \texttt, plus the break we insert by hand.
_BREAKS = ("\\allowbreak", " ", "-", "/")
_TEXTTT = re.compile(r"\\texttt\{((?:[^{}]|\{\})*)\}")


def check_column_width(problems: list) -> None:
    """Flag \texttt tokens with no way to break, in .tex documents only."""
    for name in DOCS:
        doc = Path(name) if Path(name).is_absolute() else ROOT / name
        if doc.suffix != ".tex" or not doc.is_file():
            continue
        # Not read(): that strips \allowbreak, which is exactly what we are measuring.
        text = doc.read_text(errors="replace")
        for num, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            for m in _TEXTTT.finditer(line):
                token = m.group(1).replace("-{}-", "--").replace("\\_", "_")
                for sep in _BREAKS:
                    token = token.replace(sep, "\x00")
                longest = max((len(run) for run in token.split("\x00")), default=0)
                if longest > MAX_UNBREAKABLE:
                    problems.append(
                        f"{name}:{num}: {m.group(1)!r} has {longest} characters with no "
                        f"break point; add \\allowbreak at its component boundaries"
                    )


def check_camera_ready(problems: list) -> None:
    """Only the typeset document matters here; the others are never published."""
    for name in DOCS:
        doc = Path(name) if Path(name).is_absolute() else ROOT / name
        if doc.suffix != ".tex" or not doc.is_file():
            continue
        text = read(doc)
        lines = text.splitlines()
        for pattern, why in CAMERA_READY.items():
            for m in re.finditer(pattern, text):
                line = text[:m.start()].count("\n") + 1
                # A LaTeX comment is not published, and this file's own header explains
                # the very wording we are looking for.
                if lines[line - 1].lstrip().startswith("%"):
                    continue
                problems.append(f"{name}:{line}: {m.group(0)!r} — {why}")


def check_retired(problems: list) -> None:
    for name in DOCS:
        doc = Path(name) if Path(name).is_absolute() else ROOT / name
        if not doc.is_file():
            continue
        text = read(doc)
        for pattern, why in RETIRED.items():
            for m in re.finditer(pattern, text):
                line = text[: m.start()].count("\n") + 1
                problems.append(f"{name}:{line}: {m.group(0)!r} — {why}")


def check_timings(problems: list) -> None:
    """Minute figures in prose must be budgets we actually declared, and add up."""
    if TOTAL_BUDGET != sum(BUDGETS.values()):
        problems.append(
            f"the required steps budget {sum(BUDGETS.values())} min but the entry point "
            f"advertises {TOTAL_BUDGET}; one of the two is wrong"
        )
    allowed = (set(BUDGETS.values()) | set(OPTIONAL_BUDGETS.values())
               | {TOTAL_BUDGET} | MEASURED_MINUTES)
    quoted = re.compile(r"(\d{1,3})\s*(?:\\,)?\s*min(?:ute)?s?\b")
    for name in DOCS:
        doc = Path(name) if Path(name).is_absolute() else ROOT / name
        if not doc.is_file():
            continue
        text = read(doc)
        for m in quoted.finditer(text):
            value = int(m.group(1))
            if value not in allowed:
                line = text[: m.start()].count("\n") + 1
                problems.append(
                    f"{name}:{line}: {m.group(0)!r} is not a declared budget "
                    f"({sorted(allowed)} min)"
                )


def check_against_measurements(problems: list, ref: dict) -> None:
    """Numbers quoted in prose must bracket what data/reference/ actually recorded."""
    checks = []

    ratio = ref.get("reconstruct_ratio_mean")
    if ratio:
        checks.append(("C6 reconstruct ratio", float(ratio),
                       re.compile(r"(\d{2})\s*(?:-|--|–)\s*(\d{2})\s*(?:×|\$?\\times)")))
    div = ref.get("divergence_mean")
    if div:
        checks.append(("C9 divergence", float(div),
                       re.compile(r"(1\d|2\d)\s*(?:-|--|–)\s*(\d\d)\s*\\?,?\s*%")))

    for label, measured, pattern in checks:
        seen = False
        for name in DOCS:
            doc = Path(name) if Path(name).is_absolute() else ROOT / name
            if not doc.is_file():
                continue
            text = read(doc)
            for m in pattern.finditer(text):
                lo, hi = int(m.group(1)), int(m.group(2))
                seen = True
                if not (lo <= measured <= hi):
                    line = text[: m.start()].count("\n") + 1
                    problems.append(
                        f"{name}:{line}: {label} quoted as {lo}-{hi} but "
                        f"data/reference says {measured}"
                    )
        if not seen:
            problems.append(f"{label}: measured {measured} but no document quotes a range")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra-doc", action="append", default=[],
                    help="also check a document outside this tree, such as the "
                         "Artifact Appendix in the paper repository")
    ap.add_argument("--camera-ready", action="store_true",
                    help="additionally flag, in .tex documents only, wording that is "
                         "right during evaluation and wrong in the proceedings")
    args = ap.parse_args()
    for extra in args.extra_doc:
        path = Path(extra).resolve()
        if not path.is_file():
            print(f"ERROR: --extra-doc {extra} does not exist.", file=sys.stderr)
            return 1
        DOCS.append(str(path))

    problems: list = []
    ref = load_reference()
    if not ref:
        print("ERROR: data/reference/ has no readable CSV rows.", file=sys.stderr)
        return 1

    check_paths(problems)
    check_retired(problems)
    check_column_width(problems)
    if args.camera_ready:
        check_camera_ready(problems)
    check_timings(problems)
    check_against_measurements(problems, ref)

    print("=" * 78)
    print(" Document consistency")
    print("=" * 78)
    print(f"  documents checked   : {len(DOCS)}")
    print(f"  reference metrics   : {len(ref)}")
    if problems:
        print(f"  problems            : {len(problems)}")
        print()
        for p in problems:
            print(f"    {p}")
        print("=" * 78)
        return 1
    print("  problems            : none")
    print()
    print("  Paths resolve, no retired wording, the quoted minute budgets are ones we")
    print("  declared and they add up, and")
    print("  the numbers in prose bracket what data/reference/ recorded.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
