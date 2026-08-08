#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Compare this machine's results against the ones recorded on ours.

The AE guidance asks authors to

    "Pre-record results from your platform and ship a script that automatically
     compares fresh results against them, sparing reviewers from parsing stdout."

That is what this is.  Note the distinction from what the individual checks already do:
each check prints its measurement next to *the paper's* number, whereas this compares
against *our* measurement of the same code on an H100 (``data/reference/``).  The two
answer different questions -- "does this substantiate the paper?" versus "did this
machine behave like the authors' machine?"

    python scripts/compare_to_reference.py
    python scripts/compare_to_reference.py --results results --reference data/reference

Exit codes: 0 clean, 2 something differs from the reference, 1 a result is missing
entirely.  Differing and missing are separated on purpose -- the first is what a
different machine does, the second means a check did not write what it should have.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# How each metric is compared.  Anything not listed is reported but not gated -- byte
# counts, platform strings and per-trial samples are expected to move around.
#
#   exact : must be identical (deterministic counts, fault classes)
#   rel:X : numeric, must be within X percent of the reference
#   info  : shown for context, never gated
#
# Gate only what a different machine should not change.  Anything downstream of a timing
# measurement is not in that set: Triton's autotuner sizes its benchmark loop from the
# measured latency of the kernel, so the number of launches -- and hence the number of
# trace records -- moves with the clock, the driver and whatever else is on the GPU.
TOLERANCE = {
    # C1 -- structural facts about a capture; these should not drift at all.
    "compilations": "exact",
    "launch_events": "exact",
    "stages_expected": "exact",
    "compilations_missing_stages": "exact",
    "backend": "exact",
    # C6 -- ratios.  Generous: a different GPU and allocator change the byte counts.
    "reconstruct_ratio_mean": "rel:40",
    "reconstruct_pct_of_raw_mean": "rel:40",
    "runtime_pct_of_raw_mean": "info",       # highly workload-dependent, see the C6 note
    "wholefile_gzip_pct_of_raw_mean": "info",
    # Not gated: this is the autotuner's benchmark-loop length, which do_bench derives
    # from measured latency, so it legitimately differs on every machine.  Printed
    # because it explains the ratio -- the reconstruct stage compresses redundancy
    # across records, so fewer records means a lower ratio.
    "trace_records": "info",
    # C8 -- fault classes must match exactly; that is the whole claim.
    "bug_classes_reproduced": "exact",
    "assert_reproduced_class": "exact",
    "ima_reproduced_class": "exact",
    "hang_reproduced_class": "exact",
    "assert_original_class": "exact",
    "ima_original_class": "exact",
    "hang_original_class": "exact",
    # C9 -- a rate with known seed-to-seed spread.
    "divergence_mean": "rel:25",
    "ptx_vs_model_mismatches": "exact",
    "samples_per_trial": "exact",
    # Hardware-dependent by design: mul.bf16 needs sm_90, so a reviewer on Ampere
    # legitimately reports false here.  Printed, never gated.
    "native_mul_bf16": "info",
}


def load(directory: Path) -> dict:
    """Read every CSV in a directory into {(claim, metric): (value, unit)}."""
    out = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if not row.get("metric"):
                    continue
                out[(row["claim"], row["metric"])] = (row["value"], row.get("unit", ""))
    return out


def compare(value: str, ref: str, rule: str):
    """Return (verdict, detail).

    verdict is PASS / DIFF / INFO.  Deliberately *not* FAIL: this script asks "did this
    machine behave like ours?", and the answer "no" is not a failure of the artifact.
    The pass criteria live in the individual checks, which print FAIL and mean it.  Two
    words for two questions, so a reviewer reading one screen cannot confuse them.
    """
    if rule == "info":
        return "INFO", ""
    if rule == "exact":
        return ("PASS", "") if value == ref else ("DIFF", f"expected {ref}")
    if rule.startswith("rel:"):
        pct = float(rule.split(":", 1)[1])
        try:
            v, r = float(value), float(ref)
        except ValueError:
            return ("PASS", "") if value == ref else ("DIFF", f"expected {ref}")
        if r == 0:
            return ("PASS", "") if v == 0 else ("DIFF", f"expected {ref}")
        delta = 100.0 * abs(v - r) / abs(r)
        if delta <= pct:
            return "PASS", f"{delta:+.0f}% of ref"
        return "DIFF", f"{delta:.0f}% off, tolerance {pct:.0f}%"
    return "INFO", ""


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--reference", default=str(here / "data" / "reference"))
    args = ap.parse_args()

    res_dir, ref_dir = Path(args.results), Path(args.reference)
    if not res_dir.is_dir():
        print(f"ERROR: no results in {res_dir}; run 01_gpu_core.sh first.", file=sys.stderr)
        return 1
    if not ref_dir.is_dir():
        print(f"ERROR: no reference data in {ref_dir}.", file=sys.stderr)
        return 1

    mine, theirs = load(res_dir), load(ref_dir)
    if not mine:
        print(f"ERROR: {res_dir} contains no readable CSV rows.", file=sys.stderr)
        return 1

    print()
    print("=" * 78)
    print(" Comparison against the authors' recorded results")
    print("=" * 78)
    prov = ref_dir / "PROVENANCE.txt"
    if prov.is_file():
        for line in prov.read_text().splitlines():
            if line.strip().startswith(("host", "gpu", "driver", "torch", "triton", "date")):
                print(f"  reference {line.strip()}")
    print()
    print(f" {'CLAIM':6} {'METRIC':34} {'THIS MACHINE':>14} {'REFERENCE':>12}  RESULT")
    print(" " + "-" * 76)

    differing = 0
    missing = 0
    gated_pass = 0
    shown = 0
    for key in sorted(theirs):
        claim, metric = key
        rule = TOLERANCE.get(metric)
        if rule is None:
            continue  # not gated and not interesting enough to print
        ref_val = theirs[key][0]
        if key not in mine:
            # A reference metric with no counterpart here.  Either the check that
            # writes it was skipped -- in which case 01_gpu_core.sh has already said so
            # -- or it ran and wrote nothing, which nothing else would notice.
            print(f" {claim:6} {metric:34} {'<missing>':>14} {ref_val:>12}  MISSING")
            missing += 1
            continue
        val = mine[key][0]
        verdict, detail = compare(val, ref_val, rule)
        shown += 1
        if verdict == "PASS":
            gated_pass += 1
        elif verdict == "DIFF":
            differing += 1
        mark = {"PASS": "PASS", "DIFF": "DIFF", "INFO": "info"}[verdict]
        print(f" {claim:6} {metric:34} {val:>14} {ref_val:>12}  {mark}"
              + (f"  ({detail})" if detail else ""))

    print(" " + "-" * 76)
    summary = f" {gated_pass} within tolerance, {differing} differing"
    if missing:
        summary += f", {missing} missing"
    print(f"{summary}, {shown} metrics compared")
    print()
    print(" Tolerances are deliberately wide for ratios and exact for structural counts.")
    print(" DIFF is not FAIL. This table answers 'did this machine behave like ours?',")
    print(" and a different machine legitimately answers no -- a different driver, a")
    print(" different allocator or a busier GPU all move these numbers. The pass criteria")
    print(" are the per-claim checks in 01_gpu_core.sh, which print FAIL and mean it. A")
    print(" DIFF is still worth mentioning if you report anything else.")
    if missing:
        print()
        print(" MISSING means the reference has a metric this run produced no value for.")
        print(" If a check above was skipped that is expected; otherwise it wrote no CSV.")
    print("=" * 78)

    if missing:
        return 1
    return 2 if differing else 0


if __name__ == "__main__":
    raise SystemExit(main())
