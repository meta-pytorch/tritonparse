#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Verify that TritonParse captures every stage of the lowering pipeline (paper 3.1).

The paper's Record claim is that a single capture yields the whole chain

    Python source -> TTIR -> TTGIR -> LLIR -> PTX (NVIDIA) / AMDGCN (AMD)

together with launch metadata.  This script records a live workload on whatever GPU is
present and asserts, per compilation event, that all expected stages are there --
rather than trusting a trace that shipped with the artifact.

    python scripts/check_ir_capture.py --csv results/c1_ir_capture.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# NVIDIA lowers to PTX, AMD to AMDGCN; everything before the backend is shared.
COMMON_STAGES = ("ttir", "ttgir", "llir")
BACKEND_STAGES = {"cuda": "ptx", "hip": "amdgcn"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("ERROR: this check needs a GPU.", file=sys.stderr)
        return 1

    backend = "hip" if torch.version.hip else "cuda"
    expected = COMMON_STAGES + (BACKEND_STAGES[backend],)

    work = Path(tempfile.mkdtemp(prefix="tritonparse-c1-"))
    try:
        import tritonparse.structured_logging as sl
        from tritonparse.parse.utils import unified_parse

        logs, parsed = work / "logs", work / "parsed"
        logs.mkdir()
        parsed.mkdir()

        sl.init(str(logs), enable_trace_launch=True)
        import ae_kernels

        ae_kernels.workload_small()
        unified_parse(str(logs), out=str(parsed), overwrite=True)

        archives = sorted(parsed.glob("*.ndjson.gz"))
        if not archives:
            print("ERROR: no archive produced", file=sys.stderr)
            return 1

        rows, n_comp, n_launch, failures = [], 0, 0, []
        for arc in archives:
            for line in gzip.open(arc, "rt"):
                ev = json.loads(line)
                etype = ev.get("event_type")
                if etype == "launch":
                    n_launch += 1
                    continue
                if etype != "compilation":
                    continue
                n_comp += 1
                payload = ev.get("payload", {})
                files = payload.get("file_content", {}) or {}
                # keys look like "<kernel>.ttir", "<kernel>.ptx", ...
                got = {k.rsplit(".", 1)[-1].lower() for k in files}
                kernel = payload.get("metadata", {}).get("name") or "<unknown>"
                missing = [s for s in expected if s not in got]
                sizes = {
                    s: len(str(files.get(next((k for k in files
                                               if k.lower().endswith("." + s)), ""), "")).splitlines())
                    for s in expected
                }
                if missing:
                    failures.append((kernel, missing))
                rows.append((kernel, sizes, missing))

        print()
        print("=" * 78)
        print(" [C1] Multi-IR capture  (paper 3.1)")
        print("=" * 78)
        print(f"  backend             : {backend}   expected stages: {', '.join(expected)}")
        print(f"  compilations        : {n_comp}")
        print(f"  launch events       : {n_launch}")
        print()
        header = "  " + "kernel".ljust(24) + "".join(s.upper().rjust(10) for s in expected)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for kernel, sizes, missing in rows:
            cells = "".join(
                (str(sizes[s]) if s not in missing else "MISSING").rjust(10)
                for s in expected
            )
            print("  " + kernel[:24].ljust(24) + cells)
        print("  (cell = line count of that IR stage)")
        print()

        verdict = "PASS" if (n_comp > 0 and not failures and n_launch > 0) else "FAIL"
        if failures:
            for kernel, missing in failures:
                print(f"  MISSING stages for {kernel}: {', '.join(missing)}")
        print(f"  verdict             : {verdict}  "
              f"(criterion: every compilation carries all {len(expected)} stages, "
              f"and launches were recorded)")
        print("=" * 78)

        if args.csv:
            path = Path(args.csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["claim", "metric", "value", "unit", "paper_value", "verdict"])
                w.writerow(["C1", "backend", backend, "", "cuda+hip", ""])
                w.writerow(["C1", "compilations", n_comp, "count", "", ""])
                w.writerow(["C1", "launch_events", n_launch, "count", "", ""])
                w.writerow(["C1", "stages_expected", "|".join(expected), "", "", ""])
                w.writerow(["C1", "compilations_missing_stages", len(failures), "count",
                            0, verdict])
        return 0 if verdict == "PASS" else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
