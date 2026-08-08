#!/usr/bin/env bash
# TritonParse — ASPLOS'27 Artifact Evaluation
# The "basic test" for the kick-the-tires phase.
#
# What this validates (see README.md for the full claim map):
#   C2  Reconstruct : raw structured log -> compressed, IR-mapped NDJSON archive
#   C2' Query       : structured kernel/launch inventory from an archive
#   C3  Review      : cross-compilation IR diff (the mechanism behind paper 5.1 / 5.2)
#   C4  Reproduce   : generation of a self-contained single-file reproducer
#   V&V             : the project's own CPU unit-test suite
#
# Every check here operates on the trace shipped under traces/reference/, so this script
# exercises the offline half of the workflow and isolates installation problems before
# anything touches the GPU.  The reproducer it generates is *executed* by 01_gpu_core.sh.
#
# Usage:
#   ./00_kick_the_tires.sh                 # full run
#   SKIP_UNITTESTS=1 ./00_kick_the_tires.sh   # skip the unit-test stage (faster)
#
# Expected wall time: up to about 2 minutes.  It takes ~25 s on an H100 with a warm
# cache; a slower machine roughly doubles it.

set -uo pipefail

# ---------------------------------------------------------------- configuration
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture this run to a file even when it is launched on its own, and identify the tree
# it ran from.  Both have to happen before anything prints.
# shellcheck source=scripts/ae_logging.sh
source "$HERE/scripts/ae_logging.sh"
ae_begin_log "kick_the_tires" "$@"
# shellcheck source=scripts/ae_provenance.sh
source "$HERE/scripts/ae_provenance.sh"
AE_TREE="$(ae_provenance "$HERE")"; export AE_TREE

# Use the artifact's environment whether or not the caller activated it.
# shellcheck source=scripts/activate_env.sh
source "$HERE/scripts/activate_env.sh"
# Repo root: this script normally lives in <repo>/asplos2027-ae/
REPO_ROOT="${TRITONPARSE_REPO:-$(cd "$HERE/.." && pwd)}"
WORK="${WORK_DIR:-$(mktemp -d -t tritonparse-ae-XXXXXX)}"
SKIP_UNITTESTS="${SKIP_UNITTESTS:-0}"

# Reference trace shipped with this artifact.  Recorded by scripts/make_reference_trace.py
# on the stack pinned in requirements-pinned.txt; see traces/reference/PROVENANCE.txt.
# It deliberately contains 5 compilations (3x matmul_kernel, 2x fused_op_kernel) with
# recorded launches, because `diff --events` needs >=2 comparable compilations and
# `info` / `reproduce` need launch records.
#
# The repository's own tests/example_output/ fixtures also work and can be selected via
# TRITONPARSE_AE_TRACE_DIR — but note that tests/example_output/parsed_output/ has ZERO
# launch events and therefore cannot drive `info` or `reproduce`; only
# parsed_output_complex/ can.
TRACE_DIR="${TRITONPARSE_AE_TRACE_DIR:-$HERE/traces/reference}"
RAW_LOG_DIR="$TRACE_DIR/logs"
TRACE_SRC="$(find "$TRACE_DIR/parsed" -name '*_mapped.ndjson.gz' 2>/dev/null | head -1)"

# ---------------------------------------------------------------- reporting
PASS=0; FAIL=0; SKIP=0
declare -a ROWS

_row() { ROWS+=("$1|$2|$3"); }
ok()   { PASS=$((PASS+1)); _row "$1" "PASS" "$2"; printf '  \033[32m✓ PASS\033[0m  %s — %s\n' "$1" "$2"; }
bad()  { FAIL=$((FAIL+1)); _row "$1" "FAIL" "$2"; printf '  \033[31m✗ FAIL\033[0m  %s — %s\n' "$1" "$2"; }
skip() { SKIP=$((SKIP+1)); _row "$1" "SKIP" "$2"; printf '  \033[33m- SKIP\033[0m  %s — %s\n' "$1" "$2"; }
step() { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }

trap 'printf "\n(working dir kept for inspection: %s)\n" "$WORK"' EXIT

cat <<BANNER
================================================================================
 TritonParse — ASPLOS'27 Artifact Evaluation :: basic test
================================================================================
 repo root : $REPO_ROOT
 work dir  : $WORK
BANNER

# The environment this ran in, in the console and therefore in the log.  Most reports of
# "this gave me a different number" are answered by these lines alone.
if command -v python >/dev/null 2>&1; then
    python "$HERE/scripts/ae_env.py" ${AE_LOG_DIR:+--full "$AE_LOG_DIR"} 2>/dev/null || true
fi

# ---------------------------------------------------------------- 0. environment
step "0/6" "Environment"

if ! command -v tritonparseoss >/dev/null 2>&1; then
    bad "env.cli" "'tritonparseoss' not on PATH — run ./setup_env.sh, or ./run_all.sh to do everything"
    echo; echo "Cannot continue without the CLI."; exit 1
fi
TP_VERSION="$(tritonparseoss --version 2>&1 | tail -1)"
ok "env.cli" "$TP_VERSION"

python - <<'PY' && ok "env.deps" "python / triton / torch importable" || bad "env.deps" "missing python dependency"
import sys
import triton, torch
print(f"  python {sys.version.split()[0]}  triton {triton.__version__}  torch {torch.__version__}")
PY

if [[ ! -d "$RAW_LOG_DIR" || -z "$TRACE_SRC" || ! -f "$TRACE_SRC" ]]; then
    bad "env.fixtures" "reference trace not found under $TRACE_DIR"
    cat <<MISSING

  This should not happen: the trace ships with the artifact under traces/reference/.
  If you removed it, either restore it from the archive you downloaded, or record an
  equivalent one on a GPU:

      python asplos2027-ae/scripts/make_reference_trace.py --out-dir asplos2027-ae/traces/reference

  You can also point at any existing TritonParse trace directory containing logs/ and
  parsed/:

      TRITONPARSE_AE_TRACE_DIR=/path/to/dir
MISSING
    exit 1
fi
if [[ -f "$TRACE_DIR/PROVENANCE.txt" ]]; then
    REC="$(python -c "import json;d=json.load(open('$TRACE_DIR/PROVENANCE.txt'));print(f\"torch {d['torch']} / triton {d['triton']} / {d['gpu']}\")" 2>/dev/null)"
    ok "env.fixtures" "reference trace present (recorded on ${REC:-unknown stack})"
else
    ok "env.fixtures" "reference trace present"
fi

# Work on copies: several subcommands write their output next to the input file,
# and the repository working tree must stay clean.
mkdir -p "$WORK/in"
cp -r "$RAW_LOG_DIR" "$WORK/in/logs"
cp "$TRACE_SRC" "$WORK/in/trace.ndjson.gz"
TRACE="$WORK/in/trace.ndjson.gz"

# ---------------------------------------------------- 1. C2 Record -> Reconstruct
step "1/6" "C2  Reconstruct — parse a raw structured log into a compact archive"

RAW_BYTES=$(du -sb "$WORK/in/logs" | cut -f1)
if tritonparseoss parse "$WORK/in/logs" --out "$WORK/parsed" >"$WORK/parse.log" 2>&1; then
    OUT_BYTES=$(du -sb "$WORK/parsed" | cut -f1)
    NFILES=$(find "$WORK/parsed" -name '*.ndjson.gz' | wc -l | tr -d ' ')
    if [[ "$NFILES" -ge 1 && -f "$WORK/parsed/log_file_list.json" ]]; then
        RATIO=$(python -c "print(f'{$RAW_BYTES/max($OUT_BYTES,1):.2f}')")
        ok "C2.parse" "$NFILES archive(s) + log_file_list.json; ${RAW_BYTES}B -> ${OUT_BYTES}B (${RATIO}x on this tiny fixture)"
        echo "      note: the paper's 57.4x (Fig.3) is measured on 37 TritonBench operators;"
        echo "            this fixture is a single toy kernel.  The ratio at scale is measured by 01_gpu_core.sh."
    else
        bad "C2.parse" "expected *.ndjson.gz + log_file_list.json, got $NFILES archive(s)"
    fi
else
    bad "C2.parse" "parse failed — see $WORK/parse.log"
fi

# ---------------------------------------------------------------- 2. C2' inventory
step "2/6" "C2' Query — structured kernel / launch inventory"

if tritonparseoss info "$TRACE" >"$WORK/info.log" 2>&1; then
    KLINES=$(grep -cE '^ +[A-Za-z_][A-Za-z0-9_]* +[0-9]+ launches' "$WORK/info.log" || true)
    if [[ "${KLINES:-0}" -ge 1 ]]; then
        ok "C2.info" "$KLINES kernel(s) with launch counts:"
        grep -E '^ +[A-Za-z_][A-Za-z0-9_]* +[0-9]+ launches' "$WORK/info.log" | sed 's/^/        /'
    else
        bad "C2.info" "ran, but listed no kernels — see $WORK/info.log"
    fi
else
    bad "C2.info" "info failed — see $WORK/info.log"
fi

# ---------------------------------------------------------- 3. C3 cross-trace diff
step "3/6" "C3  Review — cross-compilation IR diff (paper 5.1 / 5.2 mechanism)"

if tritonparseoss diff "$TRACE" --list >"$WORK/diff_list.log" 2>&1; then
    NCOMP=$(grep -oE 'Total: [0-9]+ compilation' "$WORK/diff_list.log" | grep -oE '[0-9]+' | head -1)
    ok "C3.list" "${NCOMP:-?} compilation(s) enumerated"
    grep -oE '\[ *[0-9]+\] .*' "$WORK/diff_list.log" | sed 's/^/        /'
else
    bad "C3.list" "diff --list failed — see $WORK/diff_list.log"
fi

if tritonparseoss diff "$TRACE" --events 0,1 --output "$WORK/diff01.ndjson.gz" >"$WORK/diff01.log" 2>&1; then
    # A meaningful diff reports per-IR-stage statistics for the whole lowering pipeline.
    STAGES=$(grep -cE '^ +(TTIR|TTGIR|LLIR|PTX|AMDGCN):' "$WORK/diff01.log" || true)
    if [[ "${STAGES:-0}" -ge 3 ]]; then
        ok "C3.diff" "IR-level diff across $STAGES lowering stages"
        grep -E '^ +(TTIR|TTGIR|LLIR|PTX|AMDGCN):' "$WORK/diff01.log" | sed 's/^/        /'
        grep -E '^ +Line [0-9]+:' "$WORK/diff01.log" | head -3 | sed 's/^/        /'
    else
        bad "C3.diff" "diff ran but reported < 3 IR stages — see $WORK/diff01.log"
    fi
else
    bad "C3.diff" "diff --events failed — see $WORK/diff01.log"
fi

# ------------------------------------------------------- 4. C4 reproducer generate
step "4/6" "C4  Reproduce — generate a self-contained single-file reproducer"

# --kernel-import copy   : embed the kernel source instead of importing the original file
# --embed-context        : inline the launch context instead of emitting a side-car JSON
# Both are REQUIRED for a reproducer that runs on a machine that never saw the workload.
KERNEL="$(grep -oE '^ +[A-Za-z_][A-Za-z0-9_]*' "$WORK/info.log" 2>/dev/null | head -1 | tr -d ' ')"
KERNEL="${KERNEL:-fused_op_kernel}"

if tritonparseoss reproduce "$TRACE" \
        --kernel "$KERNEL" --launch-id 0 \
        --kernel-import copy --embed-context \
        --out-dir "$WORK/repro" >"$WORK/repro.log" 2>&1; then
    REPRO="$(find "$WORK/repro" -name 'repro_*.py' | head -1)"
    if [[ -n "$REPRO" ]]; then
        LOC=$(wc -l <"$REPRO" | tr -d ' ')
        NJSON=$(find "$WORK/repro" -name '*_context_*.json' | wc -l | tr -d ' ')
        ok "C4.generate" "$(basename "$REPRO") — $LOC lines, kernel '$KERNEL', $NJSON side-car file(s)"

        # The claim under test is portability: the reproducer must NOT import the module
        # the workload was originally recorded from.  Report every other third-party
        # import so the reviewer can see the true dependency surface.
        # Take the workload's module name from the trace itself, not from the
        # reproduce log: `kernel_src_path` there is a path relative to a root that does
        # not always exist, and comes back empty for traces recorded outside a project
        # tree.  Silently degrading to "no module to check" would turn this into a
        # check that always passes, so an undeterminable name is a hard failure.
        ORIG_MOD="$(python - "$TRACE" <<'PY'
import gzip, json, sys, pathlib
for line in gzip.open(sys.argv[1], "rt"):
    ev = json.loads(line)
    if ev.get("event_type") != "compilation":
        continue
    # payload.file_path maps IR stages to cache files; the Python module the kernel
    # was defined in is under payload.python_source.file_path.
    fp = ((ev.get("payload") or {}).get("python_source") or {}).get("file_path") or ""
    if fp:
        print(pathlib.Path(fp).stem)
        break
PY
)"
        if [[ -z "$ORIG_MOD" ]]; then
            bad "C4.standalone" "could not determine the workload module from the trace; check skipped rather than silently passing"
        elif ORIG_MOD="$ORIG_MOD" python - "$REPRO" <<'PY' >"$WORK/selfcheck.log" 2>&1; then
import ast, os, sys

STDLIB = {
    "os","sys","json","gzip","io","math","time","logging","pathlib","typing","functools",
    "hashlib","importlib","argparse","dataclasses","contextlib","tempfile","subprocess",
    "collections","itertools","re","struct","warnings","base64","random","textwrap","copy",
}
NUMERIC = {"torch", "triton", "numpy"}
orig = os.environ["ORIG_MOD"]

tree = ast.parse(open(sys.argv[1]).read())
mods = set()
for n in ast.walk(tree):
    if isinstance(n, ast.ImportFrom) and n.module:
        mods.add(n.module)
    elif isinstance(n, ast.Import):
        mods.update(a.name for a in n.names)

third = sorted({m for m in mods if m.split(".")[0] not in STDLIB})
print("third-party imports: " + (", ".join(third) or "<none>"))
if any(m.split(".")[0] == orig for m in mods):
    print(f"VIOLATION: still imports the original workload module '{orig}'")
    sys.exit(1)
extra = sorted({m for m in third if m.split(".")[0] not in NUMERIC})
if extra:
    print("NOTE: also requires " + ", ".join(extra) +
          " (acceptable for AE: these are installed by requirements-pinned.txt, "
          "but the reproducer is not torch+triton-only)")
PY
            ok "C4.standalone" "does not import original workload module '$ORIG_MOD'"
            sed 's/^/        /' "$WORK/selfcheck.log"
        else
            bad "C4.standalone" "$(tail -1 "$WORK/selfcheck.log")"
        fi
        echo "      run it on a GPU machine with:  python $REPRO"
        echo "      (01_gpu_core.sh does this automatically)"
    else
        bad "C4.generate" "no repro_*.py produced — see $WORK/repro.log"
    fi
else
    bad "C4.generate" "reproduce failed — see $WORK/repro.log"
fi

# ---------------------------------------------------------------- 5. V&V
step "5/6" "V&V  — project CPU unit-test suite (tests/cpu)"

if [[ "$SKIP_UNITTESTS" == "1" ]]; then
    skip "vv.cpu" "skipped via SKIP_UNITTESTS=1"
elif [[ ! -d "$REPO_ROOT/tests/cpu" ]]; then
    skip "vv.cpu" "tests/cpu not found under $REPO_ROOT (set TRITONPARSE_REPO)"
else
    if (cd "$REPO_ROOT" && python -m unittest discover -s tests/cpu -t .) >"$WORK/unittest.log" 2>&1; then
        SUMMARY=$(grep -E '^Ran [0-9]+ test' "$WORK/unittest.log" | tail -1)
        ok "vv.cpu" "${SUMMARY:-all CPU tests passed}"
    else
        SUMMARY=$(grep -E '^(Ran [0-9]+ test|FAILED)' "$WORK/unittest.log" | tr '\n' ' ')
        bad "vv.cpu" "${SUMMARY:-see $WORK/unittest.log}"
    fi
fi

# ---------------------------------------------------------------- 6. GPU hint
step "6/6" "Next tier"

if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "  A CUDA device is visible — you can now run:  ./01_gpu_core.sh"
else
    echo "  No CUDA device visible from this process."
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "  nvidia-smi does see a GPU, so check CUDA_VISIBLE_DEVICES (currently: '${CUDA_VISIBLE_DEVICES-unset}')."
    fi
    echo "  The checks above are offline and their results stand regardless."
fi

# ---------------------------------------------------------------- summary
printf '\n%s\n' "================================================================================"
printf ' SUMMARY   %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
printf '%s\n' "================================================================================"
printf ' %-16s %-6s %s\n' "CHECK" "RESULT" "DETAIL"
for r in "${ROWS[@]}"; do
    IFS='|' read -r c s d <<<"$r"
    printf ' %-16s %-6s %s\n' "$c" "$s" "${d:0:96}"
done
printf '%s\n' "================================================================================"

if [[ "$FAIL" -gt 0 ]]; then
    echo " RESULT: FAIL — see logs in $WORK"
    exit 1
fi
echo " RESULT: PASS — basic test complete.  Next: ./01_gpu_core.sh"
exit 0
