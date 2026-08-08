#!/usr/bin/env bash
# TritonParse — ASPLOS'27 Artifact Evaluation
# Main check.  Needs one NVIDIA GPU, SM80+ with >=16 GB.
#
# What this validates (see README.md for the full claim map):
#   C1  Record     : a live capture carries TTIR / TTGIR / LLIR / PTX + launch metadata
#   C6  Reconstruct: two-stage log-size reduction, compared against paper Fig. 3
#   C4  Reproduce  : a generated reproducer *executes standalone* and agrees with the
#                    original kernel
#   C8  Bug repro  : the three failure classes of paper 4.4 (assert / IMA / hang) at
#                    miniature scale                       [provided by claims/C8_bugs]
#   C9  RoPE       : the BF16 rounding mechanism behind paper 5.1, run as the literal
#                    PTX of Listing 2                      [provided by claims/C9_rope]
#   V&V            : the project's own GPU test suite
#
# Run 00_kick_the_tires.sh first.
#
# Usage:
#   ./01_gpu_core.sh
#   SKIP_UNITTESTS=1 ./01_gpu_core.sh
#   RESULTS_DIR=/somewhere ./01_gpu_core.sh
#
# Expected wall time: up to about 8 minutes.  It takes ~3 min on an H100 with a warm
# cache; a slower machine roughly doubles it.  60 s of the total is the two deliberate
# 30 s hang timeouts in C8 rather than work.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture this run to a file even when it is launched on its own, and identify the tree
# it ran from.  Both have to happen before anything prints.
# shellcheck source=scripts/ae_logging.sh
source "$HERE/scripts/ae_logging.sh"
ae_begin_log "gpu_core" "$@"
# shellcheck source=scripts/ae_provenance.sh
source "$HERE/scripts/ae_provenance.sh"
AE_TREE="$(ae_provenance "$HERE")"; export AE_TREE

# Use the artifact's environment whether or not the caller activated it.
# shellcheck source=scripts/activate_env.sh
source "$HERE/scripts/activate_env.sh"
REPO_ROOT="${TRITONPARSE_REPO:-$(cd "$HERE/.." && pwd)}"
WORK="${WORK_DIR:-$(mktemp -d -t tritonparse-ae1-XXXXXX)}"
RESULTS="${RESULTS_DIR:-$HERE/results}"
SKIP_UNITTESTS="${SKIP_UNITTESTS:-0}"
mkdir -p "$RESULTS"

PASS=0; FAIL=0; SKIP=0; NOTE=0
declare -a ROWS
_row() { ROWS+=("$1|$2|$3"); }
ok()   { PASS=$((PASS+1)); _row "$1" "PASS" "$2"; printf '  \033[32m✓ PASS\033[0m  %s — %s\n' "$1" "$2"; }
bad()  { FAIL=$((FAIL+1)); _row "$1" "FAIL" "$2"; printf '  \033[31m✗ FAIL\033[0m  %s — %s\n' "$1" "$2"; }
skip() { SKIP=$((SKIP+1)); _row "$1" "SKIP" "$2"; printf '  \033[33m- SKIP\033[0m  %s — %s\n' "$1" "$2"; }
# NOTE is for something that ran, found something, and is not a failure.  Reporting that
# as SKIP would be untrue -- SKIP means the check did not run -- and reporting it as FAIL
# would mean the artifact is broken when it is not.
note() { NOTE=$((NOTE+1)); _row "$1" "NOTE" "$2"; printf '  \033[36m⚑ NOTE\033[0m  %s — %s\n' "$1" "$2"; }
step() { printf '\n\033[1m[%s]\033[0m %s\n' "$1" "$2"; }

trap 'printf "\n(working dir kept for inspection: %s)\n" "$WORK"' EXIT

cat <<BANNER
================================================================================
 TritonParse — ASPLOS'27 Artifact Evaluation :: main check
================================================================================
 repo root : $REPO_ROOT
 work dir  : $WORK
 results   : $RESULTS
BANNER

# The environment this ran in, in the console and therefore in the log.  Most reports of
# "this gave me a different number" are answered by these lines alone.
if command -v python >/dev/null 2>&1; then
    python "$HERE/scripts/ae_env.py" ${AE_LOG_DIR:+--full "$AE_LOG_DIR"} 2>/dev/null || true
fi

# ---------------------------------------------------------------- 0. environment
step "0/7" "Environment"

if ! command -v tritonparseoss >/dev/null 2>&1; then
    bad "env.cli" "'tritonparseoss' not on PATH — see README.md, Installation"
    exit 1
fi
ok "env.cli" "$(tritonparseoss --version 2>&1 | tail -1)"

GPUINFO="$(python - <<'PY' 2>/dev/null
import torch
if torch.cuda.is_available():
    cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
    print(f"{torch.cuda.get_device_name(0)} (sm_{cap.replace('.','')}), "
          f"torch {torch.__version__}")
PY
)"
if [[ -z "$GPUINFO" ]]; then
    bad "env.gpu" "no CUDA device visible (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-unset}')"
    echo
    echo "  This check requires a GPU.  If nvidia-smi lists one, check CUDA_VISIBLE_DEVICES;"
    echo "  if the driver is too old for the installed torch build, see README.md,"
    echo "  Choosing the right CUDA wheel."
    exit 1
fi
ok "env.gpu" "$GPUINFO"

# ------------------------------------------------------------------- 1. C1 Record
step "1/7" "C1  Record — live capture carries every IR stage (paper 3.1)"

if python "$HERE/scripts/check_ir_capture.py" --csv "$RESULTS/c1_ir_capture.csv" \
        >"$WORK/c1.log" 2>&1; then
    sed -n '/^  backend/,/^  verdict/p' "$WORK/c1.log" | sed 's/^/    /'
    ok "C1.capture" "all IR stages present in every compilation"
else
    sed -n '/\[C1\]/,$p' "$WORK/c1.log" | tail -25 | sed 's/^/    /'
    bad "C1.capture" "missing IR stages — see $WORK/c1.log"
fi

# -------------------------------------------------------------- 2. C6 Reconstruct
step "2/7" "C6  Reconstruct — log-size reduction vs paper Fig. 3"

if python "$HERE/scripts/measure_storage.py" --csv "$RESULTS/c6_storage.csv" \
        >"$WORK/c6.log" 2>&1; then
    sed -n '/^  workload/,/^  verdict/p' "$WORK/c6.log" | sed 's/^/    /'
    # Keep the +/- : reporting variation alongside the expected value is the point of
    # running the measurement more than once.
    RATIO="$(grep -oE 'RECONSTRUCT RATIO   : [0-9.]+x \+/- [0-9.]+' "$WORK/c6.log" \
             | sed -E 's/.*: //')"
    ok "C6.reduction" "reconstruct ratio ${RATIO:-?} over 3 trials (paper: 57.4x on H100/37 ops)"
    echo "    note: row (2) runtime-gzip is informational only — see $WORK/c6.log"
else
    tail -25 "$WORK/c6.log" | sed 's/^/    /'
    bad "C6.reduction" "measurement failed or ratio below 10x — see $WORK/c6.log"
fi

# ----------------------------------------------------------------- 3. C4 Reproduce
step "3/7" "C4  Reproduce — generated reproducer executes standalone (paper 3.4)"

REPRO_TRACE_DIR="$WORK/c4_trace"
if python "$HERE/scripts/make_reference_trace.py" --out-dir "$REPRO_TRACE_DIR" \
        >"$WORK/c4_record.log" 2>&1; then
    ok "C4.record" "fresh trace recorded on this GPU"
else
    bad "C4.record" "recording failed — see $WORK/c4_record.log"
fi

TRACE="$(find "$REPRO_TRACE_DIR/parsed" -name '*_mapped.ndjson.gz' 2>/dev/null | head -1)"
if [[ -n "$TRACE" ]]; then
    # --kernel-import copy + --embed-context are REQUIRED: the default mode emits
    # `from <original module> import <kernel>`, which cannot resolve anywhere else.
    if tritonparseoss reproduce "$TRACE" \
            --kernel fused_op_kernel --launch-id 0 \
            --kernel-import copy --embed-context \
            --out-dir "$WORK/c4_repro" >"$WORK/c4_gen.log" 2>&1; then
        REPRO="$(find "$WORK/c4_repro" -name 'repro_*.py' | head -1)"
        ok "C4.generate" "$(basename "${REPRO:-none}") — $(wc -l <"$REPRO" 2>/dev/null | tr -d ' ') lines"
    else
        bad "C4.generate" "reproducer generation failed — see $WORK/c4_gen.log"
        REPRO=""
    fi

    if [[ -n "${REPRO:-}" ]]; then
        # Execute it from an unrelated directory: a reproducer that silently depends on
        # the recording tree would pass if we ran it in place.
        RUNDIR="$WORK/c4_run"; mkdir -p "$RUNDIR"; cp "$REPRO" "$RUNDIR/"
        if (cd "$RUNDIR" && timeout 900 python "$(basename "$REPRO")") \
                >"$WORK/c4_exec.log" 2>&1; then
            ok "C4.execute" "reproducer ran standalone from a clean directory (exit 0)"
            tail -3 "$WORK/c4_exec.log" | sed 's/^/    /'
        else
            bad "C4.execute" "reproducer failed to run — see $WORK/c4_exec.log"
            tail -12 "$WORK/c4_exec.log" | sed 's/^/    /'
        fi
    fi
else
    bad "C4.generate" "no trace to reproduce from"
fi

# ------------------------------------------------------------------ 4. C8 Bug repro
step "4/7" "C8  Bug reproduction — assert / IMA / hang (paper 4.4, miniature scale)"

if [[ -x "$HERE/claims/C8_bugs/run.sh" ]]; then
    if RESULTS_DIR="$RESULTS" "$HERE/claims/C8_bugs/run.sh" >"$WORK/c8.log" 2>&1; then
        ok "C8.bugs" "$(grep -oE '[0-9]+/[0-9]+ bug classes reproduced' "$WORK/c8.log" | tail -1)"
    else
        tail -20 "$WORK/c8.log" | sed 's/^/    /'
        bad "C8.bugs" "see $WORK/c8.log"
    fi
else
    skip "C8.bugs" "claims/C8_bugs/run.sh not present in this snapshot"
fi

# ------------------------------------------------------------------------ 5. C9 RoPE
step "5/7" "C9  RoPE — BF16 rounding mechanism behind paper 5.1"

if [[ -f "$HERE/claims/C9_rope/bf16_rounding_study.py" ]]; then
    if python "$HERE/claims/C9_rope/bf16_rounding_study.py" \
            --csv "$RESULTS/c9_rope.csv" >"$WORK/c9.log" 2>&1; then
        sed -n '/^   seed/,/^  verdict/p' "$WORK/c9.log" | sed 's/^/    /'
        DIV="$(grep -oE 'DIVERGENCE          : [0-9.]+% \+/- [0-9.]+' "$WORK/c9.log" \
               | sed -E 's/.*: //')"
        ok "C9.rounding" "divergence ${DIV:-?} over 5 seeds (paper 5.1: 25.84%); the two implementations agree exactly"
    else
        tail -20 "$WORK/c9.log" | sed 's/^/    /'
        bad "C9.rounding" "see $WORK/c9.log"
    fi
else
    skip "C9.rounding" "claims/C9_rope not present in this snapshot"
fi

# ---------------------------------------------------------------------- 6. V&V
step "6/7" "V&V  — project GPU test suite (tests/gpu)"

if [[ "$SKIP_UNITTESTS" == "1" ]]; then
    skip "vv.gpu" "skipped via SKIP_UNITTESTS=1"
elif [[ ! -d "$REPO_ROOT/tests/gpu" ]]; then
    skip "vv.gpu" "tests/gpu not found under $REPO_ROOT (set TRITONPARSE_REPO)"
else
    if (cd "$REPO_ROOT" && python -m unittest discover -s tests/gpu -t .) \
            >"$WORK/unittest_gpu.log" 2>&1; then
        ok "vv.gpu" "$(grep -E '^Ran [0-9]+ test' "$WORK/unittest_gpu.log" | tail -1)"
    else
        bad "vv.gpu" "$(grep -E '^(Ran [0-9]+ test|FAILED)' "$WORK/unittest_gpu.log" | tr '\n' ' ')"
    fi
fi

# --------------------------------------------------------- 7. compare vs reference
step "7/7" "Comparison against the authors' recorded results"

if [[ -d "$HERE/data/reference" ]]; then
    python "$HERE/scripts/compare_to_reference.py" --results "$RESULTS" \
        --reference "$HERE/data/reference" >"$WORK/compare.log" 2>&1
    CMP_RC=$?
    # Print the block to the end.  An earlier version stopped at the counts line, which
    # cut off the paragraph explaining that a difference is not a failure -- so the one
    # thing a reviewer needed to read was the one thing only the log file had.
    sed -n '/^ CLAIM/,$p' "$WORK/compare.log" | sed 's/^/  /'
    CMP_COUNTS="$(grep -oE '[0-9]+ within tolerance, [0-9]+ differing(, [0-9]+ missing)?' "$WORK/compare.log")"
    case "$CMP_RC" in
        0) ok   "ref.compare" "${CMP_COUNTS:-all metrics within tolerance}" ;;
        2) note "ref.compare" "${CMP_COUNTS:-} — differs from our H100, not a failure" ;;
        1) if [[ "$SKIP" -gt 0 ]]; then
               # Some check above did not run, so the reference has metrics this run had
               # no chance to produce.  Expected, and already reported as a SKIP.
               note "ref.compare" "${CMP_COUNTS:-} — missing values belong to skipped checks"
           else
               bad "ref.compare" "${CMP_COUNTS:-} — a check ran but wrote no result; see $WORK/compare.log"
           fi ;;
        *) bad "ref.compare" "comparison itself failed (exit $CMP_RC) — see $WORK/compare.log" ;;
    esac
else
    skip "ref.compare" "data/reference not present in this snapshot"
fi

# ---------------------------------------------------------------------- summary
printf '\n%s\n' "================================================================================"
printf ' SUMMARY   %d passed, %d failed, %d skipped, %d noted\n' "$PASS" "$FAIL" "$SKIP" "$NOTE"
printf '%s\n' "================================================================================"
printf ' %-16s %-6s %s\n' "CHECK" "RESULT" "DETAIL"
for r in "${ROWS[@]}"; do
    IFS='|' read -r c s d <<<"$r"
    printf ' %-16s %-6s %s\n' "$c" "$s" "${d:0:96}"
done
printf '%s\n' "================================================================================"
echo " CSV results written to: $RESULTS"
echo
echo " Reminder: absolute numbers differ from the paper by design — the artifact pins a"
echo " newer released stack than Table 1.  See README.md, Version delta versus the paper."

[[ "$FAIL" -gt 0 ]] && { echo " RESULT: FAIL — see logs in $WORK"; exit 1; }
if [[ "$NOTE" -gt 0 ]]; then
    echo " RESULT: PASS — main check complete, with $NOTE note(s) above."
    echo " A NOTE is something that ran and found a difference from our machine. It is not"
    echo " a failure: no pass criterion was missed. Mention it if you report anything."
else
    echo " RESULT: PASS — main check complete."
fi
exit 0
