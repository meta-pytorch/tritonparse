#!/usr/bin/env bash
# C8 — Bug reproduction at miniature scale (paper 4.4).
#
# The paper injects three failure classes -- a device-side assertion, an illegal memory
# access, and a hang -- into five gpt-oss-20b kernels (15 mutations) and reports 100%
# reproduction.  Reviewers cannot download gpt-oss-20b, but the three mutations are
# independent of kernel size: they are the three edits in paper Listing 1.  This script
# applies exactly those edits to a ~30-line kernel and drives the identical code path:
#
#     inject -> record (workload fails) -> parse -> generate reproducer -> execute
#            -> assert the reproducer fails the SAME WAY
#
# The paper's own gpt-oss-20b runs are not shipped (gated, multi-tens-of-GB model); what
# is reproduced here is the mechanism at a size a reviewer can run.
#
# Usage:  ./run.sh            (invoked automatically by ../../01_gpu_core.sh)
#         HANG_TIMEOUT=30 ./run.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK_DIR:-$(mktemp -d -t tritonparse-c8-XXXXXX)}"
RESULTS="${RESULTS_DIR:-$HERE/../../results}"
# The paper applies a 30 s default timeout to classify a run as a hang (4.4).
HANG_TIMEOUT="${HANG_TIMEOUT:-30}"
mkdir -p "$RESULTS"

BUGS=(assert ima hang)
# What each mutation is expected to produce, as a *fault class* rather than an exact
# CUDA error code -- see the note printed at the end.
declare -A EXPECT=([assert]=assert [ima]=memfault [hang]=hang)

# Apply a unified diff.  `patch` is not installed everywhere (it is absent from, e.g.,
# python:3.13-slim), and `git apply` needs a repository, so fall back to a small pure
# Python applier that depends on nothing but the interpreter we already require.
apply_patch() {  # $1 = .patch file, $2 = target file
    local patchfile="$1" target="$2"
    if command -v patch >/dev/null 2>&1; then
        patch -s -p0 --input="$patchfile" "$target" && return 0
    fi
    python - "$patchfile" "$target" <<'PY'
import re, sys
patch_path, target_path = sys.argv[1], sys.argv[2]
lines = open(target_path).read().splitlines(keepends=True)
out, cursor = [], 0
hunk = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
it = iter(open(patch_path).read().splitlines(keepends=True))
for raw in it:
    m = hunk.match(raw)
    if not m:
        continue
    start = int(m.group(1)) - 1
    count = int(m.group(2) or 1)
    out.extend(lines[cursor:start])
    cursor = start
    consumed = 0
    while consumed < count:
        try:
            body = next(it)
        except StopIteration:
            break
        tag, text = body[:1], body[1:]
        if tag == " ":
            if lines[cursor] != text:
                sys.exit(f"context mismatch at line {cursor + 1}")
            out.append(lines[cursor]); cursor += 1; consumed += 1
        elif tag == "-":
            if lines[cursor] != text:
                sys.exit(f"removal mismatch at line {cursor + 1}")
            cursor += 1; consumed += 1
        elif tag == "+":
            out.append(text)
        else:
            break
out.extend(lines[cursor:])
open(target_path, "w").write("".join(out))
PY
}

classify() {  # $1 = exit code, $2 = log file
    local rc="$1" log="$2"
    if [[ "$rc" == "124" ]]; then echo "hang"; return; fi
    if grep -qE 'cudaErrorAssert|device-side assert' "$log"; then echo "assert"; return; fi
    if grep -qE 'cudaErrorIllegalAddress|cudaErrorInvalidAddressSpace|illegal memory access|misaligned address' "$log"; then
        echo "memfault"; return
    fi
    if [[ "$rc" == "0" ]]; then echo "none"; return; fi
    echo "other"
}
errcode() { grep -oE 'cudaError[A-Za-z]+' "$1" 2>/dev/null | head -1; }

echo "================================================================================"
echo " [C8] Bug reproduction — assert / IMA / hang  (paper 4.4, Listing 1)"
echo "================================================================================"
printf ' %-8s %-22s %-22s %s\n' "BUG" "ORIGINAL RUN" "REPRODUCER" "VERDICT"
printf ' %s\n' "--------------------------------------------------------------------------------"

OK=0
declare -a CSV
for bug in "${BUGS[@]}"; do
    D="$WORK/$bug"; mkdir -p "$D/logs" "$D/run"

    # 1. inject -- the patches are literal diffs, so a reviewer can read exactly what
    #    changed and compare it against paper Listing 1.
    cp "$HERE/toy_kernel.py" "$D/wl.py"
    if ! apply_patch "$HERE/patches/$bug.patch" "$D/wl.py" >"$D/patch.log" 2>&1; then
        printf ' %-8s %-22s %-22s %s\n' "$bug" "PATCH FAILED" "-" "FAIL"
        echo "   $(tail -1 "$D/patch.log")"
        CSV+=("C8,${bug}_reproduced_class,patch_failed,class,${EXPECT[$bug]},FAIL")
        continue
    fi

    # 2. record.  Two of the three mutations kill or wedge the process; the point of
    #    the paper's design is that the trace is written *before* the launch, so it
    #    survives anyway.
    AE_LOG_DIR="$D/logs" timeout "$HANG_TIMEOUT" python "$D/wl.py" >"$D/orig.log" 2>&1
    orig_rc=$?
    orig_class="$(classify "$orig_rc" "$D/orig.log")"
    orig_code="$(errcode "$D/orig.log")"
    trace_bytes=$(du -sb "$D/logs" 2>/dev/null | cut -f1)

    if [[ "${trace_bytes:-0}" -eq 0 ]]; then
        printf ' %-8s %-22s %-22s %s\n' "$bug" "no trace written" "-" "FAIL"
        CSV+=("C8,${bug}_reproduced_class,no_trace,class,${EXPECT[$bug]},FAIL")
        continue
    fi

    # 3. parse + 4. generate.  copy/embed are required for a portable reproducer.
    if ! tritonparseoss parse "$D/logs" --out "$D/parsed" >"$D/parse.log" 2>&1; then
        printf ' %-8s %-22s %-22s %s\n' "$bug" "$orig_class" "PARSE FAILED" "FAIL"
        CSV+=("C8,${bug}_reproduced_class,parse_failed,class,${EXPECT[$bug]},FAIL")
        continue
    fi
    TRACE="$(find "$D/parsed" -name '*_mapped.ndjson.gz' | head -1)"
    if ! tritonparseoss reproduce "$TRACE" --kernel scale_kernel --launch-id 0 \
            --kernel-import copy --embed-context --out-dir "$D/repro" \
            >"$D/gen.log" 2>&1; then
        printf ' %-8s %-22s %-22s %s\n' "$bug" "$orig_class" "GENERATE FAILED" "FAIL"
        CSV+=("C8,${bug}_reproduced_class,generate_failed,class,${EXPECT[$bug]},FAIL")
        continue
    fi
    REPRO="$(find "$D/repro" -name 'repro_*.py' | head -1)"

    # 5. execute from a clean directory, so a reproducer that secretly depends on the
    #    recording tree cannot pass.
    cp "$REPRO" "$D/run/"
    ( cd "$D/run" && timeout "$HANG_TIMEOUT" python "$(basename "$REPRO")" ) \
        >"$D/repro.log" 2>&1
    rep_rc=$?
    rep_class="$(classify "$rep_rc" "$D/repro.log")"
    rep_code="$(errcode "$D/repro.log")"

    want="${EXPECT[$bug]}"
    if [[ "$orig_class" == "$want" && "$rep_class" == "$want" ]]; then
        verdict="PASS"; OK=$((OK+1))
    else
        verdict="FAIL"
    fi
    printf ' %-8s %-22s %-22s %s\n' \
        "$bug" \
        "$orig_class${orig_code:+ ($orig_code)}" \
        "$rep_class${rep_code:+ ($rep_code)}" \
        "$verdict"
    CSV+=("C8,${bug}_original_class,${orig_class},class,${want},")
    CSV+=("C8,${bug}_reproduced_class,${rep_class},class,${want},${verdict}")
done

echo "--------------------------------------------------------------------------------"
echo " $OK/${#BUGS[@]} bug classes reproduced   (paper 4.4: 15/15 mutations on gpt-oss-20b)"
echo
echo " NOTE on matching.  Verdicts compare the *fault class*, not the exact CUDA error"
echo " code, and that is necessary rather than merely convenient: for the IMA mutation we"
echo " have observed the same binary report both cudaErrorIllegalAddress and"
echo " cudaErrorInvalidAddressSpace across runs, because the faulting address depends on"
echo " where the caching allocator happens to place the tensors.  Both are out-of-bounds"
echo " faults.  The codes actually seen are printed above, so the variance is visible"
echo " rather than hidden.  Offsets must also be large: at 1e6 elements the access still"
echo " lands inside PyTorch's allocator slab and does not fault at all, which is why the"
echo " paper uses N * 1000000."
echo
echo " The hang case is the paper's key design point: the trace is written before the"
echo " launch, so a workload that never returns still leaves a complete, parseable trace."
echo "================================================================================"

CSVFILE="$RESULTS/c8_bugs.csv"
{
    echo "claim,metric,value,unit,paper_value,verdict"
    printf '%s\n' "${CSV[@]}"
    echo "C8,bug_classes_reproduced,$OK,count,${#BUGS[@]},$([[ "$OK" -eq "${#BUGS[@]}" ]] && echo PASS || echo FAIL)"
} >"$CSVFILE"
echo " csv: $CSVFILE"
echo " work dir kept: $WORK"

[[ "$OK" -eq "${#BUGS[@]}" ]] || exit 1
exit 0
