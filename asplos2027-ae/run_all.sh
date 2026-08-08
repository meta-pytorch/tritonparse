#!/usr/bin/env bash
# TritonParse — ASPLOS'27 Artifact Evaluation
# The single entry point: build the environment, then run every check in order.
#
#     ./asplos2027-ae/run_all.sh
#
# That is the whole evaluation.  It installs conda if the host has none, builds the
# environment from environment.yml, and runs the basic test, the main check and the
# Review-interface build, stopping at the first failure.  Budget about 15 minutes for the
# required steps on a machine that has to download everything, plus a minute for the
# optional interface build; a second run with warm caches is far quicker.
#
# The individual scripts still work on their own if you would rather step through them --
# see README.md -- but nothing requires you to.
#
# WHERE THE OUTPUT GOES.  Each run gets its own directory under logs/, named for the time
# it started, so re-running never destroys the evidence from the previous attempt:
#
#     asplos2027-ae/logs/2026-08-06_181500/
#         summary.txt          the table printed at the end
#         01-setup_env.log     full output of each step, in order
#         02-kick_the_tires.log
#         03-gpu_core.log
#         04-review_ui.log
#         results/             the CSVs the main check writes
#     asplos2027-ae/logs/latest -> the most recent of those
#
# Console output is the summary plus the tail of anything that fails; the logs hold
# everything.  If a step fails, its log is the first thing to read and the path is
# printed.
#
# Options:
#   --skip-setup     use the environment already active instead of building one
#   --skip-ui        skip the optional Review-interface build (it is the only step
#                    that downloads npm packages)
#   --keep-going     run every step even after one fails, instead of stopping
#   --quiet          do not stream step output to the console
#   -h, --help       this text
#
# Everything the individual scripts honour still applies -- SKIP_UNITTESTS=1 to skip the
# unit-test stages, and the AE_* variables setup_env.sh documents.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_SETUP=0
SKIP_UI=0
KEEP_GOING=0
QUIET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-setup) SKIP_SETUP=1 ;;
        --skip-ui)    SKIP_UI=1 ;;
        --keep-going) KEEP_GOING=1 ;;
        --quiet)      QUIET=1 ;;
        # Print the header block, however long it happens to be, rather than a line range
        # that goes stale the first time someone edits a sentence above.
        -h|--help)    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# shellcheck source=scripts/ae_provenance.sh
source "$HERE/scripts/ae_provenance.sh"
# shellcheck source=scripts/ae_logging.sh
source "$HERE/scripts/ae_logging.sh"
TREE="$(ae_provenance "$HERE")"

RUN_DIR="$HERE/logs/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$RUN_DIR/results" || { echo "cannot create $RUN_DIR" >&2; exit 1; }
ln -sfn "$(basename "$RUN_DIR")" "$HERE/logs/latest"

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
green() { printf '\033[32m%s\033[0m' "$1"; }
red()   { printf '\033[31m%s\033[0m' "$1"; }

cat <<BANNER
================================================================================
 TritonParse — ASPLOS'27 Artifact Evaluation
================================================================================
 tree: $TREE
 Logs and results for this run:
   $RUN_DIR
BANNER

# Each entry is  key|human label|budget|command.  The budget is the time to allow, not
# the time it takes -- on an H100 with warm caches the three checks come in at roughly
# 25 s, 3 min and 9 s.  The summary prints what each one actually took.
STEPS=()
[[ "$SKIP_SETUP" == "1" ]] || STEPS+=("setup_env|environment (conda + all dependencies)|5 min|$HERE/setup_env.sh")
STEPS+=("kick_the_tires|basic test|2 min|$HERE/00_kick_the_tires.sh")
STEPS+=("gpu_core|main check|8 min|$HERE/01_gpu_core.sh")
[[ "$SKIP_UI" == "1" ]] || STEPS+=("review_ui|Review interface build (optional)|1 min|$HERE/02_review_ui.sh")

NAMES=(); STATUSES=(); SECONDS_TAKEN=(); LOGS=()
FAILED=0
INDEX=0

for entry in "${STEPS[@]}"; do
    IFS='|' read -r key label budget cmd <<<"$entry"
    INDEX=$((INDEX + 1))
    log="$RUN_DIR/$(printf '%02d' "$INDEX")-${key}.log"

    printf '\n'
    bold "[$INDEX/${#STEPS[@]}] $label   (allow up to $budget)"
    echo "      log: $log"

    start=$(date +%s)
    if [[ "$QUIET" == "1" ]]; then
        RESULTS_DIR="$RUN_DIR/results" AE_LOG_DIR="$RUN_DIR" AE_LOG_FILE="$log" \
            bash "$cmd" >"$log" 2>&1
        rc=$?
    else
        # Stream and capture.  pipefail is off for this line on purpose: we want the
        # script's status, not tee's.
        set -o pipefail
        RESULTS_DIR="$RUN_DIR/results" AE_LOG_DIR="$RUN_DIR" AE_LOG_FILE="$log" \
            bash "$cmd" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
        set +o pipefail
    fi
    # The console wants colour and full paths; a log someone pastes into a report wants
    # neither -- see ae_redact.
    ae_strip_ansi "$log"
    ae_redact "$log"
    elapsed=$(( $(date +%s) - start ))

    NAMES+=("$label"); SECONDS_TAKEN+=("$elapsed"); LOGS+=("$log")

    if [[ $rc -eq 0 ]]; then
        STATUSES+=("PASS")
        printf '  %s  %s in %ds\n' "$(green '✓')" "$label" "$elapsed"
    else
        STATUSES+=("FAIL")
        FAILED=1
        printf '  %s  %s failed (exit %d) after %ds\n' "$(red '✗')" "$label" "$rc" "$elapsed"
        if [[ "$QUIET" == "1" ]]; then
            echo "  ---- last 20 lines of $log ----"
            tail -20 "$log" | sed 's/^/  /'
        fi
        # setup_env is a precondition, not a check: nothing after it can mean anything.
        if [[ "$key" == "setup_env" || "$KEEP_GOING" != "1" ]]; then
            echo
            echo "  Stopping here. Full output: $log"
            [[ "$key" == "setup_env" ]] && echo "  (the environment did not build, so the checks were not attempted)"
            break
        fi
    fi

    # setup_env.sh records where it put the environment; everything after it runs there.
    if [[ "$key" == "setup_env" && -f "$HERE/.env-prefix" ]]; then
        prefix="$(cat "$HERE/.env-prefix")"
        if [[ -x "$prefix/bin/python" ]]; then
            export PATH="$prefix/bin:$PATH"
            export CONDA_PREFIX="$prefix"
            echo "  using $prefix"
        fi
    fi
done

# ------------------------------------------------------------------------------ summary
TOTAL=0
for s in "${SECONDS_TAKEN[@]}"; do TOTAL=$((TOTAL + s)); done

{
    echo "================================================================================"
    echo " SUMMARY"
    echo "================================================================================"
    echo " tree: $TREE"
    echo " --------------------------------------------------------------------------------"
    printf ' %-46s %-6s %8s\n' "STEP" "RESULT" "TIME"
    for i in "${!NAMES[@]}"; do
        printf ' %-46s %-6s %7ds\n' "${NAMES[$i]}" "${STATUSES[$i]}" "${SECONDS_TAKEN[$i]}"
    done
    if [[ ${#STEPS[@]} -ne ${#NAMES[@]} ]]; then
        printf ' %-46s %-6s\n' "($(( ${#STEPS[@]} - ${#NAMES[@]} )) step(s) not attempted)" "--"
    fi
    echo " --------------------------------------------------------------------------------"
    printf ' %-46s %-6s %7ds\n' "total" "" "$TOTAL"
    echo "================================================================================"
    if [[ $FAILED -eq 0 ]]; then
        echo " RESULT: PASS — every check completed."
        echo
        echo " The main check's CSVs are in $RUN_DIR/results, already diffed against the"
        echo " authors' recorded numbers at the end of its log."
    else
        echo " RESULT: FAIL — see the logs named above."
        echo
        echo " README.md has a Troubleshooting section, and 'Read this before filing"
        echo " anything' lists the differences from the paper that are expected."
    fi
    echo " Logs: $RUN_DIR"
    echo "================================================================================"
} | tee "$RUN_DIR/summary.txt"
ae_strip_ansi "$RUN_DIR/summary.txt"
ae_redact "$RUN_DIR/summary.txt"

exit $FAILED
