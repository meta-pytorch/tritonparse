#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Every run leaves a file behind, whether or not it went through run_all.sh.
#
# run_all.sh already captures each step it launches.  A reviewer running one check on its
# own -- which the scripts support, and which is what someone does after the first
# failure -- used to get console output and nothing else, so reporting a problem meant
# copying a terminal.  ae_begin_log gives that run the same logs/<timestamp>/ treatment.
#
# Source it, then call `ae_begin_log <key>` before doing any work.

# Take the reader's identity out of a file we are asking them to send us.  Artifact
# evaluation reviewers are anonymous to authors, and one `/home/<name>/` in a pasted log
# undoes that -- so a home directory becomes ~ and the artifact's own path becomes
# <artifact>.  ae_env.py keeps these out of the block it prints; this catches everything
# the checks themselves print, which is where the paths actually come from.
ae_redact() {
    [[ -f "$1" ]] || return 0
    local here esc_here esc_home
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    # Escape the delimiter and sed's metacharacters in the replacement side.
    esc_here="$(printf '%s' "$here" | sed 's/[|&\\]/\\&/g')"
    sed -i "s|$esc_here|<artifact>|g" "$1" 2>/dev/null || true
    if [[ -n "${HOME:-}" && "$HOME" != "/" ]]; then
        esc_home="$(printf '%s' "$HOME" | sed 's/[|&\\]/\\&/g')"
        sed -i "s|$esc_home|~|g" "$1" 2>/dev/null || true
    fi
}

# Colour codes belong on a terminal, not in a file someone will paste into a report.
ae_strip_ansi() {
    [[ -f "$1" ]] || return 0
    sed -i 's/\x1b\[[0-9;]*[A-Za-z]//g' "$1" 2>/dev/null || true
}

# Re-exec the calling script with its output teed to a file, unless something upstream is
# already capturing it.  Re-executing rather than redirecting in place keeps the exit
# status exact and avoids losing the last lines to a still-draining `tee`.
ae_begin_log() {
    local key="$1"; shift

    # run_all.sh sets this before launching a step; the re-exec below sets it too, so the
    # child does not recurse.
    [[ -n "${AE_LOG_FILE:-}" ]] && return 0
    [[ "${AE_NO_LOG_FILE:-0}" == "1" ]] && return 0

    local here dir
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    dir="${AE_LOG_DIR:-$here/logs/$(date +%Y-%m-%d_%H%M%S)}"
    if ! mkdir -p "$dir" 2>/dev/null; then
        printf '  (could not create %s; console only)\n' "$dir" >&2
        return 0
    fi
    ln -sfn "$(basename "$dir")" "$here/logs/latest" 2>/dev/null || true

    export AE_LOG_DIR="$dir"
    export AE_LOG_FILE="$dir/${key}.log"
    printf '  \033[2mlog: %s\033[0m\n' "$AE_LOG_FILE"

    local rc
    set -o pipefail
    "${BASH:-bash}" "$0" "$@" 2>&1 | tee "$AE_LOG_FILE"
    rc=${PIPESTATUS[0]}
    ae_strip_ansi "$AE_LOG_FILE"
    ae_redact "$AE_LOG_FILE"
    exit "$rc"
}
