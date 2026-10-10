#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# One line identifying the source used for a run.
#
# Git checkouts report their branch and commit. Source archives do not carry Git
# metadata; their commit is identified by the archive filename and archival record.
# Both forms support the default installation from pinned PyPI packages.
#
# Source it, then call ae_provenance.

ae_provenance() {
    local root="${1:-.}" branch sha dirty

    if ! command -v git >/dev/null 2>&1 \
       || ! git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
        # Archive installation is supported; report why no Git revision is available.
        printf 'source archive (no Git metadata)'
        return 0
    fi

    branch="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    sha="$(git -C "$root" rev-parse --short=10 HEAD 2>/dev/null)"
    [[ "$branch" == "HEAD" ]] && branch="detached"

    if git -C "$root" diff --quiet HEAD -- 2>/dev/null; then
        dirty=""
    else
        dirty=", locally modified"
    fi

    printf '%s @ %s%s' "$branch" "$sha" "$dirty"
}
