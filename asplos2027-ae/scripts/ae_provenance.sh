#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# One line identifying the tree being evaluated.
#
# During artifact evaluation the reviewers track a *branch*, not a tag, so that fixes we
# make in response to their reports reach them with a `git pull`.  That convenience costs
# something: "the main check failed" no longer identifies a tree.  So every run stamps the
# branch and commit into its own output, and a reviewer who pastes a log tells us exactly
# what they ran without being asked.
#
# Source it, then call ae_provenance.

ae_provenance() {
    local root="${1:-.}" branch sha dirty

    if ! command -v git >/dev/null 2>&1 \
       || ! git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
        # A zip or a tarball.  Not the supported path -- setuptools-scm needs the git
        # metadata -- but say so plainly instead of printing nothing.
        printf 'not a git checkout (see README.md, Getting the artifact)'
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
