#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Resolve a diff input to a local file, plus a shareable URL when one exists.

OSS accepts local paths only. In fbcode the work is delegated to
``tritonparse.diff.fb.input_resolver``, which additionally understands remote
trace references, can fetch them, and can upload a local trace so the website
can load it; that module is not synced to GitHub.
"""

from __future__ import annotations

from dataclasses import dataclass

from tritonparse.shared_vars import is_fbcode


@dataclass(frozen=True)
class ResolvedInput:
    """A diff input that has been made readable on the local filesystem."""

    # Path to read events from.
    local_path: str
    # Original user-supplied string, for display.
    display: str
    # Browser-fetchable URL, when one exists for this input. None for purely
    # local inputs, which cannot be linked.
    json_url: str | None = None


def is_remote(source: str) -> bool:
    """Whether this input has to be fetched before it can be read.

    Lets callers reject an unsupported combination before paying for the
    fetch. Always False in OSS, which accepts local paths only.
    """
    if not is_fbcode():
        return False
    from tritonparse.diff.fb.input_resolver import is_remote as fb_is_remote

    return fb_is_remote(source)


def resolve_input(source: str) -> ResolvedInput:
    """Resolve one diff input to something readable on this filesystem."""
    if is_fbcode():
        from tritonparse.diff.fb.input_resolver import resolve_input as fb_resolve

        return fb_resolve(source)
    return ResolvedInput(local_path=source, display=source)


def share_input(resolved: ResolvedInput) -> ResolvedInput:
    """Publish a local input so the website can fetch it, if possible.

    A no-op in OSS, which has nowhere to publish to: the input is returned
    unchanged and simply carries no link.
    """
    if resolved.json_url is not None or not is_fbcode():
        return resolved
    from tritonparse.diff.fb.input_resolver import share_input as fb_share

    return fb_share(resolved)
