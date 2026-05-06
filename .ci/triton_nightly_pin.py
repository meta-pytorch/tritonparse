#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Resolve the exact upstream Triton nightly wheel version that should be
pinned, working around the upstream issue where pip's PEP-440
lexicographic sort on `+git<hash>` local versions selects the wrong
wheel from the Triton-Nightly Azure DevOps feed.

Strategy:
  1. Query the GitHub Actions API for the latest successful run of
     triton-lang/triton's wheels.yml workflow (one HTTP call).
  2. Take head_sha[:8] as the upstream "short SHA" used in the
     PEP-440 local-version segment (`+git<short_sha>`).
  3. Fetch the Azure DevOps simple index and find the wheel whose
     filename contains that short SHA AND matches the caller's
     Python tag and platform substring (one HTTP call). This step
     also verifies the wheel has actually been published to the
     feed (not just that the build succeeded).

Output contract:
  - On success: print the exact PEP-440 version string (e.g.
    "3.7.0+git9282d719") to stdout AND nothing else, exit 0.
  - On any failure: print diagnostic to stderr, exit non-zero with
    no stdout. Caller should fall back to default pip behavior.

The "how do we determine the target version" piece is intentionally
isolated as the RESOLVERS list, so additional strategies (job-log
parsing, summary scraping, ...) can be added without touching the
caller-facing CLI contract.

This script is stdlib-only by design — it must be safe to invoke with
any Python >=3.7 on PATH (the only language requirement is
`from __future__ import annotations`, which makes the modern PEP-585
collection generics in annotations purely textual at runtime),
including the system/container interpreter, before the target conda
environment is even created.

Background: see https://github.com/triton-lang/triton (open issue on
the wheels.yml local-version sort regression).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Callable, Optional

INDEX_URL = (
    "https://aiinfra.pkgs.visualstudio.com/PublicPackages/"
    "_packaging/Triton-Nightly/pypi/simple/triton/"
)
GITHUB_API = "https://api.github.com"
UPSTREAM_REPO = "triton-lang/triton"
WORKFLOW_FILE = "wheels.yml"
HTTP_TIMEOUT = 25
USER_AGENT = "tritonparse-nightly-pin/1.0"


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def http_get_json(url: str, headers: Optional[dict] = None) -> dict:
    h = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.loads(resp.read())


def http_get_text(url: str, headers: Optional[dict] = None) -> str:
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return resp.read().decode("utf-8", "replace")


VersionResolver = Callable[[argparse.Namespace], Optional[str]]


def resolve_via_workflow_head_sha(args: argparse.Namespace) -> Optional[str]:
    """Latest successful run of upstream wheels.yml -> head_sha[:8]."""
    url = (
        f"{GITHUB_API}/repos/{UPSTREAM_REPO}/actions/workflows/"
        f"{WORKFLOW_FILE}/runs?status=success&branch=main&per_page=1"
    )
    headers: dict[str, str] = {}
    token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    log(f"[resolver:workflow] GET {url}")
    try:
        data = http_get_json(url, headers=headers)
    except urllib.error.HTTPError as e:
        rl = (e.headers or {}).get("X-RateLimit-Remaining")
        log(f"[resolver:workflow] HTTP {e.code} {e.reason} (ratelimit-remaining={rl})")
        return None
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        log(f"[resolver:workflow] ERROR: {type(e).__name__}: {e}")
        return None

    if not isinstance(data, dict):
        log(f"[resolver:workflow] unexpected response shape: {type(data).__name__}")
        return None
    runs = data.get("workflow_runs", [])
    if not runs:
        log("[resolver:workflow] no successful runs returned")
        return None
    run = runs[0]
    head_sha = run.get("head_sha")
    log(
        f"[resolver:workflow] run id={run.get('id')} "
        f"head_sha={head_sha} created={run.get('created_at')}"
    )
    if not head_sha or len(head_sha) < 8:
        log("[resolver:workflow] head_sha missing or too short")
        return None
    return head_sha[:8].lower()


def resolve_via_logs(args: argparse.Namespace) -> Optional[str]:
    """Placeholder for future fallback: parse cibuildwheel output from
    job logs ZIP, extract `triton-...whl` filenames directly. Wire in
    if upstream changes the version-string scheme and head_sha
    construction stops working."""
    log("[resolver:logs] not implemented")
    return None


def resolve_via_summary(args: argparse.Namespace) -> Optional[str]:
    """Placeholder for future fallback: scrape the rendered job summary
    markdown. Brittle (no public REST API for $GITHUB_STEP_SUMMARY);
    only as a last resort."""
    log("[resolver:summary] not implemented")
    return None


# Order matters: try in sequence until one returns a value.
RESOLVERS: list[tuple[str, VersionResolver]] = [
    ("workflow_head_sha", resolve_via_workflow_head_sha),
    ("logs", resolve_via_logs),
    ("summary", resolve_via_summary),
]


def resolve_short_sha(args: argparse.Namespace) -> Optional[tuple[str, str]]:
    """Try each resolver in order. Returns (resolver_name, short_sha)."""
    for name, fn in RESOLVERS:
        result = fn(args)
        if result:
            return (name, result)
    return None


_WHEEL_RE_TMPL = (
    r"triton-(\d+\.\d+\.\d+\+git{sha}[0-9a-f]*)-{py_tag}-{py_tag}-"
    r'[^"#]*{platform}[^"#]*\.whl'
)


def find_wheel_version(
    index_html: str, short_sha: str, py_tag: str, platform: str
) -> Optional[str]:
    """Search the simple-index HTML for a wheel matching short_sha,
    py_tag, and platform substring. Returns the exact PEP-440 version
    string from the wheel filename, or None if no match."""
    pattern = _WHEEL_RE_TMPL.format(
        sha=re.escape(short_sha.lower()),
        py_tag=re.escape(py_tag),
        platform=re.escape(platform),
    )
    m = re.search(pattern, index_html)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Resolve the latest upstream Triton nightly wheel pin. "
            "Prints the exact version (e.g. '3.7.0+git9282d719') to "
            "stdout on success; exits non-zero on any failure."
        )
    )
    ap.add_argument(
        "--pytag",
        default=f"cp{sys.version_info.major}{sys.version_info.minor}",
        help="Python tag to match (default: this interpreter's tag)",
    )
    ap.add_argument(
        "--platform",
        default="x86_64",
        help="Platform substring to match in wheel filename (default: x86_64)",
    )
    ap.add_argument(
        "--github-token",
        default=None,
        help="GitHub token (or set GITHUB_TOKEN env var) to raise rate "
        "limit from 60/hr to 5000/hr. Optional.",
    )
    args = ap.parse_args()

    log(f"[pin] target: pytag={args.pytag} platform~='{args.platform}'")

    resolved = resolve_short_sha(args)
    if not resolved:
        log("[pin] FAIL: no resolver produced a target SHA")
        return 1
    resolver_name, short_sha = resolved
    log(f"[pin] resolver={resolver_name} short_sha={short_sha}")

    log(f"[pin] GET {INDEX_URL}")
    try:
        index_html = http_get_text(INDEX_URL)
    except (urllib.error.URLError, OSError) as e:
        log(f"[pin] FAIL: index fetch failed: {type(e).__name__}: {e}")
        return 2

    version = find_wheel_version(index_html, short_sha, args.pytag, args.platform)
    if not version:
        log(
            f"[pin] FAIL: no wheel matching short_sha={short_sha} "
            f"pytag={args.pytag} platform~='{args.platform}' on the feed "
            "(workflow may have succeeded but upload not yet complete)"
        )
        return 3

    log(f"[pin] OK: version={version}")
    # The ONLY thing on stdout, no trailing whitespace beyond a single newline.
    print(version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
