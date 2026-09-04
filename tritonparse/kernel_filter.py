# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import fnmatch
from collections.abc import Sequence


KernelPatterns = tuple[str, ...]


def parse_kernel_patterns(spec: str | None) -> KernelPatterns | None:
    """Parse a comma-separated kernel allowlist into fnmatch patterns."""
    if not spec:
        return None

    patterns = tuple(pattern.strip() for pattern in spec.split(",") if pattern.strip())
    return patterns or None


def first_matching_kernel_pattern(
    kernel_name: str | None,
    patterns: Sequence[str] | None,
) -> str | None:
    """Return the first allowlist pattern matching ``kernel_name``."""
    if kernel_name is None or patterns is None:
        return None

    return next(
        (pattern for pattern in patterns if fnmatch.fnmatch(kernel_name, pattern)),
        None,
    )


def matches_kernel_name(
    kernel_name: str | None,
    patterns: Sequence[str] | None,
) -> bool:
    """Return whether a kernel name passes the allowlist."""
    if patterns is None:
        return True
    return first_matching_kernel_pattern(kernel_name, patterns) is not None
