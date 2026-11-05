#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Stable identity for the Triton compiler that produced a trace."""

from __future__ import annotations

import copy
import json
import logging
from functools import lru_cache
from importlib.resources import files as package_files
from types import ModuleType
from typing import Any


log = logging.getLogger(__name__)

_FB_VERSION_INFO_FIELDS = (
    "internal_version_label",
    "oss_version",
    "upstream_hash",
)

# Reading a packaged text resource can fail through the import system, the
# filesystem, or decoding; every failure mode just drops that field.
_RESOURCE_ERRORS = (
    AttributeError,
    ImportError,
    NotImplementedError,
    OSError,
    TypeError,
    ValueError,  # includes UnicodeDecodeError
)


def _read_packaged_text(triton_module: ModuleType, filename: str) -> str | None:
    try:
        return (
            package_files(triton_module).joinpath(filename).read_text(encoding="utf-8")
        )
    except _RESOURCE_ERRORS:
        return None


def _build_compiler_identity(triton_module: ModuleType) -> dict[str, Any]:
    raw_version = getattr(triton_module, "__version__", None)
    triton_version = (
        raw_version.strip()
        if isinstance(raw_version, str) and raw_version.strip()
        else "unknown"
    )
    identity: dict[str, Any] = {"triton_version": triton_version}

    src_hash = _read_packaged_text(triton_module, "src_hash.txt")
    if src_hash and src_hash.strip():
        identity["src_hash"] = src_hash.strip()

    version_info_text = _read_packaged_text(triton_module, "fb_version_info.json")
    if version_info_text:
        try:
            raw_version_info = json.loads(version_info_text)
        except (TypeError, ValueError):
            log.debug("Ignoring malformed triton/fb_version_info.json")
        else:
            if isinstance(raw_version_info, dict):
                version_info = {
                    field: raw_version_info[field]
                    for field in _FB_VERSION_INFO_FIELDS
                    if isinstance(raw_version_info.get(field), str)
                }
                if version_info:
                    identity["fb_version_info"] = version_info

    return identity


@lru_cache(maxsize=1)
def _get_compiler_identity() -> dict[str, Any]:
    # The fallback lives inside the cached function so a failed probe is
    # also cached: identity collection pays its cost at most once per
    # process and can never fail compilation tracing.
    try:
        import triton

        return _build_compiler_identity(triton)
    except Exception:
        log.warning("Failed to collect Triton compiler identity", exc_info=True)
        return {"triton_version": "unknown"}


def get_compiler_identity() -> dict[str, Any]:
    """Return the process-cached compiler identity. Never raises."""
    return copy.deepcopy(_get_compiler_identity())
