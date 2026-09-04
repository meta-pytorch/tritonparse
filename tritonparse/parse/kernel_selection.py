# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from tritonparse._json_compat import JSONDecodeError, loads
from tritonparse.kernel_filter import KernelPatterns, matches_kernel_name
from tritonparse.tools.compression import open_compressed_file

from .sourcemap_utils import get_autotune_session_id


@dataclass
class KernelFilterIndex:
    real_compilation_hashes: set[str] = field(default_factory=set)
    fake_compilation_seed_by_hash: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    aliases_by_hash: dict[str, set[str]] = field(default_factory=dict)
    hashes_by_session: dict[str, set[str]] = field(default_factory=dict)
    sessions_by_hash: dict[str, set[str]] = field(default_factory=dict)
    aliases_by_session: dict[str, set[str]] = field(default_factory=dict)
    available_names: set[str] = field(default_factory=set)
    unknown_name_count: int = 0


@dataclass(frozen=True)
class KernelFilterStats:
    direct_matched_hash_count: int
    direct_matched_session_count: int
    selected_hash_count: int
    selected_session_count: int
    emitted_kernel_group_count: int
    available_kernel_names: frozenset[str]
    unknown_name_count: int


@dataclass(frozen=True)
class KernelSelection:
    index: KernelFilterIndex
    direct_matched_hashes: frozenset[str]
    direct_matched_sessions: frozenset[str]
    selected_hashes: frozenset[str]
    selected_sessions: frozenset[str]

    def stats(self, emitted_kernel_group_count: int) -> KernelFilterStats:
        return KernelFilterStats(
            direct_matched_hash_count=len(self.direct_matched_hashes),
            direct_matched_session_count=len(self.direct_matched_sessions),
            selected_hash_count=len(self.selected_hashes),
            selected_session_count=len(self.selected_sessions),
            emitted_kernel_group_count=emitted_kernel_group_count,
            available_kernel_names=frozenset(self.index.available_names),
            unknown_name_count=self.index.unknown_name_count,
        )


def _nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _event_names(event: dict[str, Any]) -> tuple[str, ...]:
    event_type = event.get("event_type")
    candidates: tuple[object, ...]
    if event_type == "compilation":
        candidates = (event.get("payload", {}).get("metadata", {}).get("name"),)
    elif event_type == "launch":
        candidates = (
            event.get("name"),
            event.get("compilation_metadata", {}).get("name"),
        )
    elif event_type == "autotune":
        candidates = (event.get("kernel_name"),)
    else:
        return ()

    names = []
    for candidate in candidates:
        name = _nonempty_string(candidate)
        if name is not None and name not in names:
            names.append(name)
    return tuple(names)


def _event_kernel_hash(event: dict[str, Any]) -> str | None:
    event_type = event.get("event_type")
    if event_type == "compilation":
        value = event.get("payload", {}).get("metadata", {}).get("hash")
    elif event_type == "launch":
        value = event.get("compilation_metadata", {}).get("hash")
    else:
        return None
    return _nonempty_string(value)


def _event_session_id(event: dict[str, Any]) -> str | None:
    stack = event.get("stack", [])
    session_id, _ = get_autotune_session_id(stack)
    return session_id


def _record_event(index: KernelFilterIndex, event: dict[str, Any]) -> None:
    event_type = event.get("event_type")
    if event_type not in {"compilation", "launch", "autotune"}:
        return

    names = _event_names(event)
    if names:
        index.available_names.update(names)
    else:
        index.unknown_name_count += 1

    kernel_hash = _event_kernel_hash(event)
    session_id = _event_session_id(event)

    if event_type == "compilation" and kernel_hash is not None:
        index.real_compilation_hashes.add(kernel_hash)

    if event_type == "launch" and kernel_hash is not None:
        index.fake_compilation_seed_by_hash.setdefault(kernel_hash, event)

    if kernel_hash is not None:
        index.aliases_by_hash.setdefault(kernel_hash, set()).update(names)

    if session_id is not None:
        index.aliases_by_session.setdefault(session_id, set()).update(names)
        if kernel_hash is not None:
            index.hashes_by_session.setdefault(session_id, set()).add(kernel_hash)
            index.sessions_by_hash.setdefault(kernel_hash, set()).add(session_id)


def build_kernel_filter_index(file_paths: list[str]) -> KernelFilterIndex:
    """Build the rank-local event index needed for filtered parsing."""
    index = KernelFilterIndex()
    for file_path in file_paths:
        with open_compressed_file(file_path) as source:
            for line in source:
                json_str = line.strip()
                if not json_str:
                    continue
                try:
                    event = loads(json_str)
                except JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    _record_event(index, event)
    return index


def select_kernel_groups(
    index: KernelFilterIndex,
    patterns: KernelPatterns,
) -> KernelSelection:
    """Select the complete hash/session closure matching ``patterns``."""
    direct_hashes = {
        kernel_hash
        for kernel_hash, names in index.aliases_by_hash.items()
        if any(matches_kernel_name(name, patterns) for name in names)
    }
    direct_sessions = {
        session_id
        for session_id, names in index.aliases_by_session.items()
        if any(matches_kernel_name(name, patterns) for name in names)
    }

    selected_hashes = set(direct_hashes)
    selected_sessions = set(direct_sessions)
    pending_hashes = deque(direct_hashes)
    pending_sessions = deque(direct_sessions)

    while pending_hashes or pending_sessions:
        while pending_hashes:
            kernel_hash = pending_hashes.popleft()
            for session_id in index.sessions_by_hash.get(kernel_hash, ()):
                if session_id not in selected_sessions:
                    selected_sessions.add(session_id)
                    pending_sessions.append(session_id)

        while pending_sessions:
            session_id = pending_sessions.popleft()
            for kernel_hash in index.hashes_by_session.get(session_id, ()):
                if kernel_hash not in selected_hashes:
                    selected_hashes.add(kernel_hash)
                    pending_hashes.append(kernel_hash)

    return KernelSelection(
        index=index,
        direct_matched_hashes=frozenset(direct_hashes),
        direct_matched_sessions=frozenset(direct_sessions),
        selected_hashes=frozenset(selected_hashes),
        selected_sessions=frozenset(selected_sessions),
    )


def build_kernel_selection(
    file_paths: list[str],
    patterns: KernelPatterns,
) -> KernelSelection:
    """Index a rank's files and select matching kernel/session groups."""
    return select_kernel_groups(build_kernel_filter_index(file_paths), patterns)


def format_kernel_filter_no_match(
    patterns: KernelPatterns,
    stats: KernelFilterStats,
    *,
    name_limit: int = 20,
) -> str:
    """Build a bounded diagnostic for a filter that produced no kernel groups."""
    available_names = sorted(stats.available_kernel_names)
    sample = ", ".join(available_names[:name_limit]) or "<none>"
    omitted = len(available_names) - min(len(available_names), name_limit)
    suffix = f" (+{omitted} more)" if omitted else ""
    return (
        "No kernels matched the parse-time kernel allowlist "
        f"'{','.join(patterns)}'. Available kernel names "
        f"({len(available_names)}): {sample}{suffix}. "
        f"{stats.unknown_name_count} relevant event(s) had no usable kernel name."
    )
