#  Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

"""
Multi-strategy kernel matcher for trace-level diff.

Matches kernels across two traces using a two-phase approach:

Phase 1 — Group-level matching:
  Groups compilations by kernel name within each trace, then matches
  groups across traces using name, source similarity, and fuzzy name.

Phase 2 — Within-group config pairing:
  For each matched group, pairs individual compilations by configuration
  similarity (num_stages, num_warps, shared memory). Unpaired compilations
  are reported as autotuning extras rather than "only in trace A/B".
"""

import difflib
from typing import Any

from tritonparse.diff.core.diff_types import KernelMatchResult, MatchMethod
from tritonparse.diff.core.event_matcher import get_kernel_hash, get_kernel_name
from tritonparse.diff.core.source_analyzer import normalize_python_source


class KernelMatcher:
    """Matches kernels across two traces using group-aware strategies.

    First groups compilations by kernel name, then matches groups across
    traces, then pairs individual compilations within matched groups by
    configuration similarity.

    Args:
        events_a: List of (compilation_index, event) tuples from trace A.
        events_b: List of (compilation_index, event) tuples from trace B.
    """

    SOURCE_SIMILARITY_THRESHOLD = 0.75
    FUZZY_NAME_THRESHOLD = 0.7

    def __init__(
        self,
        events_a: list[tuple[int, dict[str, Any]]],
        events_b: list[tuple[int, dict[str, Any]]],
    ) -> None:
        self.events_a = list(events_a)
        self.events_b = list(events_b)

    def match(
        self,
    ) -> tuple[
        list[KernelMatchResult],
        list[tuple[int, dict[str, Any]]],
        list[tuple[int, dict[str, Any]]],
        list[tuple[int, dict[str, Any]]],
        list[tuple[int, dict[str, Any]]],
    ]:
        """Run group-aware matching.

        Returns:
            Tuple of (matched_pairs, unmatched_a, unmatched_b, extra_a, extra_b).
            unmatched_a/b: events from kernel groups with no cross-trace match
                (truly absent kernels).
            extra_a/b: unpaired compilations from matched groups
                (autotuning candidates with no counterpart).
        """
        # Phase 0: Hash matching (highest priority, individual events)
        # Hash matches work across different kernel names, so they must
        # run before group-by-name to avoid missing cross-name matches.
        hash_matched, remaining_a, remaining_b = self._match_by_hash(
            self.events_a, self.events_b
        )

        # Phase 1-2: Group-based matching on remaining events
        groups_a = self._group_by_name(remaining_a)
        groups_b = self._group_by_name(remaining_b)

        matched_groups, unmatched_grp_a, unmatched_grp_b = self._match_groups(
            groups_a, groups_b
        )

        # Pair compilations within each matched group by config
        all_matched: list[KernelMatchResult] = list(hash_matched)
        extra_a: list[tuple[int, dict[str, Any]]] = []
        extra_b: list[tuple[int, dict[str, Any]]] = []

        for (
            _name_a,
            _name_b,
            evts_a,
            evts_b,
            group_method,
            group_confidence,
        ) in matched_groups:
            paired, leftover_a, leftover_b = self._pair_within_group(
                evts_a, evts_b, group_method, group_confidence
            )
            all_matched.extend(paired)
            extra_a.extend(leftover_a)
            extra_b.extend(leftover_b)

        # Flatten unmatched groups
        unmatched_a = [e for _name, evts in unmatched_grp_a for e in evts]
        unmatched_b = [e for _name, evts in unmatched_grp_b for e in evts]

        return all_matched, unmatched_a, unmatched_b, extra_a, extra_b

    # ------------------------------------------------------------------
    # Phase 0: Hash-based matching (individual events)
    # ------------------------------------------------------------------

    @staticmethod
    def _match_by_hash(
        events_a: list[tuple[int, dict[str, Any]]],
        events_b: list[tuple[int, dict[str, Any]]],
    ) -> tuple[
        list[KernelMatchResult],
        list[tuple[int, dict[str, Any]]],
        list[tuple[int, dict[str, Any]]],
    ]:
        """Match events by exact kernel hash.

        Groups events by hash on both sides, pairs by ordinal position
        within each hash group. Returns matched pairs and remaining events.
        """
        skip = {"", "unknown"}

        groups_a: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for idx, event in events_a:
            h = get_kernel_hash(event)
            if h not in skip:
                groups_a.setdefault(h, []).append((idx, event))

        groups_b: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for idx, event in events_b:
            h = get_kernel_hash(event)
            if h not in skip:
                groups_b.setdefault(h, []).append((idx, event))

        matched: list[KernelMatchResult] = []
        matched_a_ids: set[int] = set()
        matched_b_ids: set[int] = set()

        common_hashes = set(groups_a) & set(groups_b)
        for h in sorted(common_hashes):
            ga = groups_a[h]
            gb = groups_b[h]
            for i in range(min(len(ga), len(gb))):
                idx_a, evt_a = ga[i]
                idx_b, evt_b = gb[i]
                matched.append(
                    KernelMatchResult(
                        kernel_name_a=get_kernel_name(evt_a),
                        kernel_name_b=get_kernel_name(evt_b),
                        hash_a=get_kernel_hash(evt_a),
                        hash_b=get_kernel_hash(evt_b),
                        event_index_a=idx_a,
                        event_index_b=idx_b,
                        match_method=MatchMethod.HASH,
                        match_confidence=1.0,
                    )
                )
                matched_a_ids.add(idx_a)
                matched_b_ids.add(idx_b)

        remaining_a = [e for e in events_a if e[0] not in matched_a_ids]
        remaining_b = [e for e in events_b if e[0] not in matched_b_ids]

        return matched, remaining_a, remaining_b

    # ------------------------------------------------------------------
    # Phase 1: Group-level matching
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_name(
        events: list[tuple[int, dict[str, Any]]],
    ) -> dict[str, list[tuple[int, dict[str, Any]]]]:
        """Group events by kernel name."""
        groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for idx, event in events:
            name = get_kernel_name(event)
            groups.setdefault(name, []).append((idx, event))
        return groups

    def _match_groups(
        self,
        groups_a: dict[str, list[tuple[int, dict[str, Any]]]],
        groups_b: dict[str, list[tuple[int, dict[str, Any]]]],
    ) -> tuple[
        list[
            tuple[
                str,
                str,
                list[tuple[int, dict[str, Any]]],
                list[tuple[int, dict[str, Any]]],
                MatchMethod,
                float,
            ]
        ],
        list[tuple[str, list[tuple[int, dict[str, Any]]]]],
        list[tuple[str, list[tuple[int, dict[str, Any]]]]],
    ]:
        """Match kernel groups across traces.

        Strategies applied in order: exact name, source similarity, fuzzy name.

        Returns:
            Tuple of (matched_groups, unmatched_groups_a, unmatched_groups_b).
            Each matched group is (name_a, name_b, events_a, events_b, method, confidence).
        """
        skip = {"", "unknown"}
        used_a: set[str] = set()
        used_b: set[str] = set()
        matched: list[
            tuple[
                str,
                str,
                list[tuple[int, dict[str, Any]]],
                list[tuple[int, dict[str, Any]]],
                MatchMethod,
                float,
            ]
        ] = []

        # Strategy 1: Exact name match
        common = (set(groups_a) & set(groups_b)) - skip
        for name in sorted(common):
            matched.append(
                (name, name, groups_a[name], groups_b[name], MatchMethod.NAME, 1.0)
            )
            used_a.add(name)
            used_b.add(name)

        # Strategy 2: Source similarity across group members.
        # Uses the best (max) similarity across sampled event pairs from
        # each group rather than a single representative, so that groups
        # with heterogeneous compilations aren't missed.
        # Note: "unknown" groups ARE eligible for source similarity matching
        # (only exact name and fuzzy name strategies skip them).
        remaining_a = {n: e for n, e in groups_a.items() if n not in used_a}
        remaining_b = {n: e for n, e in groups_b.items() if n not in used_b}

        if remaining_a and remaining_b:
            candidates: list[tuple[float, str, str]] = []
            for na, ea in remaining_a.items():
                for nb, eb in remaining_b.items():
                    best_sim = self._best_source_similarity(ea, eb)
                    if best_sim >= self.SOURCE_SIMILARITY_THRESHOLD:
                        candidates.append((best_sim, na, nb))

            candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
            for sim, na, nb in candidates:
                if na in used_a or nb in used_b:
                    continue
                used_a.add(na)
                used_b.add(nb)
                matched.append(
                    (na, nb, groups_a[na], groups_b[nb], MatchMethod.SOURCE, sim)
                )

        # Strategy 3: Fuzzy name match
        remaining_a = {
            n: e for n, e in groups_a.items() if n not in used_a and n not in skip
        }
        remaining_b = {
            n: e for n, e in groups_b.items() if n not in used_b and n not in skip
        }

        if remaining_a and remaining_b:
            candidates = []
            for na in remaining_a:
                for nb in remaining_b:
                    sim = difflib.SequenceMatcher(None, na, nb).ratio()
                    if sim >= self.FUZZY_NAME_THRESHOLD:
                        candidates.append((sim, na, nb))

            candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
            for sim, na, nb in candidates:
                if na in used_a or nb in used_b:
                    continue
                used_a.add(na)
                used_b.add(nb)
                matched.append(
                    (na, nb, groups_a[na], groups_b[nb], MatchMethod.FUZZY_NAME, sim)
                )

        unmatched_a = [(n, e) for n, e in groups_a.items() if n not in used_a]
        unmatched_b = [(n, e) for n, e in groups_b.items() if n not in used_b]

        return matched, unmatched_a, unmatched_b

    # ------------------------------------------------------------------
    # Phase 2: Within-group config pairing
    # ------------------------------------------------------------------

    def _pair_within_group(
        self,
        events_a: list[tuple[int, dict[str, Any]]],
        events_b: list[tuple[int, dict[str, Any]]],
        group_method: MatchMethod,
        group_confidence: float,
    ) -> tuple[
        list[KernelMatchResult],
        list[tuple[int, dict[str, Any]]],
        list[tuple[int, dict[str, Any]]],
    ]:
        """Pair compilations within a matched group by config similarity.

        Returns:
            (paired, extra_a, extra_b) where extra are unpaired autotuning
            candidates within the group.
        """
        if len(events_a) == 1 and len(events_b) == 1:
            idx_a, event_a = events_a[0]
            idx_b, event_b = events_b[0]

            # Compute the actual similarity for this specific pair
            if group_method in (MatchMethod.SOURCE, MatchMethod.FUZZY_NAME):
                confidence = self._source_similarity(event_a, event_b)
                if confidence < self.SOURCE_SIMILARITY_THRESHOLD:
                    confidence = group_confidence
            else:
                confidence = group_confidence

            return (
                [
                    KernelMatchResult(
                        kernel_name_a=get_kernel_name(event_a),
                        kernel_name_b=get_kernel_name(event_b),
                        hash_a=get_kernel_hash(event_a),
                        hash_b=get_kernel_hash(event_b),
                        event_index_a=idx_a,
                        event_index_b=idx_b,
                        match_method=group_method,
                        match_confidence=confidence,
                    )
                ],
                [],
                [],
            )

        # Multiple compilations — pair by config similarity (greedy)
        candidates: list[tuple[float, int, int, dict[str, Any], dict[str, Any]]] = []
        for idx_a, event_a in events_a:
            for idx_b, event_b in events_b:
                sim = self._config_similarity(event_a, event_b)
                candidates.append((sim, idx_a, idx_b, event_a, event_b))

        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        used_a: set[int] = set()
        used_b: set[int] = set()
        paired: list[KernelMatchResult] = []

        for config_sim, idx_a, idx_b, event_a, event_b in candidates:
            if idx_a in used_a or idx_b in used_b:
                continue
            used_a.add(idx_a)
            used_b.add(idx_b)

            # Report the actual source similarity for this specific pair
            # when the group was matched by source, otherwise use config
            # similarity. Always compute from the paired events themselves
            # so the confidence reflects THIS pair, not the group-level best.
            if group_method in (MatchMethod.SOURCE, MatchMethod.FUZZY_NAME):
                pair_source_sim = self._source_similarity(event_a, event_b)
                if pair_source_sim >= self.SOURCE_SIMILARITY_THRESHOLD:
                    method = group_method
                    confidence = pair_source_sim
                else:
                    method = MatchMethod.CONFIG
                    confidence = config_sim
            elif config_sim >= 0.99:
                method = group_method
                confidence = group_confidence
            else:
                method = MatchMethod.CONFIG
                confidence = config_sim

            paired.append(
                KernelMatchResult(
                    kernel_name_a=get_kernel_name(event_a),
                    kernel_name_b=get_kernel_name(event_b),
                    hash_a=get_kernel_hash(event_a),
                    hash_b=get_kernel_hash(event_b),
                    event_index_a=idx_a,
                    event_index_b=idx_b,
                    match_method=method,
                    match_confidence=confidence,
                )
            )

        extra_a = [(idx, evt) for idx, evt in events_a if idx not in used_a]
        extra_b = [(idx, evt) for idx, evt in events_b if idx not in used_b]

        return paired, extra_a, extra_b

    # ------------------------------------------------------------------
    # Similarity functions
    # ------------------------------------------------------------------

    # Maximum number of events to sample per group for source similarity.
    # Keeps group matching cost bounded at O(K^2 * G_a * G_b) where K is
    # small, instead of O(N * M) for full pairwise.
    _MAX_GROUP_SAMPLES = 5

    def _best_source_similarity(
        self,
        events_a: list[tuple[int, dict[str, Any]]],
        events_b: list[tuple[int, dict[str, Any]]],
    ) -> float:
        """Compute the best source similarity across sampled event pairs.

        Samples up to _MAX_GROUP_SAMPLES events from each group (first,
        last, and evenly-spaced middle events) and returns the maximum
        pairwise source similarity. This balances robustness against
        within-group variance with bounded cost.
        """
        samples_a = self._sample_group(events_a)
        samples_b = self._sample_group(events_b)

        best = 0.0
        for _, evt_a in samples_a:
            for _, evt_b in samples_b:
                sim = self._source_similarity(evt_a, evt_b)
                best = max(best, sim)
                if best >= 0.99:
                    return best
        return best

    @staticmethod
    def _sample_group(
        events: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        """Sample up to _MAX_GROUP_SAMPLES events from a group.

        Picks first, last, and evenly-spaced middle events to cover
        the range of configurations in the group.
        """
        k = KernelMatcher._MAX_GROUP_SAMPLES
        if len(events) <= k:
            return events
        # Always include first and last; fill middle evenly
        indices = {0, len(events) - 1}
        step = (len(events) - 1) / (k - 1)
        for i in range(1, k - 1):
            indices.add(round(i * step))
        return [events[i] for i in sorted(indices)]

    @staticmethod
    def _config_similarity(event_a: dict[str, Any], event_b: dict[str, Any]) -> float:
        """Compute configuration similarity between two compilation events.

        Scores based on matching num_stages (0.4), num_warps (0.4),
        and shared memory closeness (0.2).
        """
        meta_a = event_a.get("payload", {}).get("metadata", {})
        meta_b = event_b.get("payload", {}).get("metadata", {})

        score = 0.0
        if meta_a.get("num_stages") == meta_b.get("num_stages"):
            score += 0.4
        if meta_a.get("num_warps") == meta_b.get("num_warps"):
            score += 0.4

        shared_a = meta_a.get("shared", 0) or 0
        shared_b = meta_b.get("shared", 0) or 0
        if shared_a > 0 and shared_b > 0:
            score += 0.2 * min(shared_a, shared_b) / max(shared_a, shared_b)
        elif shared_a == shared_b:
            score += 0.2

        return score

    @staticmethod
    def _get_event_source(event: dict[str, Any]) -> str:
        """Extract Python source from event."""
        payload = event.get("payload", {})
        python_source = payload.get("python_source", {})
        if isinstance(python_source, dict):
            content = python_source.get("content") or python_source.get("code")
            if content:
                return content
        python_field = payload.get("python", "")
        if python_field:
            return python_field
        return ""

    def _source_similarity(
        self, event_a: dict[str, Any], event_b: dict[str, Any]
    ) -> float:
        """Compute normalized Python source similarity."""
        source_a = self._get_event_source(event_a)
        source_b = self._get_event_source(event_b)
        if not source_a or not source_b:
            return 0.0
        norm_a = normalize_python_source(source_a)
        norm_b = normalize_python_source(source_b)
        if not norm_a or not norm_b:
            return 0.0
        return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
