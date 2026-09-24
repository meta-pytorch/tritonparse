#  Copyright (c) Meta Platforms, Inc. and affiliates.

import ast
import hashlib
import re
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

from tritonparse._json_compat import dumps, loads

from .sourcemap_utils import _flatten_dict, _to_ranges, _unflatten_dict

# Fields that are expected to vary but are not useful to list out in the diff.
SUMMARY_FIELDS = ["pid", "timestamp", "stream", "function", "data_ptr"]

# Fields to completely exclude from launch diff (internal tracking fields)
EXCLUDED_FIELDS = ["occurrence_id", "launch_group_hash", "autotune_launch_type"]

# Upper bound, in characters, on any single string carried through a launch
# payload. Two known values sit far above it and carry no information the
# trace doesn't already hold elsewhere:
#   - `compilation_metadata.asm.*` duplicates IR that the compilation event
#     already stores verbatim in `file_content`.
#   - `function` is an opaque backend handle that some backends report as the
#     entire compiled binary (~4 MB per launch event; 98% of one real trace).
# Both also render into the web viewer as a single unbounded blob.
MAX_VALUE_CHARS = 4096


def summarize_oversized_strings(value: Any) -> Any:
    """
    Recursively replace strings longer than ``MAX_VALUE_CHARS`` with a summary.

    Returns a new object; the input is left untouched so callers that hash or
    compare the original are unaffected.

    Args:
        value: Any JSON-like value (dict, list, or scalar).

    Returns:
        The value with over-long strings swapped for ``<N chars omitted>``.
    """
    if isinstance(value, str):
        if len(value) > MAX_VALUE_CHARS:
            return f"<{len(value)} chars omitted>"
        return value
    if isinstance(value, dict):
        return {k: summarize_oversized_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [summarize_oversized_strings(v) for v in value]
    return value


def _is_excluded_field(flat_key: str) -> bool:
    """True when a flattened launch key is internal bookkeeping, not trace data."""
    return any(excluded in flat_key for excluded in EXCLUDED_FIELDS)


def _format_id_ranges(ids: List[int]) -> str:
    """
    Format a list of occurrence IDs into a human-readable range string.

    Example: [1, 2, 3, 10, 11, 12, 20] -> "1-3, 10-12, 20"
    """
    if not ids:
        return ""
    sorted_ids = sorted(ids)
    ranges = []
    start = prev = sorted_ids[0]

    for i in sorted_ids[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = i
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def _dedup_compilations_by_hash(
    compilation_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Order-preserving dedup of compilation events by `payload.metadata.hash`.

    `parse_single_rank` deliberately appends a compilation event to its
    autotune session bucket EVERY time it sees the hash — even when an
    earlier file already recorded a real compilation for that hash —
    because per-PID session metadata may differ and ingest must preserve
    it. Analysis-time consumers MUST dedup by hash before deciding "is
    this a real benchmark session", otherwise N PIDs hitting the same
    Triton cache would be mis-counted as N distinct configs.

    First-seen wins, matching the cross-PID merge semantics in
    parse_single_rank's kernels_by_hash.
    """
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for comp in compilation_events:
        comp_hash = comp.get("payload", {}).get("metadata", {}).get("hash")
        if not comp_hash:
            # Defensive: malformed event without a hash. Keep it so we
            # don't silently lose data; downstream code already filters
            # by hash presence when building configs.
            deduped.append(comp)
            continue
        if comp_hash in seen:
            continue
        seen.add(comp_hash)
        deduped.append(comp)
    return deduped


def _is_tensor_arg_value(arg_val: Any) -> bool:
    """True when an extracted arg value describes a plain tensor argument."""
    return isinstance(arg_val, dict) and arg_val.get("type") == "tensor"


def _is_tensor_like_value(arg_val: Any) -> bool:
    """True for tensor args including descriptors and foreign wrappers.

    Covers plain tensors plus TMA TensorDescriptors and triton_kernels
    Tensor/Storage wrappers, which nest backing-tensor metadata.
    """
    if not isinstance(arg_val, dict):
        return False
    arg_type = arg_val.get("type", "")
    return (
        arg_type == "tensor"
        or arg_type == "TensorDescriptor"
        or (isinstance(arg_type, str) and arg_type.startswith("triton_kernels.tensor."))
    )


def _extract_scalar_value(arg_val: Any) -> Any:
    """Unwrap {"type": ..., "value": ...} scalar wrappers, if present.

    Falls back to the "repr" payload for values recorded without a
    structured value (None, tuples, other objects), so config values
    such as "None" still compare against best_config spellings.
    """
    if isinstance(arg_val, dict) and "type" in arg_val and "value" in arg_val:
        return arg_val["value"]
    if isinstance(arg_val, dict) and "repr" in arg_val:
        return arg_val["repr"]
    return arg_val


def _is_note_only_launch(extracted: Dict[str, Any]) -> bool:
    """True when a launch carries no extractable arguments.

    During CUDA graph capture, argument extraction is skipped and the
    launch records {"_note": ...} instead of real args. Such launches
    carry no key information (a plain string marker, never a wrapped
    {"type", "value"} kernel parameter), so key partitioning ignores
    them; the same round's warmup launches outside capture still carry
    the full arguments.
    """
    return isinstance(extracted.get("_note"), str)


def _drop_data_ptrs(value: Any) -> Any:
    """Recursively drop data_ptr fields from extracted argument values.

    Tensor descriptors and foreign tensor wrappers nest backing-tensor
    metadata (including volatile data_ptrs) inside the argument dict.
    Dropping data_ptrs keeps signatures stable across reallocations
    while retaining identity metadata (shapes, dtypes, block shapes).
    """
    if isinstance(value, dict):
        return {k: _drop_data_ptrs(v) for k, v in value.items() if k != "data_ptr"}
    if isinstance(value, list):
        return [_drop_data_ptrs(v) for v in value]
    return value


def _fingerprint_value(value: Any) -> str:
    """Stable fingerprint of one config-detection leaf value."""
    try:
        return dumps(_drop_data_ptrs(value), sort_keys=True)
    except TypeError:
        return dumps(str(value))


def _config_leaf_fingerprints(arg_val: Any) -> List[Tuple[Tuple[str, ...], str]]:
    """Locate config correlation at (path, fingerprint) granularity.

    Nested dicts (descriptors, foreign wrappers) are fingerprinted per
    leaf path so a config-shaped nested field (e.g. block_shape) does
    not drag key-shaped siblings (e.g. shape) out of the signature.
    Lists are leaves: per-index splits are contrived, and whole-list
    comparison already matches the legacy behavior when a list mixes
    key/config content. data_ptr entries are skipped everywhere since
    they are volatile across reallocations.
    """
    if not isinstance(arg_val, dict):
        return [((), _fingerprint_value(arg_val))]
    flattened: List[Tuple[Tuple[str, ...], str]] = []
    stack: List[Tuple[Dict[str, Any], Tuple[str, ...]]] = [(arg_val, ())]
    while stack:
        current, path = stack.pop()
        for key in sorted(current, key=str):
            if key == "data_ptr":
                continue
            child = current[key]
            child_path = path + (key,)
            if isinstance(child, dict):
                stack.append((child, child_path))
            else:
                flattened.append((child_path, _fingerprint_value(child)))
    return flattened


def _mask_config_paths(
    value: Dict[str, Any], config_paths: Set[Tuple[str, ...]]
) -> Dict[str, Any]:
    """Copy a nested arg value with config-correlated leaves removed.

    data_ptr fields are dropped first (see _drop_data_ptrs); paths that
    are absent in this launch's shape are skipped.
    """
    masked = _drop_data_ptrs(value)
    for path in config_paths:
        current = masked
        for step in path[:-1]:
            if not isinstance(current, dict) or step not in current:
                current = None
                break
            current = current[step]
        if path and isinstance(current, dict):
            current.pop(path[-1], None)
    return masked


# Marks a nested path missing from a launch's arg shape during config
# detection. Presence/absence correlated with compilation is itself a
# config signal (e.g. "value" vs "repr" in None/int scalar wrappers).
_ABSENT: Any = object()


def _find_config_args(
    group_infos: List[Tuple[str, Optional[str], Dict[str, Any]]],
) -> Dict[str, Optional[Set[Tuple[str, ...]]]]:
    """Find launch args (or nested fields) determined by compilation hash.

    A (arg, path) pair is config-correlated when every compilation hash
    in the session shows exactly one fingerprint for it, while distinct
    hashes disagree (e.g. BLOCK_SIZE_M). Tensor args never participate.
    Launches without a compilation hash cannot be attributed, so they
    are skipped here.

    Returns a mapping of arg name to the config-correlated paths within
    it, or None when the whole arg is config-correlated (scalars and
    fully config-shaped descriptors).
    """
    values_by_path: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Set[Any]]] = (
        defaultdict(lambda: defaultdict(set))
    )
    paths_by_arg: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    hash_universe: Set[str] = set()
    for _group_hash, comp_hash, extracted in group_infos:
        if not comp_hash:
            continue
        hash_universe.add(comp_hash)
        for arg_name, arg_val in extracted.items():
            # Plain tensors never participate: tensor identity is key-like
            # (shapes group rounds), never config-like. Descriptors and
            # foreign wrappers DO participate on data_ptr-free content, so
            # config-shaped fields (e.g. a pre-hook rewriting block_shape
            # per config) are still recognized as config params.
            if _is_tensor_arg_value(arg_val):
                continue
            for path, fingerprint in _config_leaf_fingerprints(arg_val):
                paths_by_arg[arg_name].add(path)
                values_by_path[(arg_name, path)][comp_hash].add(fingerprint)
    for (_arg_name, path), per_hash in values_by_path.items():
        if not path:
            continue
        # A nested path missing under a hash counts as a distinct value;
        # otherwise a field present under only one hash would escape
        # config detection and split the session by config.
        for comp_hash in hash_universe:
            if comp_hash not in per_hash:
                per_hash[comp_hash] = {_ABSENT}
    config_paths_by_arg: Dict[str, Set[Tuple[str, ...]]] = defaultdict(set)
    for (arg_name, path), per_hash in values_by_path.items():
        if len(per_hash) <= 1:
            continue
        if not all(len(values) == 1 for values in per_hash.values()):
            continue
        # The values must actually disagree across hashes; a constant
        # field is not a config param even though it trivially has "one
        # value per hash".
        if len(set().union(*per_hash.values())) > 1:
            config_paths_by_arg[arg_name].add(path)
    config_args: Dict[str, Optional[Set[Tuple[str, ...]]]] = {}
    for arg_name, paths in config_paths_by_arg.items():
        if paths and paths == paths_by_arg[arg_name]:
            config_args[arg_name] = None
        else:
            config_args[arg_name] = paths
    return config_args


def _launch_key_signature(
    extracted: Dict[str, Any],
    config_args: Dict[str, Optional[Set[Tuple[str, ...]]]],
) -> str:
    """Stable signature of a launch's autotune-key arguments.

    Covers all scalar args except autotune config params, plus tensor
    identity metadata (dtype/shape/strides; data_ptr excluded since it
    changes on every allocation). Launches from one key invocation share a
    signature; different keys differ in at least one runtime arg.
    """
    payload: Dict[str, Any] = {}
    for arg_name in sorted(extracted):
        arg_val = extracted[arg_name]
        if arg_name in config_args:
            paths = config_args[arg_name]
            if paths is None or not isinstance(arg_val, dict):
                continue
            # Mixed key/config descriptor: mask only the config paths so
            # key-shaped nested fields still separate key rounds.
            payload[arg_name] = _mask_config_paths(arg_val, paths)
        elif _is_tensor_arg_value(arg_val):
            payload[arg_name] = {
                key: arg_val.get(key)
                for key in ("dtype", "shape", "stride", "strides", "numel")
                if key in arg_val
            }
        elif isinstance(arg_val, dict):
            # Descriptors and foreign wrappers: keep identity metadata
            # but drop volatile nested data_ptrs for stability.
            payload[arg_name] = _drop_data_ptrs(arg_val)
        else:
            payload[arg_name] = arg_val
    try:
        serialized = dumps(payload, sort_keys=True)
    except TypeError:
        serialized = dumps(str(payload))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _split_session_by_launch_key(
    session_id: str,
    session_data: Dict[str, Any],
    launch_by_group_hash: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Partition one call-site session into per-autotune-key sub-sessions.

    Repeated ``Autotuner.run()`` invocations at the same call site (e.g. a
    loop over problem sizes) share one call-site session id. Each invocation
    benchmarks with identical runtime args, so grouping launches by key
    signature recovers the original invocations.

    Single-partition sessions keep the plain coarse session id so their
    output is unchanged; split sessions get "{session_id}:{full_sig}"
    ids (the full 16-char signature: truncating it could collide and
    silently drop a key round). Partitions are ordered by first
    occurrence for stable output. Each partition carries
    launch_group_hashes (ordered as the legacy path orders them) and
    benchmark/winner occurrence ids.

    Limitation: launches without distinguishing arguments cannot be
    attributed to a key (``extracted_args`` is optional in the launch
    schema). Distinct listener keys alone cannot partition
    indistinguishable launches, so such rounds share one partition and
    only the latest listener result is attached. Likewise, an autotune
    key on a tl.constexpr/specialized argument is misread as a config
    param (one value per compilation hash) whenever tensor shapes do
    not already separate the rounds; deriving config fields from
    best_config is impossible on traces without listener events, and
    compilation metadata does not record constexpr values. Note-only
    launches in a split session stay unattributed: their occurrence ids
    are excluded from every partition since no key can be derived for
    them. Result matching is heuristic as well: a result matches when
    every best_config pair resolvable against the winner launch agrees,
    so rounds whose distinguishing config fields are all unresolvable
    are told apart by shared fields and file order only.
    """
    all_occurrences = session_data.get("launch_occurrences", []) or []
    group_hashes = session_data.get("launch_group_hashes", set()) or set()

    group_infos: List[Tuple[str, Optional[str], Dict[str, Any]]] = []
    skipped_note_groups = set()
    for group_hash in group_hashes:
        launch = launch_by_group_hash.get(group_hash, {})
        comp_hash = launch.get("compilation_metadata", {}).get("hash")
        extracted = launch.get("extracted_args", {}) or {}
        # Launches captured inside CUDA graphs carry a _note marker
        # instead of real args. They hold no key information and would
        # pool unrelated rounds into one partition, so key derivation
        # ignores them; emission still counts their occurrences whenever
        # no attribution choice is needed (see below).
        if _is_note_only_launch(extracted):
            skipped_note_groups.add(group_hash)
            continue
        group_infos.append((group_hash, comp_hash, extracted))
    occurrences = [
        record
        for record in all_occurrences
        if record.get("launch_group_hash") not in skipped_note_groups
    ]

    def _single_partition(
        benchmark_ids: List[int], winner_ids: List[int]
    ) -> List[Dict[str, Any]]:
        ordered = sorted(
            (info[0] for info in group_infos),
            key=lambda h: launch_by_group_hash.get(h, {}).get("occurrence_id", 0),
        )
        return [
            {
                "sub_session_id": session_id,
                "launch_group_hashes": ordered,
                "benchmark_occurrence_ids": benchmark_ids,
                "winner_occurrence_ids": winner_ids,
            }
        ]

    if not group_infos:
        # Nothing attributable to split on: either a compilations-only
        # session (benchmark launches untraced) or an all-note session
        # (every launch captured inside CUDA graphs). Preserve the coarse
        # session verbatim (legacy parity): the groups keep output-file
        # lookup and the neither-compilations-nor-launches guard working
        # when the kernels compiled outside this call site, and the coarse
        # occurrence ids keep the real launches instead of reporting an
        # empty, cache-looking session.
        ordered = sorted(
            group_hashes,
            key=lambda h: launch_by_group_hash.get(h, {}).get("occurrence_id", 0),
        )
        return [
            {
                "sub_session_id": session_id,
                "launch_group_hashes": ordered,
                "benchmark_occurrence_ids": list(
                    session_data.get("benchmark_occurrence_ids", [])
                ),
                "winner_occurrence_ids": list(
                    session_data.get("winner_occurrence_ids", [])
                ),
            }
        ]
    if not occurrences:
        # Defensive: same-process ingest always records occurrences, but
        # keep emission working (with coarse occurrence lists) if absent.
        return _single_partition(
            list(session_data.get("benchmark_occurrence_ids", [])),
            list(session_data.get("winner_occurrence_ids", [])),
        )

    config_args = _find_config_args(group_infos)
    signature_by_group = {
        group_hash: _launch_key_signature(extracted, config_args)
        for group_hash, _comp_hash, extracted in group_infos
    }
    grouped: Dict[str, List[str]] = defaultdict(list)
    for group_hash, _comp_hash, _extracted in group_infos:
        grouped[signature_by_group[group_hash]].append(group_hash)

    if len(grouped) == 1:
        # One key round: no attribution choice, so note occurrences join
        # the emission (their arg payload stays excluded from the groups
        # above, keeping _note markers out of the varies table).
        bench_ids = [
            record["occurrence_id"]
            for record in all_occurrences
            if record.get("is_benchmark")
        ]
        winner_ids = [
            record["occurrence_id"]
            for record in all_occurrences
            if not record.get("is_benchmark")
        ]
        return _single_partition(bench_ids, winner_ids)

    def _partition_min_occurrence(sig: str) -> int:
        member_groups = set(grouped[sig])
        occs = [
            record["occurrence_id"]
            for record in occurrences
            if record.get("launch_group_hash") in member_groups
        ]
        return min(occs) if occs else 0

    partitions = []
    for sig in sorted(grouped, key=_partition_min_occurrence):
        member_groups = set(grouped[sig])
        bench_ids = []
        winner_ids = []
        for record in occurrences:
            if record.get("launch_group_hash") not in member_groups:
                continue
            if record.get("is_benchmark"):
                bench_ids.append(record["occurrence_id"])
            else:
                winner_ids.append(record["occurrence_id"])
        ordered = sorted(
            grouped[sig],
            key=lambda h: launch_by_group_hash.get(h, {}).get("occurrence_id", 0),
        )
        partitions.append(
            {
                "sub_session_id": f"{session_id}:{sig}",
                "launch_group_hashes": ordered,
                "benchmark_occurrence_ids": bench_ids,
                "winner_occurrence_ids": winner_ids,
            }
        )
    return partitions


def _resolve_partition_winner(
    winner_occurrence_ids: List[int],
    occurrence_to_group: Dict[int, str],
    launch_by_group_hash: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (selected launch_group_hash, winner compilation hash).

    Mirrors the legacy last-non-benchmark-launch rule, scoped to one
    partition: the latest winner occurrence whose launch carries a hash.
    """
    for occurrence_id in reversed(winner_occurrence_ids):
        group_hash = occurrence_to_group.get(occurrence_id)
        if not group_hash:
            continue
        launch = launch_by_group_hash.get(group_hash, {})
        comp_hash = launch.get("compilation_metadata", {}).get("hash")
        if comp_hash:
            return group_hash, comp_hash
    return None, None


# A value runs until the next "name:" pair (or end of string), so tuple/list
# values containing commas are kept intact.
_BEST_CONFIG_PAIR_RE = re.compile(
    r"([A-Za-z_]\w*)\s*:\s*(.*?)(?=,\s*[A-Za-z_]\w*\s*:|$)"
)


def _parse_best_config(best_config: Any) -> Dict[str, str]:
    """Parse Triton Config str form ("BLOCK_SIZE_M: 16, num_warps: 1, ...")."""
    if not isinstance(best_config, str):
        return {}
    return {
        match.group(1): match.group(2).strip()
        for match in _BEST_CONFIG_PAIR_RE.finditer(best_config)
    }


def _parse_sequence_literal(text: str) -> Optional[List[Any]]:
    """Parse a best_config sequence spelling ("(2, 1, 1)") into a list."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _config_value_matches(expected: str, actual: Any) -> bool:
    """Compare a parsed best_config value against a launch value.

    Int-aware first (exact for large ints; also accepts hex), then
    sequences (best_config prints tuples while JSON round-trips turn
    them into lists), then numeric, then string fallback. Bools compare
    against true/false/1/0 spellings.
    """
    text = expected.strip()
    if actual is None:
        return text in ("None", "null", "")
    if isinstance(actual, bool):
        normalized = text.lower()
        if normalized in ("1", "true"):
            return actual is True
        if normalized in ("0", "false"):
            return actual is False
        return False
    if isinstance(actual, int):
        try:
            return int(text, 0) == actual
        except ValueError:
            pass
    if isinstance(actual, (list, tuple)):
        parsed = _parse_sequence_literal(text)
        return parsed is not None and parsed == list(actual)
    try:
        return float(text) == float(actual)
    except (TypeError, ValueError):
        return text == str(actual)


def _match_autotune_result(
    results: List[Dict[str, Any]],
    winner_launch: Dict[str, Any],
    skip_indices: Set[int],
    winner_warps_base: Optional[int] = None,
) -> Optional[int]:
    """Match listener results to a partition via its winner launch.

    Every best_config pair resolvable against the winner's config values
    (extracted scalar args, then compilation metadata) must agree.
    Tensor-like extracted args are skipped (a config name can never
    denote a tensor). Under warp specialization the launch metadata
    carries the expanded num_warps while best_config prints the
    requested one, so num_warps compares against winner_warps_base when
    the compilation payload recovered one. Returns the index of the
    first match in file order that is not in skip_indices, so each
    result is consumed by at most one partition.

    This is heuristic: pairs unresolvable against the winner (e.g. BLOCK
    sizes, which are neither kernel args nor compilation metadata) are
    skipped, so a match on shared fields alone can misattribute results
    when file order disagrees with round order.
    """
    extracted = winner_launch.get("extracted_args", {}) or {}
    compilation_metadata = winner_launch.get("compilation_metadata", {}) or {}
    for index, result in enumerate(results):
        if index in skip_indices:
            continue
        pairs = _parse_best_config(result.get("best_config"))
        compared = 0
        matched = 0
        for name, expected in pairs.items():
            if name == "num_warps" and winner_warps_base is not None:
                actual = winner_warps_base
            elif name in extracted and not _is_tensor_like_value(extracted[name]):
                actual = _extract_scalar_value(extracted[name])
            elif name in compilation_metadata:
                actual = compilation_metadata[name]
            else:
                continue
            compared += 1
            if _config_value_matches(expected, actual):
                matched += 1
        if compared > 0 and matched == compared:
            return index
    return None


def _globally_referenced_hashes(
    partitions: List[Dict[str, Any]],
    launch_by_group_hash: Dict[str, Dict[str, Any]],
) -> Set[str]:
    """Compilation hashes referenced by any partition's launches."""
    referenced = set()
    for part in partitions:
        for group_hash in part["launch_group_hashes"]:
            launch = launch_by_group_hash.get(group_hash, {})
            comp_hash = launch.get("compilation_metadata", {}).get("hash")
            if comp_hash:
                referenced.add(comp_hash)
    return referenced


def _partition_compilations(
    part: Dict[str, Any],
    coarse_compilations: List[Dict[str, Any]],
    launch_by_group_hash: Dict[str, Dict[str, Any]],
    unattributed_hashes: Set[str],
) -> List[Dict[str, Any]]:
    """Compilations for a partition, in coarse order.

    Besides compilations referenced by this partition's launches, this
    keeps compilations unattributed to any partition (e.g. a candidate
    whose benchmark launches were never traced), so a split never
    silently drops a config that the unsplit session would have shown.
    """
    referenced = set(unattributed_hashes)
    for group_hash in part["launch_group_hashes"]:
        launch = launch_by_group_hash.get(group_hash, {})
        comp_hash = launch.get("compilation_metadata", {}).get("hash")
        if comp_hash:
            referenced.add(comp_hash)
    return [
        comp
        for comp in coarse_compilations
        if comp.get("payload", {}).get("metadata", {}).get("hash") in referenced
    ]


def _build_sub_session(
    part: Dict[str, Any],
    coarse_compilations: List[Dict[str, Any]],
    occurrence_to_group: Dict[int, str],
    results: List[Dict[str, Any]],
    used_result_indices: Set[int],
    unattributed_hashes: Set[str],
    single: bool,
    launch_by_group_hash: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Build one expanded sub-session entry plus its selected group hash."""
    if single:
        # No split: keep every coarse compilation (including ones no
        # launch references) so output matches the legacy path.
        part_compilations = list(coarse_compilations)
    else:
        part_compilations = _partition_compilations(
            part, coarse_compilations, launch_by_group_hash, unattributed_hashes
        )

    selected_group, winner_hash = _resolve_partition_winner(
        part["winner_occurrence_ids"],
        occurrence_to_group,
        launch_by_group_hash,
    )

    matched_result: Optional[Dict[str, Any]] = None
    if results:
        if single:
            # Legacy parity: ingest used to keep only the last result.
            matched_result = results[-1]
        elif selected_group:
            # Warp-specialized kernels record the requested num_warps
            # as num_warps_base on the compilation payload; the launch
            # metadata only carries the expanded count.
            winner_warps_base = None
            if winner_hash:
                for comp in part_compilations:
                    comp_meta = comp.get("payload", {}).get("metadata", {})
                    if comp_meta.get("hash") == winner_hash:
                        winner_warps_base = comp_meta.get("num_warps_base")
                        break
            match_index = _match_autotune_result(
                results,
                launch_by_group_hash.get(selected_group, {}),
                used_result_indices,
                winner_warps_base,
            )
            if match_index is not None:
                used_result_indices.add(match_index)
                matched_result = results[match_index]

    sub_data: Dict[str, Any] = {
        "compilations": part_compilations,
        "launch_group_hashes": set(part["launch_group_hashes"]),
        "benchmark_occurrence_ids": part["benchmark_occurrence_ids"],
        "winner_occurrence_ids": part["winner_occurrence_ids"],
    }
    if matched_result:
        sub_data["autotune_result"] = matched_result
    return sub_data, selected_group


def _expand_sessions_by_launch_key(
    autotune_sessions: Dict[str, Dict[str, Any]],
    session_stacks: Dict[str, List[Dict[str, Any]]],
    launch_by_group_hash: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, List[Dict[str, Any]]]]:
    """Expand call-site sessions into per-autotune-key sub-sessions.

    Returns (expanded_sessions, expanded_winners, expanded_stacks) with the
    same shapes the emission passes already consume: each sub-session keeps
    compilations, launch_group_hashes, and occurrence id lists; winners map
    to the selected launch_group_hash; stacks still describe the shared
    call site. Each sub-session also carries its own "autotune_result"
    (matched by best config with each result consumed at most once, or
    the last result when no split happened).
    """
    expanded_sessions: Dict[str, Dict[str, Any]] = {}
    expanded_winners: Dict[str, str] = {}
    expanded_stacks: Dict[str, List[Dict[str, Any]]] = {}

    for session_id, session_data in autotune_sessions.items():
        if not session_data:
            continue
        partitions = _split_session_by_launch_key(
            session_id, session_data, launch_by_group_hash
        )
        if not partitions:
            continue
        occurrence_to_group = {
            record["occurrence_id"]: record.get("launch_group_hash", "")
            for record in session_data.get("launch_occurrences", []) or []
        }
        # Deduped coarse compilations define the reference order; shared
        # compilations legitimately appear in several partitions.
        coarse_compilations = _dedup_compilations_by_hash(
            session_data.get("compilations", [])
        )
        results = list(session_data.get("autotune_results", []) or [])
        legacy_single = session_data.get("autotune_result")
        if legacy_single and not results:
            results = [legacy_single]
        single = len(partitions) == 1
        used_result_indices: Set[int] = set()
        # Compilations no partition's launches reference cannot be
        # attributed to a key round; promote them to every partition so
        # a split never drops them (single-partition sessions already
        # keep all coarse compilations).
        unattributed_hashes = {
            comp.get("payload", {}).get("metadata", {}).get("hash")
            for comp in coarse_compilations
        } - _globally_referenced_hashes(partitions, launch_by_group_hash)
        unattributed_hashes.discard(None)

        for part in partitions:
            sub_id = part["sub_session_id"]
            sub_data, selected_group = _build_sub_session(
                part,
                coarse_compilations,
                occurrence_to_group,
                results,
                used_result_indices,
                unattributed_hashes,
                single,
                launch_by_group_hash,
            )
            expanded_sessions[sub_id] = sub_data
            if selected_group:
                expanded_winners[sub_id] = selected_group
            expanded_stacks[sub_id] = session_stacks.get(session_id, [])

    return expanded_sessions, expanded_winners, expanded_stacks


def _generate_autotune_analysis_events(
    autotune_sessions: Dict[str, Dict[str, Any]],
    compilations_by_hash: Dict[str, Any],
    session_stacks: Dict[str, List[Dict[str, Any]]],
    launch_by_group_hash: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Generates autotune_analysis events from grouped compilation sessions.

    Call-site sessions are first expanded into per-autotune-key
    sub-sessions (see _expand_sessions_by_launch_key); every pass below
    then operates on sub-sessions. Unsplit sessions keep their plain
    session id and behave exactly as before.

    Args:
        autotune_sessions: Dict mapping session_id to
            {"compilations": [...], "launch_group_hashes": set([...]),
             "launch_occurrences": [...], "autotune_results": [...]}.
        compilations_by_hash: Dict containing processed kernel data,
            used to find output files.
        session_stacks: Dict mapping session_id to the user call stack.
        launch_by_group_hash: Dict mapping launch_group_hash to launch_event data.

    Returns:
        A dictionary mapping output file paths to a list of autotune_analysis
        event strings.
    """
    output_events: Dict[str, List[str]] = defaultdict(list)

    # Expand each call-site session into per-autotune-key sub-sessions.
    # Winners are resolved per sub-session from occurrence records, and
    # each sub-session carries its own matched autotune result.
    expanded_sessions, expanded_winners, expanded_stacks = (
        _expand_sessions_by_launch_key(
            autotune_sessions, session_stacks, launch_by_group_hash
        )
    )

    # Pre-compute hash-deduped compilations per session. Cross-PID merge
    # in parse_single_rank intentionally appends every observation; the
    # analysis below MUST collapse them by hash before counting "configs"
    # or judging "is this a benchmark session". See
    # _dedup_compilations_by_hash for the rationale.
    deduped_compilations: Dict[str, List[Dict[str, Any]]] = {
        sid: _dedup_compilations_by_hash(sd.get("compilations", []) if sd else [])
        for sid, sd in expanded_sessions.items()
    }

    # First pass: Build hash → groups mapping from sessions with benchmarks
    # Each compilation hash maps to a list of groups it belongs to
    # This allows cached sessions to be associated with all possible groups
    hash_to_groups: Dict[str, List[List[str]]] = defaultdict(list)

    for _session_id, session_data in expanded_sessions.items():
        if not session_data:
            continue
        compilation_events = deduped_compilations[_session_id]
        if len(compilation_events) < 2:
            # Only sessions with actual benchmarks (2+ compilations) define groups
            continue

        # Extract compilation hashes for this session (this is the "group")
        compilation_hashes: List[str] = []
        for comp in compilation_events:
            meta = comp.get("payload", {}).get("metadata", {})
            comp_hash = meta.get("hash")
            if comp_hash:
                compilation_hashes.append(comp_hash)

        if not compilation_hashes:
            continue

        # Map each hash in this group to the group itself
        # This allows lookup from any hash in the group
        for h in compilation_hashes:
            # Avoid adding duplicate groups
            if compilation_hashes not in hash_to_groups[h]:
                hash_to_groups[h].append(compilation_hashes)

    # Second pass: Generate autotune_analysis events
    for session_id, session_data in expanded_sessions.items():
        if not session_data:
            continue

        # Get compilation and launch data — compilations come from the
        # hash-deduped cache so cross-PID duplicates don't inflate counts.
        compilation_events = deduped_compilations[session_id]
        launch_group_hashes = session_data.get("launch_group_hashes", set())
        # Convert to a deterministically ordered list for downstream analysis
        launch_group_hashes = sorted(
            launch_group_hashes,
            key=lambda h: launch_by_group_hash.get(h, {}).get("occurrence_id", 0),
        )

        # Get occurrence_ids for benchmark and winner launches
        benchmark_occurrence_ids = session_data.get("benchmark_occurrence_ids", [])
        winner_occurrence_ids = session_data.get("winner_occurrence_ids", [])

        # Skip sessions with neither compilations nor launches
        if not compilation_events and not launch_group_hashes:
            continue

        # Only generate autotune_analysis for sessions with real benchmarking
        # A real autotune session must have at least 2 benchmark launches (one per config)
        # or at least 2 compilations (benchmark launches may not be traced)
        # Sessions with only cached winner launches should not count as autotune sessions
        if len(benchmark_occurrence_ids) < 2 and len(compilation_events) < 2:
            continue

        # Analyze compilation events (if any)
        compilation_analysis: Optional[Dict[str, Any]] = None
        output_file: Optional[str] = None
        name: Optional[str] = None

        if compilation_events:
            first_comp = compilation_events[0]
            metadata = first_comp.get("payload", {}).get("metadata", {})
            first_comp_hash = metadata.get("hash")
            name = metadata.get("name")

            if first_comp_hash and first_comp_hash in compilations_by_hash:
                output_file = compilations_by_hash[first_comp_hash].get("output_file")

                configs = []
                compilation_hashes = []
                for comp in compilation_events:
                    meta = comp.get("payload", {}).get("metadata", {})
                    comp_hash = meta.get("hash")
                    if comp_hash:
                        compilation_hashes.append(comp_hash)
                    # Collect selected config params only when present in metadata
                    compilation_config_params = {}
                    for key in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
                        value = meta.get(key)
                        if value is not None:
                            compilation_config_params[key] = value
                    configs.append(
                        {
                            "compilation_config_params": compilation_config_params,
                            "compilation_hash": meta.get("hash"),
                        }
                    )

                compilation_analysis = {
                    "configs": configs,
                    "compilation_hashes": compilation_hashes,
                    "common_info": {
                        "stack": first_comp.get("stack"),
                        "python_source": first_comp.get("payload", {}).get(
                            "python_source"
                        ),
                    },
                }

        # Analyze launch events (if any)
        launch_analysis: Optional[Dict[str, Any]] = None
        autotune_args_summary: Optional[Dict[str, Any]] = None

        if launch_group_hashes:
            launch_params_diff = _analyze_launch_params(
                launch_group_hashes, launch_by_group_hash
            )

            # Build autotune_args_summary with full distributions
            sames_args = (
                launch_params_diff.get("sames", {}).get("extracted_args", {})
                if isinstance(launch_params_diff, dict)
                else {}
            )

            # Aggregate full value distributions per compilation config
            per_config_aggregates: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
            arg_first_seen_order: OrderedDict[str, None] = OrderedDict()

            for idx, h in enumerate(launch_group_hashes):
                ev = launch_by_group_hash.get(h, {})
                if not isinstance(ev, dict):
                    continue
                comp_hash = ev.get("compilation_metadata", {}).get("hash")
                if not comp_hash:
                    continue
                extracted = ev.get("extracted_args", {}) or {}

                # Record stable argument order by first appearance
                for arg_name in extracted.keys():
                    if arg_name not in arg_first_seen_order:
                        arg_first_seen_order[arg_name] = None

                # Aggregate distributions per config
                config_bucket = per_config_aggregates.setdefault(comp_hash, {})
                for arg_name, arg_val in extracted.items():
                    try:
                        value_key = dumps(
                            arg_val,
                            sort_keys=True,
                        )
                    except TypeError:
                        value_key = dumps(str(arg_val))
                    arg_bucket = config_bucket.setdefault(arg_name, {})
                    if value_key not in arg_bucket:
                        arg_bucket[value_key] = {
                            "value": arg_val,
                            "count": 1,
                            "_first": idx,
                        }
                    else:
                        arg_bucket[value_key]["count"] += 1

            # Build per-config varied args with full values
            per_config_args: Dict[str, Any] = {}
            for comp_hash, arg_map in per_config_aggregates.items():
                per_config_entry: Dict[str, Any] = {}
                for arg_name, grouped in arg_map.items():
                    entries = list(grouped.values())
                    entries.sort(key=lambda d: (-d["count"], d["_first"]))
                    for e in entries:
                        e.pop("_first", None)
                    per_config_entry[arg_name] = {
                        "unique_count": len(entries),
                        "values": entries,
                    }
                per_config_args[comp_hash] = per_config_entry

            # Build a stable arg order from first appearance
            arg_order = list(arg_first_seen_order.keys())
            remaining = set(sames_args.keys())
            for cfg in per_config_args.values():
                remaining.update(cfg.keys())
            for n in arg_order:
                remaining.discard(n)
            if remaining:
                arg_order.extend(sorted(remaining))

            # Attach compilation_config_params for each compilation hash
            if compilation_analysis and "configs" in compilation_analysis:
                for entry in compilation_analysis["configs"]:
                    ch = entry.get("compilation_hash")
                    if ch and ch in per_config_args:
                        per_config_args[ch]["compilation_config_params"] = entry.get(
                            "compilation_config_params"
                        )
            else:
                # No compilation_analysis: look up config params from compilations_by_hash
                for ch in per_config_args.keys():
                    if ch in compilations_by_hash:
                        comp_data = compilations_by_hash[ch].get("compilation")
                        if comp_data:
                            # Handle both dict and JSON string formats
                            if isinstance(comp_data, dict):
                                comp_event = comp_data
                            else:
                                comp_event = loads(comp_data)
                            meta = comp_event.get("payload", {}).get("metadata", {})
                            config_params = {}
                            for key in (
                                "num_warps",
                                "num_stages",
                                "num_ctas",
                                "maxnreg",
                            ):
                                value = meta.get(key)
                                if value is not None:
                                    config_params[key] = value
                            if config_params:
                                per_config_args[ch]["compilation_config_params"] = (
                                    config_params
                                )

            # Build autotune_configs summary across configs
            def _is_tensor_value(val: Any) -> bool:
                try:
                    return isinstance(val, dict) and val.get("type") == "tensor"
                except Exception:
                    return False

            autotune_configs: Dict[str, Any] = {"sames": {}, "varies": {}}
            config_hashes = list(per_config_args.keys())

            # Summarize compilation_config_params
            all_comp_param_keys: set[str] = set()
            for ch in config_hashes:
                comp_params = (
                    per_config_args.get(ch, {}).get("compilation_config_params", {})
                    or {}
                )
                all_comp_param_keys.update(comp_params.keys())

            for key in sorted(all_comp_param_keys):
                values_by_ch = {}
                all_equal = True
                baseline = None
                for ch in config_hashes:
                    comp_params = (
                        per_config_args.get(ch, {}).get("compilation_config_params", {})
                        or {}
                    )
                    v = comp_params.get(key, None)
                    values_by_ch[ch] = v
                    if baseline is None:
                        baseline = v
                    if v != baseline:
                        all_equal = False
                if all_equal:
                    autotune_configs["sames"][key] = baseline
                else:
                    autotune_configs["varies"][key] = values_by_ch

            # Summarize per-config args (filter out tensor args)
            reserved_per_config_keys = {"compilation_config_params"}
            all_launch_arg_names: set[str] = set()
            for ch in config_hashes:
                la = per_config_args.get(ch, {}) or {}
                for k in la.keys():
                    if k not in reserved_per_config_keys:
                        all_launch_arg_names.add(k)

            for arg_name in sorted(all_launch_arg_names):
                # Skip tensor args entirely
                tensor_found_anywhere = False
                for ch in config_hashes:
                    la = per_config_args.get(ch, {}) or {}
                    dist = la.get(arg_name)
                    if not dist:
                        continue
                    for ve in dist.get("values") or []:
                        if _is_tensor_value(ve.get("value")):
                            tensor_found_anywhere = True
                            break
                    if tensor_found_anywhere:
                        break
                if tensor_found_anywhere:
                    continue

                all_single_and_equal = True
                baseline_val = None
                for ch in config_hashes:
                    la = per_config_args.get(ch, {}) or {}
                    dist = la.get(arg_name)
                    if (
                        not dist
                        or not isinstance(dist, dict)
                        or dist.get("unique_count") != 1
                    ):
                        all_single_and_equal = False
                        continue
                    v = (dist.get("values") or [{}])[0].get("value")
                    if baseline_val is None:
                        baseline_val = v
                    if v != baseline_val:
                        all_single_and_equal = False

                if all_single_and_equal and baseline_val is not None:
                    autotune_configs["sames"][arg_name] = baseline_val
                else:
                    # Build per-config view
                    per_ch_view = {}
                    for ch in config_hashes:
                        la = per_config_args.get(ch, {}) or {}
                        dist = la.get(arg_name)
                        if not dist:
                            per_ch_view[ch] = None
                            continue
                        if dist.get("unique_count") == 1:
                            v = (dist.get("values") or [{}])[0].get("value")
                            per_ch_view[ch] = v if not _is_tensor_value(v) else None
                        else:
                            per_ch_view[ch] = {
                                "unique_count": dist.get("unique_count"),
                                "values": dist.get("values"),
                            }
                    autotune_configs["varies"][arg_name] = per_ch_view

            autotune_args_summary = {
                "summary_version": 1,
                "unchanged_args": sames_args,
                "per_config_args": per_config_args,
                "arg_order": arg_order,
                "autotune_configs": autotune_configs,
            }

            launch_analysis = {
                "launch_group_hashes": launch_group_hashes,
                "launch_params_diff": launch_params_diff,
            }

        # If no output_file from compilation, try to get it from first launch
        if not output_file and launch_group_hashes:
            first_launch_hash = launch_group_hashes[0]
            if first_launch_hash in launch_by_group_hash:
                first_launch = launch_by_group_hash[first_launch_hash]
                kernel_hash = first_launch.get("compilation_metadata", {}).get("hash")
                if kernel_hash and kernel_hash in compilations_by_hash:
                    output_file = compilations_by_hash[kernel_hash].get("output_file")
                if not name:
                    name = first_launch.get("compilation_metadata", {}).get("name")

        # Skip if we still can't determine output file
        if not output_file:
            continue

        # Resolve winner_compilation_hash from selected launch_group_hash
        winner_compilation_hash: Optional[str] = None
        selected_launch_group_hash = expanded_winners.get(session_id)
        if (
            selected_launch_group_hash
            and selected_launch_group_hash in launch_by_group_hash
        ):
            selected_launch_event = launch_by_group_hash.get(
                selected_launch_group_hash, {}
            )
            winner_compilation_hash = selected_launch_event.get(
                "compilation_metadata", {}
            ).get("hash")

        # Determine possible_groups for kernel association
        # This field helps the frontend associate autotune sessions with kernels
        # It contains a list of groups (each group is a list of compilation hashes)
        compilation_hashes = (
            compilation_analysis.get("compilation_hashes", [])
            if compilation_analysis
            else []
        )
        if compilation_hashes:
            # Session has actual benchmarks, use its own compilation_hashes as a single group
            possible_groups: List[List[str]] = [compilation_hashes]
        elif winner_compilation_hash:
            # Cached session: look up all groups that contain this winner_hash
            possible_groups = hash_to_groups.get(winner_compilation_hash, [])
        else:
            possible_groups = []

        analysis_event: Dict[str, Any] = {
            "event_type": "autotune_analysis",
            "session_id": session_id,
            "session_stack": expanded_stacks.get(session_id, []),
            "name": name,
            "selected_hash": expanded_winners.get(session_id),
            "winner_compilation_hash": winner_compilation_hash,
            "possible_groups": possible_groups,
            "compilation_analysis": compilation_analysis,
            "launch_analysis": launch_analysis,
            # cache_usage is True only when there are no benchmark launches
            # (i.e., the session just used a cached winner without benchmarking)
            "cache_usage": len(benchmark_occurrence_ids) == 0,
            # Launch occurrence ID ranges
            "launch_ranges": {
                "benchmark": _format_id_ranges(benchmark_occurrence_ids),
                "winner": _format_id_ranges(winner_occurrence_ids),
            },
            "launch_occurrence_ids": {
                "benchmark": sorted(benchmark_occurrence_ids),
                "winner": sorted(winner_occurrence_ids),
            },
        }
        if autotune_args_summary is not None:
            analysis_event["autotune_args_summary"] = autotune_args_summary

        # Add authoritative autotune result from AutotuneListener (if available)
        autotune_result = session_data.get("autotune_result")
        if autotune_result:
            analysis_event["autotune_result"] = {
                "best_config": autotune_result["best_config"],
                "configs_timings": autotune_result["configs_timings"],
                "benchmark_duration": autotune_result["duration"],
                "cache_hit": autotune_result["cache_hit"],
            }

        output_events[output_file].append(dumps(analysis_event) + "\n")

    # Third pass: Generate autotune_summary event with winner usage statistics
    # This provides a global view of how often each winner was used
    # We count all winner runs, including:
    # 1. Winner run after benchmark (winner_occurrence_ids is not empty)
    # 2. Cached winner call (benchmark_occurrence_ids is empty)
    winner_run_counts: Dict[str, int] = defaultdict(int)
    for session_id, session_data in expanded_sessions.items():
        if not session_data:
            continue
        winner_occurrence_ids = session_data.get("winner_occurrence_ids", [])
        # Count sessions that have winner runs (either after benchmark or cached)
        if len(winner_occurrence_ids) > 0:
            # Calculate winner_compilation_hash the same way as in Second pass
            selected_launch_group_hash = expanded_winners.get(session_id)
            if (
                selected_launch_group_hash
                and selected_launch_group_hash in launch_by_group_hash
            ):
                selected_launch_event = launch_by_group_hash.get(
                    selected_launch_group_hash, {}
                )
                winner_hash = selected_launch_event.get("compilation_metadata", {}).get(
                    "hash"
                )
                if winner_hash:
                    winner_run_counts[winner_hash] += 1

    # Add summary event to each output file that has autotune_analysis events
    if winner_run_counts:
        summary_event: Dict[str, Any] = {
            "event_type": "autotune_summary",
            "winner_run_counts": dict(winner_run_counts),
        }
        summary_line = dumps(summary_event) + "\n"
        for output_file in output_events.keys():
            output_events[output_file].append(summary_line)

    return output_events


def _generate_launch_diff(
    launches: List[Tuple[Dict[str, Any], int]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, int]]]:
    """
    Compares a list of launch events and returns sames, diffs, and an index map.
    """
    if not launches:
        return {}, {}, []

    launch_events = [launch[0] for launch in launches]
    launch_index_map = [launch[1] for launch in launches]

    if len(launch_events) == 1:
        # A single launch has nothing to compare against, but it still goes
        # through the same field filtering as the multi-launch path below —
        # otherwise internal bookkeeping and multi-MB blobs land in `sames`.
        sames_flat = {
            key: summarize_oversized_strings(value)
            for key, value in _flatten_dict(launch_events[0]).items()
            if not _is_excluded_field(key)
        }
        return (
            _unflatten_dict(sames_flat),
            {},
            _to_ranges(launch_index_map),
        )

    # Group values by key
    data_by_key = defaultdict(lambda: defaultdict(list))
    for i, launch in enumerate(launch_events):
        launch_flat = _flatten_dict(launch)
        for key, value in launch_flat.items():
            # JSON doesn't support all Python types as values directly, str is safer
            value_str = dumps(
                value,
                sort_keys=True,
            )
            data_by_key[key][value_str].append(i)

    sames_flat = {}
    diffs_flat = {}

    for key, value_groups in data_by_key.items():
        # Skip internal tracking fields
        if _is_excluded_field(key):
            continue
        if len(value_groups) == 1:
            # This key has the same value across all launches
            value_str = list(value_groups.keys())[0]
            sames_flat[key] = summarize_oversized_strings(loads(value_str))
        else:
            # This key has different values
            is_summary = any(summary_key in key for summary_key in SUMMARY_FIELDS)
            if is_summary:
                diffs_flat[key] = {
                    "diff_type": "summary",
                    "summary_text": f"Varies across {len(value_groups)} unique values",
                }
            else:
                values_dist = []
                for value_str, indices in value_groups.items():
                    values_dist.append(
                        {
                            "value": summarize_oversized_strings(loads(value_str)),
                            "count": len(indices),
                            "launches": _to_ranges(indices),
                        }
                    )
                # Sort by first occurrence
                values_dist.sort(key=lambda x: x["launches"][0]["start"])
                diffs_flat[key] = {
                    "diff_type": "distribution",
                    "values": values_dist,
                }

    # Unflatten the results
    sames_unflattened = _unflatten_dict(sames_flat)
    diffs_unflattened = _unflatten_dict(diffs_flat)

    # Special handling for extracted_args to create argument_diff structures
    if "extracted_args" in sames_unflattened or "extracted_args" in diffs_unflattened:
        sames_args = sames_unflattened.pop("extracted_args", {})
        diffs_args_flat = diffs_unflattened.pop("extracted_args", {})

        all_arg_names = set(sames_args.keys()) | set(diffs_args_flat.keys())

        final_arg_diffs = {}

        for arg_name in all_arg_names:
            if arg_name in diffs_args_flat:
                # This argument has at least one differing sub-field.
                arg_sames = {}
                arg_diffs_internal = {}

                # Collect all sub-fields for this argument from the original data
                all_sub_fields = set()
                for launch in launch_events:
                    arg_data = launch.get("extracted_args", {}).get(arg_name, {})
                    all_sub_fields.update(arg_data.keys())

                for sub_field in all_sub_fields:
                    flat_key = f"extracted_args.{arg_name}.{sub_field}"
                    if flat_key in diffs_flat:
                        arg_diffs_internal[sub_field] = diffs_flat[flat_key]
                    elif flat_key in sames_flat:
                        arg_sames[sub_field] = sames_flat[flat_key]

                if arg_sames or arg_diffs_internal:
                    final_arg_diffs[arg_name] = {
                        "diff_type": "argument_diff",
                        "sames": arg_sames,
                        "diffs": arg_diffs_internal,
                    }
            elif arg_name in sames_args:
                # This argument is entirely the same across all launches.
                # We move it back to the main sames dict for consistency.
                if "extracted_args" not in sames_unflattened:
                    sames_unflattened["extracted_args"] = {}
                sames_unflattened["extracted_args"][arg_name] = sames_args[arg_name]

        if final_arg_diffs:
            diffs_unflattened["extracted_args"] = final_arg_diffs

    return sames_unflattened, diffs_unflattened, _to_ranges(launch_index_map)


def _analyze_launch_params(
    launch_group_hashes: List[str], launch_by_group_hash: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze launch parameters to find what's the same and what differs across launches.

    Args:
        launch_group_hashes: List of launch hashes for this session
        launch_by_group_hash: Dict mapping launch_group_hash to launch_event data

    Returns:
        Dict with 'sames' and 'diffs' keys containing parameter analysis
    """
    if not launch_group_hashes:
        return {"sames": {}, "diffs": {}}

    # Build input format similar to _generate_launch_diff
    launches_with_indices = []
    for i, launch_hash in enumerate(launch_group_hashes):
        if launch_hash in launch_by_group_hash:
            launch_event = launch_by_group_hash[launch_hash]
            launches_with_indices.append((launch_event, i))

    if not launches_with_indices:
        return {"sames": {}, "diffs": {}}

    # Reuse existing logic from _generate_launch_diff
    sames, diffs, _ = _generate_launch_diff(launches_with_indices)
    return {"sames": sames, "diffs": diffs}
