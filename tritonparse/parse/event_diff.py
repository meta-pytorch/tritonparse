#  Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from .sourcemap_utils import _flatten_dict, _to_ranges, _unflatten_dict

# Fields that are expected to vary but are not useful to list out in the diff.
SUMMARY_FIELDS = ["pid", "timestamp", "stream", "function", "data_ptr"]

# Fields to completely exclude from launch diff (internal tracking fields)
EXCLUDED_FIELDS = ["occurrence_id", "launch_group_hash", "autotune_launch_type"]


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


def _generate_autotune_analysis_events(
    autotune_sessions: Dict[str, Dict[str, Any]],
    autotune_winners: Dict[str, str],
    compilations_by_hash: Dict[str, Any],
    session_stacks: Dict[str, List[Dict[str, Any]]],
    launch_by_group_hash: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Generates autotune_analysis events from grouped compilation sessions.

    Args:
        autotune_sessions: Dict mapping session_id to
            {"compilations": [...], "launch_group_hashes": set([...])}.
        autotune_winners: Dict mapping session_id to the selected launch_group_hash
            (the winning compilation hash is derived from the launch event).
        compilations_by_hash: Dict containing processed kernel data,
            used to find output files.
        session_stacks: Dict mapping session_id to the user call stack.
        launch_by_group_hash: Dict mapping launch_group_hash to launch_event data.

    Returns:
        A dictionary mapping output file paths to a list of autotune_analysis
        event strings.
    """
    output_events: Dict[str, List[str]] = defaultdict(list)

    # First pass: Build hash → groups mapping from sessions with benchmarks
    # Each compilation hash maps to a list of groups it belongs to
    # This allows cached sessions to be associated with all possible groups
    hash_to_groups: Dict[str, List[List[str]]] = defaultdict(list)

    for _session_id, session_data in autotune_sessions.items():
        if not session_data:
            continue
        compilation_events = session_data.get("compilations", [])
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
    for session_id, session_data in autotune_sessions.items():
        if not session_data:
            continue

        # Get compilation and launch data
        compilation_events = session_data["compilations"]
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
                        value_key = json.dumps(arg_val, sort_keys=True)
                    except TypeError:
                        value_key = json.dumps(str(arg_val))
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
                        comp_json_str = compilations_by_hash[ch].get("compilation")
                        if comp_json_str:
                            comp_event = json.loads(comp_json_str)
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
        selected_launch_group_hash = autotune_winners.get(session_id)
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
            "session_stack": session_stacks.get(session_id, []),
            "name": name,
            "selected_hash": autotune_winners.get(session_id),
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

        output_events[output_file].append(
            json.dumps(analysis_event, separators=(",", ":")) + "\n"
        )

    # Third pass: Generate autotune_summary event with winner usage statistics
    # This provides a global view of how often each winner was used
    # We count all winner runs, including:
    # 1. Winner run after benchmark (winner_occurrence_ids is not empty)
    # 2. Cached winner call (benchmark_occurrence_ids is empty)
    winner_run_counts: Dict[str, int] = defaultdict(int)
    for session_id, session_data in autotune_sessions.items():
        if not session_data:
            continue
        winner_occurrence_ids = session_data.get("winner_occurrence_ids", [])
        # Count sessions that have winner runs (either after benchmark or cached)
        if len(winner_occurrence_ids) > 0:
            # Calculate winner_compilation_hash the same way as in Second pass
            selected_launch_group_hash = autotune_winners.get(session_id)
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
        summary_line = json.dumps(summary_event, separators=(",", ":")) + "\n"
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
        return (
            _unflatten_dict(_flatten_dict(launch_events[0])),
            {},
            _to_ranges(launch_index_map),
        )

    # Group values by key
    data_by_key = defaultdict(lambda: defaultdict(list))
    for i, launch in enumerate(launch_events):
        launch_flat = _flatten_dict(launch)
        for key, value in launch_flat.items():
            # JSON doesn't support all Python types as values directly, str is safer
            value_str = json.dumps(value, sort_keys=True)
            data_by_key[key][value_str].append(i)

    sames_flat = {}
    diffs_flat = {}

    for key, value_groups in data_by_key.items():
        # Skip internal tracking fields
        if any(excluded in key for excluded in EXCLUDED_FIELDS):
            continue
        if len(value_groups) == 1:
            # This key has the same value across all launches
            value_str = list(value_groups.keys())[0]
            sames_flat[key] = json.loads(value_str)
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
                            "value": json.loads(value_str),
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
