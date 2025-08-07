import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from .sourcemap_utils import _flatten_dict, _to_ranges, _unflatten_dict

# Fields that are expected to vary but are not useful to list out in the diff.
SUMMARY_FIELDS = ["pid", "timestamp", "stream", "function", "data_ptr"]


def _generate_autotune_analysis_events(
    autotune_sessions: Dict[str, Dict[str, List]],
    autotune_winners: Dict[str, str],
    kernels_by_hash: Dict[str, Any],
    session_stacks: Dict[str, List[Dict[str, Any]]],
    launches_by_hash: Dict[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Generates autotune_analysis events from grouped compilation sessions.

    Args:
        autotune_sessions: Dict mapping session_id to {"compilations": [...], "launches": [...]}.
        autotune_winners: Dict mapping session_id to the winning kernel hash.
        kernels_by_hash: Dict containing processed kernel data, used to find output files.
        session_stacks: Dict mapping session_id to the user call stack.
        launches_by_hash: Dict mapping launch_hash to launch_event data.

    Returns:
        A dictionary mapping output file paths to a list of autotune_analysis event strings.
    """
    output_events = defaultdict(list)
    for session_id, session_data in autotune_sessions.items():
        if not session_data:
            continue

        # Get compilation and launch data
        compilation_events = session_data["compilations"]
        launch_hashes = session_data["launches"]
        
        # Skip sessions with neither compilations nor launches
        if not compilation_events and not launch_hashes:
            continue
        
        # Analyze compilation events (if any)
        compilation_analysis = None
        output_file = None
        name = None
        
        if compilation_events:
            first_comp = compilation_events[0]
            metadata = first_comp.get("payload", {}).get("metadata", {})
            first_comp_hash = metadata.get("hash")
            name = metadata.get("name")
            
            if first_comp_hash and first_comp_hash in kernels_by_hash:
                output_file = kernels_by_hash[first_comp_hash].get("output_file")
                
                configs = []
                compilation_hashes = []
                for comp in compilation_events:
                    meta = comp.get("payload", {}).get("metadata", {})
                    compilation_hashes.append(meta.get("hash"))
                    # TODO: the config params should be more complete
                    config_params = {
                        "num_warps": meta.get("num_warps"),
                        "num_stages": meta.get("num_stages"),
                    }
                    # TODO: src_constants don't have too much name info, we should add more info
                    if "src_constants" in meta:
                        config_params.update(meta["src_constants"])
                    configs.append(
                        {"config_params": config_params, "compilation_hash": meta.get("hash")}
                    )
                
                compilation_analysis = {
                    "configs": configs,
                    "compilation_hashes": compilation_hashes,
                    "common_info": {
                        "stack": first_comp.get("stack"),
                        "python_source": first_comp.get("payload", {}).get("python_source"),
                    }
                }
        
        # Analyze launch events (if any)
        launch_analysis = None
        if launch_hashes:
            launch_params_diff = _analyze_launch_params(launch_hashes, launches_by_hash)
            launch_analysis = {
                "launch_hashes": launch_hashes,
                "launch_params_diff": launch_params_diff
            }
            
            # If no output_file from compilation, try to get it from first launch
            if not output_file and launch_hashes:
                first_launch_hash = launch_hashes[0]
                if first_launch_hash in launches_by_hash:
                    first_launch = launches_by_hash[first_launch_hash]
                    kernel_hash = first_launch.get("compilation_metadata", {}).get("hash")
                    if kernel_hash and kernel_hash in kernels_by_hash:
                        output_file = kernels_by_hash[kernel_hash].get("output_file")
                        if not name:
                            # Try to get name from compilation metadata
                            name = first_launch.get("compilation_metadata", {}).get("name")
        
        # Skip if we still can't determine output file
        if not output_file:
            continue

        analysis_event = {
            "event_type": "autotune_analysis",
            "session_id": session_id,
            "session_stack": session_stacks.get(session_id, []),
            "name": name,
            "selected_hash": autotune_winners.get(session_id),
            "compilation_analysis": compilation_analysis,
            "launch_analysis": launch_analysis,
            "cache_usage": compilation_analysis is None,
            "launch_count": len(launch_hashes),
        }
        output_events[output_file].append(
            json.dumps(analysis_event, separators=(",", ":")) + "\n"
        )
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


def _analyze_launch_params(launch_hashes: List[str], launches_by_hash: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze launch parameters to find what's the same and what differs across launches.
    
    Args:
        launch_hashes: List of launch hashes for this session
        launches_by_hash: Dict mapping launch_hash to launch_event data
        
    Returns:
        Dict with 'sames' and 'diffs' keys containing parameter analysis
    """
    if not launch_hashes:
        return {"sames": {}, "diffs": {}}
    
    # Build input format similar to _generate_launch_diff
    launches_with_indices = []
    # TODO: the i here is not the index in the original launch_events, but rather the index in the launch_hashes. This should be fixed in the future if we want to use the index for anything meaningful.
    for i, launch_hash in enumerate(launch_hashes):
        if launch_hash in launches_by_hash:
            launch_event = launches_by_hash[launch_hash]
            launches_with_indices.append((launch_event, i))
    
    if not launches_with_indices:
        return {"sames": {}, "diffs": {}}
    
    # Reuse existing logic from _generate_launch_diff
    sames, diffs, _ = _generate_launch_diff(launches_with_indices)
    return {"sames": sames, "diffs": diffs}
