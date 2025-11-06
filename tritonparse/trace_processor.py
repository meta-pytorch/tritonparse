#  Copyright (c) Meta Platforms, Inc. and affiliates.

import gzip
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List

from .event_diff import _generate_launch_diff
from .ir_analysis import _generate_ir_analysis
from .ir_parser import (
    extract_code_locations,
    extract_loc_definitions,
    extract_ptx_amdgcn_mappings,
)
from .mapper import create_bidirectional_mapping, create_python_mapping
from .sourcemap_utils import get_file_extension, load_ir_contents

logger = logging.getLogger("SourceMapping")


def generate_source_mappings(
    ir_content: str, ir_type: str, other_mappings: List[Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Generate source mappings from intermediate representation (IR) content to the source file.
    Example:
    loc definition: Line 39 in ttir: #loc2 = loc("/tmp/torchinductor_yhao/yp/abcdef.py":20:28)
    loc reference: Line 9 in ttir: %0 = tt.get_program_id x : i32 loc(#loc2)
    Then, the output will be:
    {
        "9": {
            "file": "/tmp/torchinductor_yhao/yp/abcdef.py",
            "line": 20,
            "column": 28,
            "ttir_line": 9
        },
    }

    Args:
        ir_content (str): The content of the intermediate representation.
        ir_type (str): The type of the intermediate representation (e.g., 'ttir').
        other_mappings (List[Any]): A collection of additional mappings, primarily utilized for PTX mappings since PTX's location annotations reference the file name instead of the complete path.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping line numbers to their corresponding source file,
        line, column, and the line number in the IR.
    """
    if other_mappings is None:
        other_mappings = []
    if ir_type == "ptx" or ir_type == "amdgcn":
        return extract_ptx_amdgcn_mappings(ir_content, other_mappings, ir_type)

    loc_defs = extract_loc_definitions(ir_content)
    logger.debug(f"Found {len(loc_defs)} #loc definitions")

    loc_refs = extract_code_locations(ir_content)
    logger.debug(f"Found {len(loc_refs)} loc references")

    mappings = {}
    for ln, loc_id in loc_refs.items():
        if loc_id.startswith("direct:"):
            _, file_path, line, col = loc_id.split(":", 3)
            mappings[str(ln)] = {
                "file": file_path,
                "line": int(line),
                "column": int(col),
                f"{ir_type}_line": ln,
            }
        elif loc_id in loc_defs:
            info = loc_defs[loc_id]
            entry = {
                "file": info["file"],
                "line": info["line"],
                "column": info["column"],
                f"{ir_type}_line": ln,
            }
            # Propagate callsite metadata if present
            if info.get("is_callsite"):
                entry["is_callsite"] = True
                entry["callsite_callee"] = info["callsite_callee"]
                entry["callsite_caller"] = info["callsite_caller"]
            # Propagate alias metadata if present
            if "alias_name" in info:
                entry["alias_name"] = info["alias_name"]
            if "alias_of" in info:
                entry["loc_id"] = loc_id
            mappings[str(ln)] = entry

    # Add separate entries for loc definition lines
    for loc_id, info in loc_defs.items():
        if "def_line" not in info:
            continue
        def_ln = info["def_line"]
        # Only create mapping if this line doesn't already have one
        if str(def_ln) not in mappings:
            entry = {
                "file": info["file"],
                "line": info["line"],
                "column": info["column"],
                f"{ir_type}_line": def_ln,
                "kind": "loc_def",
            }
            if "alias_name" in info:
                entry["alias_name"] = info["alias_name"]
            if "alias_of" in info:
                entry["loc_id"] = loc_id
            mappings[str(def_ln)] = entry

    return mappings


def process_ir(
    key: str,
    file_content: Dict[str, str],
    file_path: Dict[str, str],
    other_mappings: List[Any] = None,
):
    ir_content = load_ir_contents(key, file_content, file_path)
    if not ir_content:
        return {}
    mapping = generate_source_mappings(ir_content, key.split(".")[1], other_mappings)
    logger.debug(f"Generated source mapping for {key}")
    return mapping


def parse_single_trace_content(trace_content: str) -> str:
    """
    Process a single trace content and extract source code mappings.

    This function takes a trace content as input, extracts the IR files, generates source mappings,
    creates bidirectional mappings between different IR types, and updates the payload with the mappings.

    Args:
        trace_content (str): The content of the trace file as a string.

    Returns:
        str: The updated trace content with source mappings as a JSON string.
    """

    entry = json.loads(trace_content)
    if entry.get("event_type") == "compilation":
        payload = entry.setdefault("payload", {})
        file_content = payload.get("file_content", {})
        file_path = payload.get("file_path", {})

        # Find the IR file keys
        ttir_key = next((k for k in file_content if k.endswith(".ttir")), None)
        ttgir_key = next((k for k in file_content if k.endswith(".ttgir")), None)
        ptx_key = next((k for k in file_content if k.endswith(".ptx")), None)
        amdgcn_key = next((k for k in file_content if k.endswith(".amdgcn")), None)
        # Skip if no IR files found
        if not (ttir_key or ttgir_key or ptx_key or amdgcn_key):
            logger.warning("No IR files found in the payload.")
            return trace_content

        # generate ttir->source, ttgir->source, ptx->source
        ttir_map = process_ir(ttir_key, file_content, file_path)
        ttgir_map = process_ir(ttgir_key, file_content, file_path)
        ptx_map = process_ir(ptx_key, file_content, file_path, [ttir_map, ttgir_map])
        amdgcn_map = process_ir(
            amdgcn_key, file_content, file_path, [ttir_map, ttgir_map]
        )

        # Create bidirectional mappings between all IR types
        ir_maps = {
            "ttir": ttir_map,
            "ttgir": ttgir_map,
            "ptx": ptx_map,
            "amdgcn": amdgcn_map,
        }

        # Create mappings between all pairs of IR types
        ir_types = list(ir_maps.keys())
        for i, src_type in enumerate(ir_types):
            for tgt_type in ir_types[i + 1 :]:
                if ir_maps[src_type] and ir_maps[tgt_type]:
                    create_bidirectional_mapping(
                        ir_maps[src_type], ir_maps[tgt_type], src_type, tgt_type
                    )
                    logger.debug(
                        f"Created bidirectional mapping between {src_type} and {tgt_type}"
                    )

        py_map = {}

        if "python_source" in payload:
            logger.debug(
                f"Added Python source information (lines {payload['python_source']['start_line']}-{payload['python_source']['end_line']})"
            )

            # 4. Create Python source to IR mappings. We use the original line numbers as key in the python source code.
            # Create a list of valid IR mappings, filtering out None keys
            ir_mappings = []
            ir_keys_and_maps = [
                (ttir_key, ttir_map),
                (ttgir_key, ttgir_map),
                (ptx_key, ptx_map),
                (amdgcn_key, amdgcn_map),
            ]

            for key, mapping in ir_keys_and_maps:
                if key:
                    ir_mappings.append((get_file_extension(key), mapping))

            py_map = create_python_mapping(ir_mappings)

        # Store the mappings in the payload
        payload["source_mappings"] = {
            "ttir": ttir_map,
            "ttgir": ttgir_map,
            **({"ptx": ptx_map} if ptx_map else {}),
            **({"amdgcn": amdgcn_map} if amdgcn_map else {}),
            "python": py_map,
        }
    # NDJSON format requires a newline at the end of each line
    return json.dumps(entry, separators=(",", ":")) + "\n"


def parse_single_file(
    file_path: str,
    output_dir: str = None,
    split_inductor_compilations: bool = True,
):
    """
    Process a single file, correctly group events by kernel, and extract mappings.

    This function reads a trace file, groups compilation and launch events by
    their kernel hash, generates a launch_diff event for each kernel, and writes
    the processed data to output files.

    Args:
        file_path (str): The path to the file to be processed.
        output_dir (str, optional): Directory to save the output files.
        split_inductor_compilations (bool, optional): Whether to split
            output files by frame_id, compile_id, attempt_id, and compiled_autograd_id.
            Defaults to True. This rule follows tlparse's behavior.
    """
    kernels_by_hash = defaultdict(
        lambda: {"compilation": None, "launches": [], "output_file": None}
    )

    output_dir = output_dir or os.path.dirname(file_path)
    is_compressed_input = file_path.endswith(".bin.ndjson")
    file_handle = (
        gzip.open(file_path, "rt", encoding="utf-8")
        if is_compressed_input
        else open(file_path, "r")
    )

    with file_handle as f:
        file_name = os.path.basename(file_path)
        file_name_without_extension = (
            file_name[:-11] if is_compressed_input else os.path.splitext(file_name)[0]
        )

        for i, line in enumerate(f):
            logger.debug(f"Processing line {i + 1} in {file_path}")
            json_str = line.strip()
            if not json_str:
                continue

            # We don't need to generate full mappings for every line here,
            # just enough to get the event type and necessary IDs.
            try:
                parsed_json = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON on line {i + 1} in {file_path}")
                continue

            event_type = parsed_json.get("event_type", None)
            payload = parsed_json.get("payload", {})

            if event_type == "compilation":
                kernel_hash = payload.get("metadata", {}).get("hash")
                if not kernel_hash:
                    continue

                # Split inductor compilations into separate files
                # This rule follows tlparse's behavior.
                if split_inductor_compilations:
                    pt_info = payload.get("pt_info", {})
                    frame_id = pt_info.get("frame_id")
                    frame_compile_id = pt_info.get("frame_compile_id")
                    attempt_id = pt_info.get("attempt_id", 0)
                    cai = pt_info.get("compiled_autograd_id", "-")
                    if frame_id is not None or frame_compile_id is not None:
                        fname = f"f{frame_id}_fc{frame_compile_id}_a{attempt_id}_cai{cai}.ndjson"
                    else:
                        fname = f"{file_name_without_extension}_mapped.ndjson"
                else:
                    fname = f"{file_name_without_extension}_mapped.ndjson"

                output_file = os.path.join(output_dir, fname)
                # The full processing is deferred until the final write.
                kernels_by_hash[kernel_hash]["compilation"] = json_str
                kernels_by_hash[kernel_hash]["output_file"] = output_file

            elif event_type == "launch":
                kernel_hash = parsed_json.get("compilation_metadata", {}).get("hash")
                if kernel_hash:
                    kernels_by_hash[kernel_hash]["launches"].append(
                        (parsed_json, i + 1)
                    )

    # Organize lines for final output, keyed by output file path
    all_output_lines = defaultdict(list)
    for _kernel_hash, data in kernels_by_hash.items():
        compilation_json_str = data["compilation"]
        launches_with_indices = data["launches"]
        output_file = data["output_file"]

        if not output_file:
            logger.warning(f"No output file for kernel hash {_kernel_hash}, skipping.")
            continue

        # Process the compilation event now to include source mappings
        if compilation_json_str:
            processed_compilation_line = parse_single_trace_content(
                compilation_json_str
            )
            all_output_lines[output_file].append(processed_compilation_line)
            compilation_event = json.loads(processed_compilation_line)
        else:
            compilation_event = None

        for launch_event, _ in launches_with_indices:
            all_output_lines[output_file].append(
                json.dumps(launch_event, separators=(",", ":")) + "\n"
            )

        if compilation_event:
            ir_analysis = _generate_ir_analysis(compilation_event)
            if ir_analysis:
                ir_analysis_event = {
                    "event_type": "ir_analysis",
                    "hash": _kernel_hash,
                    "ir_analysis": ir_analysis,
                }
                all_output_lines[output_file].append(
                    json.dumps(ir_analysis_event, separators=(",", ":")) + "\n"
                )

        if compilation_event and launches_with_indices:
            sames, diffs, launch_index_map = _generate_launch_diff(
                launches_with_indices
            )
            launch_diff_event = {
                "event_type": "launch_diff",
                "hash": _kernel_hash,
                "name": compilation_event.get("payload", {})
                .get("metadata", {})
                .get("name"),
                "total_launches": len(launches_with_indices),
                "launch_index_map": launch_index_map,
                "diffs": diffs,
                "sames": sames,
            }
            all_output_lines[output_file].append(
                json.dumps(launch_diff_event, separators=(",", ":")) + "\n"
            )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for output_file, final_lines in all_output_lines.items():
        with open(output_file, "w") as out:
            out.writelines(final_lines)
