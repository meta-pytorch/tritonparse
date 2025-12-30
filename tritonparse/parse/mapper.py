#  Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple


logger = logging.getLogger("SourceMapping")


def create_python_mapping(
    ir_maps: List[Tuple[str, Dict[str, Dict[str, Any]]]],
) -> Dict[int, Dict[str, List[int]]]:
    """
    Create a mapping from Python source code to IR mappings. We assume there is only one Python source code for each triton kernel.
    Args:
        ir_maps: A list of tuples containing the IR type and the IR mappings.

    Returns:
        A dictionary mapping Python source code line numbers to their corresponding IR mappings.
    """
    py_map = defaultdict(lambda: defaultdict(list))
    for ir_type, ir_map in ir_maps:
        for line_number, info in ir_map.items():
            py_line_number: int = info["line"]
            py_map[py_line_number][f"{ir_type}_lines"].append(line_number)
    return {k: dict(v) for k, v in py_map.items()}


def create_ir_mapping(
    source_map: Dict[str, Dict[str, Any]], target_map: Dict[str, Dict[str, Any]]
) -> Dict[str, List[int]]:
    """
    Create a mapping from source IR lines to target IR lines.

    This function takes two mappings: one for source IR and one for target IR, and creates a new mapping
    that associates lines in the source IR with corresponding lines in the target IR based on their file,
    line, and column information.

    Args:
        source_map (Dict[str, Dict[str, Any]]): A dictionary mapping source IR line numbers to their source file,
            line, and column information.
        target_map (Dict[str, Dict[str, Any]]): A dictionary mapping target IR line numbers to their source file,
            line, and column information.

    Returns:
        Dict[str, List[int]]: A dictionary mapping source IR line numbers to lists of corresponding target IR line numbers.
    """
    source_to_target = defaultdict(list)

    # Build a mapping from source file locations to target lines
    for tgt_line, tgt_info in target_map.items():
        if "file" in tgt_info and "line" in tgt_info:
            key = f"{tgt_info['file']}:{tgt_info['line']}:{tgt_info.get('column', 0)}"
            source_to_target[key].append(int(tgt_line))

    # Map source lines to target lines
    mapping = {}
    for src_line, src_info in source_map.items():
        if "file" in src_info and "line" in src_info:
            key = f"{src_info['file']}:{src_info['line']}:{src_info.get('column', 0)}"
            if key in source_to_target:
                mapping[src_line] = sorted(source_to_target[key])

    return mapping


def create_bidirectional_mapping(
    source_map: Dict[str, Dict[str, Any]],
    target_map: Dict[str, Dict[str, Any]],
    source_type: str,
    target_type: str,
) -> None:
    """
    Create bidirectional mappings between two IR types and update their mapping dictionaries.

    This function creates mappings from source IR to target IR and vice versa, then
    updates both mapping dictionaries with the line references.

    Args:
        source_map: Dictionary mapping source IR line numbers to source locations
        target_map: Dictionary mapping target IR line numbers to source locations
        source_type: String identifier for the source IR type (e.g., 'ttir', 'ttgir', 'ptx')
        target_type: String identifier for the target IR type (e.g., 'ttir', 'ttgir', 'ptx')
    """
    # Create forward mapping (source to target)
    source_to_target = create_ir_mapping(source_map, target_map)

    # Add target line references to source mappings
    for source_line, target_lines in source_to_target.items():
        if source_line in source_map and target_lines:
            source_map[source_line][f"{target_type}_lines"] = target_lines

    # Create reverse mapping (target to source)
    target_to_source = create_ir_mapping(target_map, source_map)

    # Add source line references to target mappings
    for target_line, source_lines in target_to_source.items():
        if target_line in target_map:
            target_map[target_line][f"{source_type}_lines"] = source_lines

    logger.debug(f"Created {source_type} to {target_type} mappings (and reverse)")
