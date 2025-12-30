#  Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import Any, Dict, List

logger = logging.getLogger("SourceMapping")


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a given filename or return the filename itself if it has no extension.

    Args:
        filename (str): The filename or file extension.

    Returns:
        str: The file extension or the filename itself if no extension is present.
    """
    # Split the filename by '.' and return the last part if it exists
    parts = filename.split(".")
    return parts[-1] if len(parts) > 1 else filename


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Unflattens a dictionary with delimited keys.
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ref = result
        for part in parts[:-1]:
            if part not in d_ref:
                d_ref[part] = {}
            d_ref = d_ref[part]
        d_ref[parts[-1]] = value
    return result


def _to_ranges(indices: List[int]) -> List[Dict[str, int]]:
    """
    Converts a sorted list of indices into a list of continuous ranges.
    e.g., [0, 1, 2, 5, 6, 8] -> [{'start': 0, 'end': 2}, {'start': 5, 'end': 6}, {'start': 8, 'end': 8}]
    """
    if not indices:
        return []

    indices = sorted(indices)
    ranges = []
    start = indices[0]
    end = indices[0]

    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            ranges.append({"start": start, "end": end})
            start = end = indices[i]

    ranges.append({"start": start, "end": end})
    return ranges


def load_ir_contents(
    key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
):
    if not key:
        return {}
    logger.debug(f"Processing {key}")
    ir_content = file_content.get(key, None)
    if not ir_content:
        ir_file_path = file_path.get(key, None)
        if not ir_file_path:
            logger.warning(f"No content found for {key}")
            return {}
        with open(ir_file_path, "r") as f:
            ir_content = f.read()
    return ir_content
