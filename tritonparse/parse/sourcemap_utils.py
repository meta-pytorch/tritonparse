#  Copyright (c) Meta Platforms, Inc. and affiliates.

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from tritonparse.tp_logger import get_logger

logger = get_logger("SourceMapping")


def _remove_keys_recursive(obj: Any, keys_to_remove: List[str]) -> Any:
    """
    Recursively remove any keys present in keys_to_remove from a nested dict/list structure.

    Args:
        obj: Arbitrary JSON-like object to process (dict, list, or primitive types).
        keys_to_remove: List of dictionary keys to remove wherever they appear.

    Returns:
        A new object with matching keys removed from any dictionaries encountered.
    """
    if isinstance(obj, dict):
        return {
            key: _remove_keys_recursive(value, keys_to_remove)
            for key, value in obj.items()
            if key not in keys_to_remove
        }
    if isinstance(obj, list):
        return [_remove_keys_recursive(item, keys_to_remove) for item in obj]
    return obj


def compute_launch_event_hash(launch_event: Dict[str, Any]) -> str:
    """
    Compute a stable hash for a launch event.

    Args:
        launch_event: The launch event dictionary

    Returns:
        A SHA-256 hash string (16 characters) of the launch event
    """
    # Create a deep-cleaned copy excluding volatile/non-semantic fields
    # Excluded keys:
    # - "timestamp", "pid": runtime-variant
    # - "launch_group_hash", "occurrence_id": injected during processing
    # - "function": currently unstable/None for first kernel in samples
    excluded = ["timestamp", "pid", "launch_group_hash", "occurrence_id", "function"]
    stable_event = _remove_keys_recursive(launch_event, excluded)

    # Sort keys for stable serialization
    stable_json = json.dumps(stable_event, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_json.encode()).hexdigest()[:16]


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


def _is_autotune_benchmark_launch(stack: List[Dict[str, Any]]) -> bool:
    """
    Check if a stack trace corresponds to an autotune benchmark launch.

    Args:
        stack: A list of stack frame dictionaries, each containing "filename"
            and "name" keys.

    Returns:
        True if any frame corresponds to the Triton autotuner benchmark helper
        (_bench in triton/runtime/autotuner.py), otherwise False.
    """
    for frame in stack:
        filename = frame.get("filename", "")
        func_name = frame.get("name")
        if "triton/runtime/autotuner.py" in filename and func_name == "_bench":
            return True
    return False


def get_autotune_session_id(
    stack_trace: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    """
    Generate a session ID for an autotune instance by finding the user-level
    call site and computing a hash of the complete user call stack.

    Args:
        stack_trace: The stack trace from a compilation or launch event.

    Returns:
        A tuple of (session_id, user_stack) where:
        - session_id: A unique hash-based session ID string if the event is part
          of an autotune process, otherwise None.
        - user_stack: The complete user call stack before autotuner.py frames,
          otherwise None.
    """
    # Find the first frame corresponding to the autotuner's method
    autotuner_boundary = -1
    for i, frame in enumerate(stack_trace):
        filename = frame.get("filename", "")
        func_name = frame.get("name")
        if "triton/runtime/autotuner.py" in filename and func_name in [
            "run",
            "_bench",
        ]:
            autotuner_boundary = i
            break

    if autotuner_boundary == -1:
        return None, None

    # Extract user stack (everything before autotuner.py)
    user_stack = stack_trace[:autotuner_boundary]

    if not user_stack:
        return None, None

    # Create a stable string representation of the user stack
    stack_parts = []
    for frame in user_stack:
        filename = frame.get("filename", "")
        line = frame.get("line", 0)
        name = frame.get("name", "")
        stack_parts.append(f"{filename}:{line}:{name}")

    stack_key = "|".join(stack_parts)

    # Generate a hash-based session ID
    session_id = hashlib.sha256(stack_key.encode()).hexdigest()[:16]

    return session_id, user_stack
