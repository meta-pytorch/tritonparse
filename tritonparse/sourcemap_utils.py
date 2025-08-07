import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


def compute_launch_event_hash(launch_event: Dict[str, Any]) -> str:
    """
    Compute a stable hash for a launch event.

    Args:
        launch_event: The launch event dictionary

    Returns:
        A SHA-256 hash string (16 characters) of the launch event
    """
    # Create a copy without volatile fields like timestamp
    stable_event = launch_event.copy()
    # # Remove fields that change between identical launches
    # for field in ["timestamp", "pid"]:
    #     stable_event.pop(field, None)

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


def _is_autotune_benchmark_launch(stack: List[Dict[str, Any]]) -> bool:
    """Checks if a stack trace corresponds to an autotune benchmark launch."""
    for frame in stack:
        filename = frame.get("filename", "")
        func_name = frame.get("name")
        if "triton/runtime/autotuner.py" in filename and func_name == "_bench":
            return True
    return False


def get_autotune_session_id(stack_trace: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
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
    try:
        # Find the first frame corresponding to the autotuner's method
        autotuner_boundary = -1
        for i, frame in enumerate(stack_trace):
            filename = frame.get("filename", "")
            func_name = frame.get("name")
            if "triton/runtime/autotuner.py" in filename and func_name in ["run", "_bench"]:
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
        
    except (IndexError, KeyError):
        # Fallback if stack trace structure is unexpected
        return None, None
