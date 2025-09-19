from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tritonparse.tp_logger import logger


@dataclass
class KernelInfo:
    """Information about a Triton kernel extracted from compilation events."""

    file_path: str
    function_name: str
    source_code: str
    call_stack: List[Dict[str, Any]]


def get_launch_and_compilation_events(
    events: List[Dict[str, Any]], line_index: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract launch and compilation events from the event list.

    Args:
        events: List of parsed event dictionaries.
        line_index: Index of the launch event to process.

    Returns:
        Tuple of (launch_event, compilation_event).

    Raises:
        ValueError: If the event at line_index is not a launch event.
        RuntimeError: If compilation event cannot be found or is ambiguous.
    """
    if line_index is None or line_index >= len(events):
        raise ValueError(f"Invalid line_index: {line_index}")

    launch_event = events[line_index]
    if launch_event["event_type"] != "launch":
        raise ValueError(f"Event at index {line_index} is not a launch event")

    comp_meta = launch_event.get("compilation_metadata", {})
    comp_hash = comp_meta.get("hash")
    if not comp_hash:
        raise RuntimeError("Could not find compilation hash in launch event.")

    comp_event = [
        event
        for event in events
        if event["event_type"] == "compilation" and event.get("hash") == comp_hash
    ]
    if not comp_event:
        raise RuntimeError(f"Could not find compilation event for hash {comp_hash}.")
    if len(comp_event) != 1:
        raise RuntimeError(
            f"Expected 1 compilation event for hash {comp_hash}, got {len(comp_event)}"
        )

    return launch_event, comp_event[0]


def get_kernel_info(comp_event: Dict[str, Any]) -> KernelInfo:
    """
    Extract kernel information from a compilation event.

    Args:
        comp_event: Compilation event dictionary containing kernel metadata.

    Returns:
        KernelInfo object with extracted kernel details.

    Raises:
        RuntimeError: If file path or function name cannot be resolved.
    """
    payload = comp_event.get("payload") or {}
    py_source = payload.get("python_source") or {}
    code = py_source.get("code", "")

    # Extract file path and function name
    file_path = py_source.get("file_path")
    # The function name is in the compilation metadata payload
    func_name = (comp_event.get("payload", {}).get("metadata") or {}).get("name")

    # Find '@triton.jit' decorator and slice the string from there
    jit_marker = "@triton.jit"
    jit_pos = code.find(jit_marker)
    if jit_pos != -1:
        code = code[jit_pos:]
        logger.debug("Extracted kernel source starting from '@triton.jit'.")

    if not file_path or not func_name:
        raise RuntimeError(
            "Could not resolve kernel file path or function name from compilation event."
            " The import-based strategy cannot proceed."
        )
    return KernelInfo(file_path, func_name, code, comp_event.get("stack", []))


def build_context_bundle(
    events: List[Dict[str, Any]], line_index: Optional[int] = None
):
    """
    Build a complete context bundle from events and line index.

    Args:
        events: List of parsed event dictionaries.
        line_index: Index of the launch event to process.

    Returns:
        ContextBundle containing all information needed to reproduce the kernel launch.

    Raises:
        ValueError: If line_index is invalid or event is not a launch event.
        RuntimeError: If compilation event cannot be found.
    """
    launch_event, comp_event = get_launch_and_compilation_events(events, line_index)
    kernel_info = get_kernel_info(comp_event)
    grid = launch_event.get("grid")
    extracted_args = launch_event.get("extracted_args", {})
    comp_meta = launch_event.get("compilation_metadata", {})

    # Compile metadata subset we care about
    compile_block = {
        "num_warps": comp_meta.get("num_warps"),
        "num_stages": comp_meta.get("num_stages"),
        "arch": comp_meta.get("arch"),
        "backend": comp_meta.get("backend_name") or comp_meta.get("backend"),
        "triton_version": comp_meta.get("triton_version"),
        "hash": comp_meta.get("hash"),
    }
