#  Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tritonparse.tp_logger import logger

# Sentinel object to mark arguments that should be skipped during processing
_SKIP = object()


@dataclass
class KernelInfo:
    """Information about a Triton kernel extracted from compilation events."""

    file_path: str
    function_name: str
    source_code: str
    call_stack: List[Dict[str, Any]]


@dataclass
class ContextBundle:
    """Bundle of all context information needed to reproduce a kernel launch."""

    kernel_info: KernelInfo
    compile: Dict[str, Any]
    launch: Dict[str, Any]
    args: Dict[str, Any]
    tensor_args: Dict[str, Any]
    raw_launch_event: Dict[str, Any]
    raw_comp_event: Dict[str, Any]


def get_launch_and_compilation_events(
    events: List[Dict[str, Any]], line_index: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract launch and compilation events from the event list.

    Args:
        events: List of parsed event dictionaries.
        line_index: 0-based index of the launch event to process.

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

    comp_event = None
    for event in events:
        if (
            event["event_type"] == "compilation"
            and event.get("payload", {}).get("metadata", {}).get("hash") == comp_hash
        ):
            comp_event = event
            break
    if not comp_event:
        raise RuntimeError(f"Could not find compilation event for hash {comp_hash}.")
    return launch_event, comp_event


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


def _decode_arg(raw: Any) -> Any:
    """
    Decode a raw argument value from event data.

    Args:
        raw: Raw argument value from event data.

    Returns:
        Decoded argument value, or _SKIP sentinel for tensors.
    """
    if not isinstance(raw, dict):
        return raw
    t = raw.get("type")
    if t == "tensor":
        return _SKIP
    if t == "NoneType":
        return None
    return raw.get("value", raw.get("repr"))


def _pack_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pack argument values into a standardized format.

    Args:
        args: Dictionary of argument names to values.

    Returns:
        Dictionary with packed argument information including type and metadata.
    """
    packed = {}
    for k, v in args.items():
        t = v.get("type") if isinstance(v, dict) else None
        if t == "tensor":
            packed[k] = {
                "type": "tensor",
                "shape": v.get("shape") if isinstance(v, dict) else None,
                "dtype": v.get("dtype") if isinstance(v, dict) else None,
                "device": v.get("device") if isinstance(v, dict) else None,
                "stride": v.get("stride") if isinstance(v, dict) else None,
                "is_contiguous": (
                    v.get("is_contiguous") if isinstance(v, dict) else None
                ),
                "numel": v.get("numel") if isinstance(v, dict) else None,
            }
        else:
            # scalar / NoneType etc
            if isinstance(v, dict):
                packed[k] = {
                    "type": v.get("type"),
                    "value": v.get("value", v.get("repr")),
                }
            else:
                packed[k] = {
                    "type": None,
                    "value": v,
                }
    return packed


def build_context_bundle(
    events: List[Dict[str, Any]], line_index: Optional[int] = None
):
    """
    Build a complete context bundle from events and line index.

    Args:
        events: List of parsed event dictionaries.
        line_index: 0-based index of the launch event to process.

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

    # kwargs: include constexpr + explicit scalars used for launch (skip tensor args)
    kwargs = {}
    for k, v in extracted_args.items():
        val = _decode_arg(v)
        if val is _SKIP:
            continue
        kwargs[k] = val

    # tensor args: only tensors
    raw_tensor_args = {
        k: v
        for k, v in extracted_args.items()
        if isinstance(v, dict) and v.get("type") == "tensor"
    }

    primitive_args = _pack_args(extracted_args)
    tensor_args = _pack_args(raw_tensor_args)
    launch_block = {
        "grid": grid,
        "kwargs": kwargs,
    }

    return ContextBundle(
        kernel_info,
        compile_block,
        launch_block,
        primitive_args,
        tensor_args,
        launch_event,
        comp_event,
    )
