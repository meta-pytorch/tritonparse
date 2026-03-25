#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Parser for inductor's torch trace logs.

Extracts kernel_source_path -> CompileInfo mappings from inductor_output_code
events in torch trace log files. These mappings can be used to attribute Triton
kernels to their originating PyTorch compilation frame when pt_info is missing
(e.g., in multi-process Triton JIT compilation scenarios).
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from tritonparse._json_compat import JSONDecodeError, loads
from tritonparse.tp_logger import get_logger

logger = get_logger("TorchTraceParser")

# Pattern to extract kernel path from output_code payload
KERNEL_PATH_PATTERN = re.compile(r"^# kernel path: (.+)$", re.MULTILINE)


@dataclass
class CompileInfo:
    """Compilation frame info extracted from inductor's torch trace log."""

    frame_id: Optional[int] = None
    frame_compile_id: Optional[int] = None
    attempt: int = 0
    compiled_autograd_id: Optional[int] = None


def _extract_json_from_glog_line(line: str) -> Optional[str]:
    """
    Extract JSON string from a glog-formatted line.

    Glog format: V{timestamp} {pid} {filepath}:{lineno}] {json_metadata}

    Returns the JSON string portion, or None if the line doesn't match.
    """
    idx = line.find("] ")
    if idx == -1:
        return None
    return line[idx + 2 :]


def _parse_torch_trace_log(log_path: str) -> Dict[str, CompileInfo]:
    """
    Parse a single torch trace log file and extract kernel_source_path -> CompileInfo mappings.

    The torch trace log format is:
    - Each record starts with a glog prefix line containing JSON metadata
    - Subsequent lines starting with \\t are the payload (continuation of the record)

    For inductor_output_code events:
    - The JSON metadata contains frame_id, frame_compile_id, attempt, compiled_autograd_id
    - The payload contains the output_code.py content, which has '# kernel path: ...' comments

    Args:
        log_path: Path to the torch trace log file.

    Returns:
        Dict mapping kernel_source_path (absolute path) to CompileInfo.
    """
    mapping: Dict[str, CompileInfo] = {}

    current_compile_info: Optional[CompileInfo] = None
    current_payload_lines: List[str] = []
    in_output_code_event = False

    def _flush_current_event() -> None:
        """Process the accumulated payload for the current inductor_output_code event."""
        nonlocal current_compile_info, current_payload_lines, in_output_code_event
        if not in_output_code_event or current_compile_info is None:
            in_output_code_event = False
            current_compile_info = None
            current_payload_lines = []
            return

        payload_text = "\n".join(current_payload_lines)
        kernel_paths = KERNEL_PATH_PATTERN.findall(payload_text)
        for kp in kernel_paths:
            kp = kp.strip()
            if kp:
                mapping[kp] = current_compile_info
                logger.debug(
                    f"Mapped kernel path {kp} -> frame_id={current_compile_info.frame_id}, "
                    f"frame_compile_id={current_compile_info.frame_compile_id}"
                )

        in_output_code_event = False
        current_compile_info = None
        current_payload_lines = []

    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")

                # Continuation line (tab-indented payload)
                if line.startswith("\t"):
                    if in_output_code_event:
                        # Strip the leading tab
                        current_payload_lines.append(line[1:])
                    continue

                # New record — flush the previous event first
                _flush_current_event()

                # Try to parse this as a new glog record with JSON metadata
                json_str = _extract_json_from_glog_line(line)
                if not json_str:
                    continue

                try:
                    metadata = loads(json_str)
                except (JSONDecodeError, ValueError):
                    continue

                # Check if this is an inductor_output_code event
                if not isinstance(metadata, dict):
                    continue
                if "inductor_output_code" not in metadata:
                    continue

                # Extract compile info from the metadata
                in_output_code_event = True
                current_compile_info = CompileInfo(
                    frame_id=metadata.get("frame_id"),
                    frame_compile_id=metadata.get("frame_compile_id"),
                    attempt=metadata.get("attempt", 0),
                    compiled_autograd_id=metadata.get("compiled_autograd_id"),
                )

            # Flush the last event
            _flush_current_event()

    except OSError as e:
        logger.warning(f"Failed to read torch trace log {log_path}: {e}")

    return mapping


def parse_torch_trace_logs(
    log_paths: List[str],
) -> Dict[str, CompileInfo]:
    """
    Parse multiple torch trace log files and merge their mappings.

    Args:
        log_paths: List of paths to torch trace log files.

    Returns:
        Merged dict mapping kernel_source_path to CompileInfo.
    """
    merged: Dict[str, CompileInfo] = {}
    for path in log_paths:
        logger.info(f"Parsing torch trace log: {path}")
        file_mapping = _parse_torch_trace_log(path)
        logger.info(f"Extracted {len(file_mapping)} kernel path mappings from {path}")
        merged.update(file_mapping)
    return merged


# Prefix used by torch's structured trace logging
TORCH_TRACE_PREFIX = "dedicated_log_torch_trace_"


def discover_torch_trace_files(
    search_dir: str,
) -> Dict[Optional[int], List[str]]:
    """
    Discover torch trace log files in a directory, grouped by rank.

    Args:
        search_dir: Directory to search for torch trace log files.

    Returns:
        Dict mapping rank (int or None for no-rank files) to list of file paths.
    """
    rank_pattern = re.compile(r"rank_(\d+)_")
    result: Dict[Optional[int], List[str]] = {}

    try:
        for item in os.listdir(search_dir):
            if TORCH_TRACE_PREFIX not in item:
                continue
            if not item.endswith(".log"):
                continue
            full_path = os.path.join(search_dir, item)
            if not os.path.isfile(full_path):
                continue

            rank_match = rank_pattern.search(item)
            rank = int(rank_match.group(1)) if rank_match else None
            result.setdefault(rank, []).append(full_path)
    except OSError as e:
        logger.warning(
            f"Failed to scan directory {search_dir} for torch trace logs: {e}"
        )

    if result:
        total_files = sum(len(v) for v in result.values())
        logger.info(
            f"Discovered {total_files} torch trace log file(s) across "
            f"{len(result)} rank(s) in {search_dir}"
        )

    return result
