#  Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import re
from dataclasses import dataclass
from enum import Enum

from .sourcemap_utils import load_ir_contents


logger = logging.getLogger("IRAnalysis")


class BlockPingpongCategory(Enum):
    """BlockPingpong scheduling categories based on numWarps and PP clusters.

    Categories are derived from the MLIR test file:
    third-party/tp2/triton/2.0.x/triton/test/TritonGPU/amd/amd-block-pingpong.mlir
    """

    # numWarps=4, 1 PP cluster - uses setprio interleaving without cond_barrier
    PINGPONG_SMALL = "pingpong_small"

    # numWarps=8, 2 PP clusters - uses amdgpu.cond_barrier with 2 dot ops
    PINGPONG_MEDIUM = "pingpong_medium"

    # numWarps=8, 4 PP clusters - uses amdgpu.cond_barrier with 4 dot ops
    PINGPONG_LARGE = "pingpong_large"

    # No BlockPingpong detected
    NONE = "none"


@dataclass
class BlockPingpongResult:
    """Result of BlockPingpong detection and categorization."""

    category: BlockPingpongCategory
    detected: bool
    num_warps: int | None
    num_pp_clusters: int | None
    cond_barrier_count: int
    setprio_count: int
    dot_count: int


def detect_blockpingpong_category(ttgir_content: str) -> BlockPingpongResult:
    """
    Detect BlockPingpong scheduling category from TTGIR content.

    The detection is based on ordered pattern matching derived from the MLIR CHECK
    patterns in amd-block-pingpong.mlir.

    Categories:
    - pingpong_small (numWarps=4, 1 PP cluster):
      Pattern: local_load → setprio 1 → tt.load → sched.barrier → local_load →
               setprio 0 → tt.load → sched.barrier → setprio 1 → tt.dot → setprio 0

    - pingpong_medium (numWarps=8, 2 PP clusters):
      Pattern: amdgpu.cond_barrier + 2 dot operations with setprio interleaving

    - pingpong_large (numWarps=8, 4 PP clusters):
      Pattern: amdgpu.cond_barrier + 4 dot operations with setprio interleaving

    Args:
        ttgir_content: The TTGIR content as a string.

    Returns:
        BlockPingpongResult with detection details.
    """
    if not ttgir_content:
        return BlockPingpongResult(
            category=BlockPingpongCategory.NONE,
            detected=False,
            num_warps=None,
            num_pp_clusters=None,
            cond_barrier_count=0,
            setprio_count=0,
            dot_count=0,
        )

    lines = ttgir_content.split("\n")

    # Count key operations
    cond_barrier_count = sum(1 for line in lines if "cond_barrier" in line)
    setprio_count = sum(1 for line in lines if "rocdl.s.setprio" in line)
    sched_barrier_count = sum(1 for line in lines if "rocdl.sched.barrier" in line)
    dot_count = sum(1 for line in lines if "tt.dot " in line)

    # Extract numWarps from module attributes if present
    num_warps = None
    for line in lines:
        if '"triton_gpu.num-warps"' in line:
            match = re.search(r'"triton_gpu.num-warps"\s*=\s*(\d+)', line)
            if match:
                num_warps = int(match.group(1))
                break

    # Check for cond_barrier presence (indicates 2 or 4 PP clusters)
    has_cond_barrier = cond_barrier_count > 0

    # Check for pingpong_small pattern (numWarps=4, 1 PP cluster)
    # Pattern: local_load, setprio 1, tt.load, sched.barrier, local_load,
    #          setprio 0, tt.load, sched.barrier, setprio 1, tt.dot, setprio 0
    pingpong_small_pattern = [
        "local_load",
        "setprio",
        "tt.load",
        "rocdl.sched.barrier",
        "local_load",
        "setprio",
        "tt.load",
        "rocdl.sched.barrier",
        "setprio",
        "tt.dot",
        "setprio",
    ]

    # Determine category based on pattern analysis
    category = BlockPingpongCategory.NONE
    num_pp_clusters = None
    detected = False

    if has_cond_barrier:
        # 2 or 4 PP clusters (numWarps=8)
        detected = True
        if setprio_count > 0:
            # Count dots within scf.for loop to determine PP clusters
            # 2 PP clusters: 2 dots per iteration
            # 4 PP clusters: 4 dots per iteration
            in_loop = False
            loop_dot_count = 0
            for line in lines:
                if "scf.for" in line:
                    in_loop = True
                elif "scf.yield" in line:
                    in_loop = False
                elif in_loop and "tt.dot " in line:
                    loop_dot_count += 1

            if loop_dot_count >= 4:
                category = BlockPingpongCategory.PINGPONG_LARGE
                num_pp_clusters = 4
            else:
                category = BlockPingpongCategory.PINGPONG_MEDIUM
                num_pp_clusters = 2
    elif setprio_count > 0 and sched_barrier_count > 0:
        # Check for pingpong_small pattern (1 PP cluster, numWarps=4)
        # Verify ordered pattern matching
        if _check_ordered_pattern(ttgir_content, pingpong_small_pattern):
            detected = True
            category = BlockPingpongCategory.PINGPONG_SMALL
            num_pp_clusters = 1

    return BlockPingpongResult(
        category=category,
        detected=detected,
        num_warps=num_warps,
        num_pp_clusters=num_pp_clusters,
        cond_barrier_count=cond_barrier_count,
        setprio_count=setprio_count,
        dot_count=dot_count,
    )


def _check_ordered_pattern(content: str, pattern: list[str]) -> bool:
    """
    Check if the content contains the pattern elements in order.

    Args:
        content: The content to search.
        pattern: List of strings to find in order.

    Returns:
        True if all pattern elements are found in order.
    """
    pos = 0
    for element in pattern:
        idx = content.find(element, pos)
        if idx == -1:
            return False
        pos = idx + len(element)
    return True


def find_blockpingpong_scheduling(
    ttgir_key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
) -> dict[str, any]:
    """
    Detect and categorize BlockPingpong scheduling from TTGIR content.

    Args:
        ttgir_key: Key for the TTGIR file.
        file_content: Dictionary mapping file keys to content.
        file_path: Dictionary mapping file keys to file paths.

    Returns:
        A dictionary containing:
        - "detected": Whether BlockPingpong was detected
        - "category": The BlockPingpong category name
        - "num_warps": Number of warps (4 or 8)
        - "num_pp_clusters": Number of PP clusters (1, 2, or 4)
        - "cond_barrier_count": Number of cond_barrier operations
        - "setprio_count": Number of setprio operations
        - "dot_count": Number of dot operations
    """
    ttgir_content = load_ir_contents(ttgir_key, file_content, file_path)

    if not ttgir_content:
        return {
            "detected": False,
            "category": BlockPingpongCategory.NONE.value,
            "num_warps": None,
            "num_pp_clusters": None,
            "cond_barrier_count": 0,
            "setprio_count": 0,
            "dot_count": 0,
        }

    result = detect_blockpingpong_category(ttgir_content)

    return {
        "detected": result.detected,
        "category": result.category.value,
        "num_warps": result.num_warps,
        "num_pp_clusters": result.num_pp_clusters,
        "cond_barrier_count": result.cond_barrier_count,
        "setprio_count": result.setprio_count,
        "dot_count": result.dot_count,
    }


def process_amd_bufferop(ir_content: str, io_keys: list[str]) -> dict[str, int]:
    def make_key(prefix: str) -> str:
        return f"{prefix}_count"

    io_keys = [(make_key(prefix), prefix) for prefix in io_keys]
    output: dict[str, int] = {}
    for dict_key, _ in io_keys:
        output[dict_key] = 0
    if ir_content:
        for line in ir_content.split("\n"):
            for dict_key, code_key in io_keys:
                if code_key in line:
                    output[dict_key] += 1
    return output


def process_amd_ttgir_bufferops(
    key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
) -> dict[str, int]:
    ir_content = load_ir_contents(key, file_content, file_path)
    # TODO: Add atomics
    io_keys = ["tt.load", "tt.store", "amdgpu.buffer_load", "amdgpu.buffer_store"]
    return process_amd_bufferop(ir_content, io_keys)


def process_amd_gcn_bufferops(
    key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
) -> dict[str, int]:
    ir_content = load_ir_contents(key, file_content, file_path)
    # TODO: Add atomics
    io_keys = ["global_load", "global_store", "buffer_load", "buffer_store"]
    return process_amd_bufferop(ir_content, io_keys)


def find_loop_bounds(ir_content: str) -> list[tuple[int, int]]:
    """
    Find the bounds of all scf.for loops in the IR content.
    These are the only candidates for Software Pipelining (SWP).

    A loop starts with 'scf.for' and ends when its closing brace '}' is found.
    Brace counts are tracked to determine when each loop closes.

    Args:
        ir_content: The IR content as a string.

    Returns:
        A list of tuples (start_line, end_line) for each scf.for loop found.
        Line numbers are 0-indexed.
    """
    if not ir_content:
        return []

    loop_bounds: list[tuple[int, int]] = []
    lines = ir_content.split("\n")

    # Stack to track loop starts and their brace counts
    # Each entry is (start_line, brace_count_at_start)
    loop_stack: list[tuple[int, int]] = []
    current_brace_count = 0

    for line_idx, line in enumerate(lines):
        # Check if this line starts a new scf.for loop
        if "scf.for" in line:
            loop_stack.append((line_idx, current_brace_count))

        # Count braces on this line
        for char in line:
            if char == "{":
                current_brace_count += 1
            elif char == "}":
                current_brace_count -= 1

        # Check if we've closed any loops
        while loop_stack and current_brace_count <= loop_stack[-1][1]:
            start_line, _start_brace_count = loop_stack.pop()
            # The loop ends at this line
            loop_bounds.append((start_line, line_idx))

    return loop_bounds


def find_inner_loop_bounds(ir_content: str) -> list[tuple[int, int]]:
    """
    Find the bounds of inner scf.for loops (loops without nested loops inside).

    Inner loops are the primary candidates for Software Pipelining (SWP) as they
    represent the innermost computation that can be optimized.

    Args:
        ir_content: The IR content as a string.

    Returns:
        A list of tuples (start_line, end_line) for each inner scf.for loop found.
        Line numbers are 0-indexed.
    """
    all_loops = find_loop_bounds(ir_content)

    if not all_loops:
        return []

    # Filter to keep only inner loops (loops that don't contain other loops)
    inner_loops: list[tuple[int, int]] = []

    for i, (start_i, end_i) in enumerate(all_loops):
        # Check if any other loop is nested inside this loop
        has_nested_loop = False
        for j, (start_j, end_j) in enumerate(all_loops):
            if i != j:
                # Check if loop j is nested inside loop i
                if start_i < start_j and end_j < end_i:
                    has_nested_loop = True
                    break

        # If no nested loops found, this is an inner loop
        if not has_nested_loop:
            inner_loops.append((start_i, end_i))

    return inner_loops


def find_loop_pipelining(
    ttir_content: str,
    ttgir_content: str,
    ttir_loop_start: int,
    ttir_loop_end: int,
    loop_index: int,
    ttir_to_ttgir_mapping: dict[str, dict],
    ttgir_to_source_mapping: dict[str, dict],
    python_source_content: str | None,
    python_source_start_line: int,
) -> dict[str, list[str]]:
    """
    Find pipelining information for a specific loop by identifying tt.load and tt.dot operations
    in TTIR and mapping them to their corresponding operations in the original Python source code.

    For each tt.load or tt.dot operation found in the TTIR loop, this function uses source
    mappings to find the corresponding operations in TTGIR, then maps them back to the original
    Python source code. Operations are categorized into three sections:
    - prologue: Operations that appear before the loop body
    - loop_body: Operations that appear within the loop body
    - epilogue: Operations that appear after the loop body

    Operations are merged together (both loads and dots) and sorted in program order
    within each section.

    Args:
        ttir_content: The TTIR content as a string.
        ttgir_content: The TTGIR content as a string.
        ttir_loop_start: The starting line number of the loop in TTIR (0-indexed).
        ttir_loop_end: The ending line number of the loop in TTIR (0-indexed).
        ttir_to_ttgir_mapping: Source mapping from TTIR lines to TTGIR lines.
        ttgir_to_source_mapping: Source mapping from TTGIR lines to original Python source.
        python_source_content: The original Python source code content.

    Returns:
        A dictionary containing:
        - "prologue": List of Python source line strings in program order
        - "loop_body": List of Python source line strings in program order
        - "epilogue": List of Python source line strings in program order
    """
    if not ttir_content or not ttgir_content:
        return {
            "prologue": [],
            "loop_body": [],
            "epilogue": [],
        }

    ttir_lines = ttir_content.split("\n")
    ttgir_lines = ttgir_content.split("\n")
    python_lines = python_source_content.split("\n") if python_source_content else []

    def apply_trailing_space(op: str) -> str:
        """
        Add a trailing space to all ops to avoid false positives like
        warp_group_dot and warp_group_dot_wait.
        """
        return op + " "

    # Step 1: Find tt.load and tt.dot operations in TTIR loop
    ttir_pipeline_lines: list[int] = []
    pipeline_tt_ops = ["tt.load", "tt.dot"]
    pipeline_tt_ops = [apply_trailing_space(op) for op in pipeline_tt_ops]
    pipeline_ttgir_ops = [
        "tt.load",
        "tt.dot",
        "async_copy_global_to_local",
        "warp_group_dot",
    ]
    pipeline_ttgir_ops = [apply_trailing_space(op) for op in pipeline_ttgir_ops]
    for line_idx in range(ttir_loop_start, min(ttir_loop_end + 1, len(ttir_lines))):
        line = ttir_lines[line_idx]
        for op in pipeline_tt_ops:
            if op in line:
                ttir_pipeline_lines.append(line_idx)
                break

    # Step 2: Find the corresponding loop in TTGIR using source mappings
    # Map the TTIR loop bounds to TTGIR using source mappings
    ttgir_inner_loops = find_inner_loop_bounds(ttgir_content)

    if not ttgir_inner_loops:
        # No loop found in TTGIR, return empty results
        return {
            "prologue": [],
            "loop_body": [],
            "epilogue": [],
        }

    # Use the first inner loop as the reference
    # TODO: Implement more sophisticated mapping logic to match TTIR loops to TTGIR loops
    ttgir_loop_start, ttgir_loop_end = ttgir_inner_loops[loop_index]

    # Step 3: Map TTIR operations to TTGIR operations using source mappings
    # and categorize them by their position relative to the TTGIR loop
    # Store as (line_number, source_line) to maintain order before extracting just the source
    prologue_ops: list[tuple[int, str]] = []
    loop_body_ops: list[tuple[int, str]] = []
    epilogue_ops: list[tuple[int, str]] = []

    for ttir_line in ttir_pipeline_lines:
        # Convert 0-indexed line to 1-indexed string key for mapping lookup
        ttir_line_key = str(ttir_line + 1)

        # Get the corresponding TTGIR lines from the source mapping
        if ttir_line_key in ttir_to_ttgir_mapping:
            ttgir_lines_list = ttir_to_ttgir_mapping[ttir_line_key].get(
                "ttgir_lines", []
            )

            # For each mapped TTGIR line, categorize it
            for ttgir_line in ttgir_lines_list:
                # Convert back to 0-indexed
                ttgir_line_idx = ttgir_line - 1

                # Get the actual TTGIR line content to check if it's relevant
                if ttgir_line_idx < len(ttgir_lines):
                    ttgir_source_line = ttgir_lines[ttgir_line_idx].strip()

                    # Only keep mappings to the "compute" op.
                    if any(op in ttgir_source_line for op in pipeline_ttgir_ops):
                        # Map TTGIR line back to Python source
                        ttgir_line_key = str(ttgir_line)
                        python_source_line = ttgir_source_line  # Default to TTGIR line

                        if ttgir_line_key in ttgir_to_source_mapping:
                            source_info = ttgir_to_source_mapping[ttgir_line_key]
                            python_line_num = source_info.get("line")

                            if python_line_num and python_lines:
                                # Account for the offset: the Python source may not start at line 1
                                # python_line_num is the absolute line number in the original file
                                # python_source_start_line is where the extracted code starts
                                # So we need to subtract the offset to get the index in our python_lines array
                                python_line_idx = (
                                    python_line_num - python_source_start_line
                                )
                                if 0 <= python_line_idx < len(python_lines):
                                    python_source_line = python_lines[
                                        python_line_idx
                                    ].strip()

                        if ttgir_line_idx < ttgir_loop_start:
                            prologue_ops.append((ttgir_line_idx, python_source_line))
                        elif ttgir_loop_start <= ttgir_line_idx <= ttgir_loop_end:
                            loop_body_ops.append((ttgir_line_idx, python_source_line))
                        else:
                            epilogue_ops.append((ttgir_line_idx, python_source_line))

    # Step 4: Sort each section by line number to maintain program order
    prologue_ops.sort(key=lambda x: x[0])
    loop_body_ops.sort(key=lambda x: x[0])
    epilogue_ops.sort(key=lambda x: x[0])

    # Extract just the source lines (without line numbers)
    prologue_lines = [line for _, line in prologue_ops]
    loop_body_lines = [line for _, line in loop_body_ops]
    epilogue_lines = [line for _, line in epilogue_ops]

    # Log the pipelining results
    logger.debug(
        f"Loop pipelining results (TTIR lines {ttir_loop_start}-{ttir_loop_end}):"
    )
    logger.debug(f"  Prologue ({len(prologue_lines)} ops):")
    for line in prologue_lines:
        logger.debug(f"    {line}")
    logger.debug(f"  Loop Body ({len(loop_body_lines)} ops):")
    for line in loop_body_lines:
        logger.debug(f"    {line}")
    logger.debug(f"  Epilogue ({len(epilogue_lines)} ops):")
    for line in epilogue_lines:
        logger.debug(f"    {line}")

    return {
        "prologue": prologue_lines,
        "loop_body": loop_body_lines,
        "epilogue": epilogue_lines,
    }


def generate_loop_schedule(
    ttir_key: str,
    ttgir_key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
    source_mappings: dict[str, dict],
    python_source_content: str | None,
    python_source_start_line: int,
) -> list[dict]:
    """
    Generate loop schedule information by finding inner scf.for loops in TTIR
    and analyzing their pipelining potential using source mappings.

    Only inner loops (loops without nested loops) are considered as they are
    the primary candidates for Software Pipelining (SWP).

    Args:
        ttir_key: Key for the TTIR file.
        ttgir_key: Key for the TTGIR file.
        file_content: Dictionary mapping file keys to content.
        file_path: Dictionary mapping file keys to file paths.
        source_mappings: Dictionary containing source mappings between IR stages.
        python_source_content: The original Python source code content.
        python_source_start_line: The starting line number of the Python source in the original file.

    Returns:
        A list of dictionaries, each containing:
        - "loop_bounds": Tuple of (start_line, end_line) for the loop in TTIR
        - "pipelining": Dictionary with Python source lines for operations
    """
    ttir_content = load_ir_contents(ttir_key, file_content, file_path)
    ttgir_content = load_ir_contents(ttgir_key, file_content, file_path)

    # Get the TTIR to TTGIR mapping and TTGIR to source mapping
    ttir_to_ttgir_mapping = source_mappings.get("ttir", {})
    ttgir_to_source_mapping = source_mappings.get("ttgir", {})

    # Find only inner loops (loops without nested loops inside)
    inner_loop_bounds = find_inner_loop_bounds(ttir_content)
    # TODO: Fix loop mapping with multiple loops.
    inner_loop_bounds = inner_loop_bounds[:1]

    # For each inner loop, find pipelining information
    loop_schedules = []
    for i, (loop_start, loop_end) in enumerate(inner_loop_bounds):
        pipelining_info = find_loop_pipelining(
            ttir_content,
            ttgir_content,
            loop_start,
            loop_end,
            i,
            ttir_to_ttgir_mapping,
            ttgir_to_source_mapping,
            python_source_content,
            python_source_start_line,
        )
        loop_schedules.append(pipelining_info)

    return loop_schedules


def _generate_ir_analysis(entry: str):
    payload = entry.setdefault("payload", {})
    file_content = payload.get("file_content", {})
    file_path = payload.get("file_path", {})
    source_mappings = payload.get("source_mappings", {})

    # Find the IR file keys
    ttir_key = next((k for k in file_content if k.endswith(".ttir")), None)
    ttgir_key = next((k for k in file_content if k.endswith(".ttgir")), None)
    amdgcn_key = next((k for k in file_content if k.endswith(".amdgcn")), None)
    # Skip if no IR files found
    if not (ttir_key or ttgir_key or amdgcn_key):
        logger.debug("No IR found")
        return {}
    ir_analysis = {}
    if amdgcn_key and ttgir_key:
        # Add BufferOps information
        ttgir_bufferops_info = process_amd_ttgir_bufferops(
            ttgir_key, file_content, file_path
        )
        gcn_bufferops_info = process_amd_gcn_bufferops(
            amdgcn_key, file_content, file_path
        )
        io_counts = {}
        # NDJSON format requires a newline at the end of each line
        if ttgir_bufferops_info:
            io_counts["amd_ttgir_bufferops_count"] = ttgir_bufferops_info
        if gcn_bufferops_info:
            io_counts["amd_gcn_bufferops_count"] = gcn_bufferops_info
        if io_counts:
            ir_analysis["io_counts"] = io_counts
    if ttir_key and ttgir_key:
        # Get Python source content and start line if available
        python_source_content = None
        python_source_start_line = 1  # Default to 1 if not available
        python_source_info = payload.get("python_source")
        if python_source_info:
            python_source_content = python_source_info.get("code")
            python_source_start_line = python_source_info.get("start_line", 1)

        # Get source mappings
        ttir_to_ttgir_mapping = source_mappings.get("ttir", {})
        ttgir_to_source_mapping = source_mappings.get("ttgir", {})

        # Add loop schedule information
        loop_schedule = generate_loop_schedule(
            ttir_key,
            ttgir_key,
            file_content,
            file_path,
            source_mappings,
            python_source_content,
            python_source_start_line,
        )
        if loop_schedule:
            ir_analysis["loop_schedules"] = loop_schedule

    # Add BlockPingpong scheduling detection for AMD (only if TTGIR is available)
    if ttgir_key:
        blockpingpong_info = find_blockpingpong_scheduling(
            ttgir_key, file_content, file_path
        )
        if blockpingpong_info:
            ir_analysis["blockpingpong"] = blockpingpong_info

    return ir_analysis
