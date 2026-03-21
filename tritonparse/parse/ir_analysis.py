#  Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any

from tritonparse.tp_logger import get_logger

from .sourcemap_utils import load_ir_contents

logger = get_logger("IRAnalysis")

# Default timeout (in seconds) for FileCheck subprocess calls
FILECHECK_TIMEOUT_SECONDS = 30


# =============================================================================
# FILECHECK BINARY DETECTION
# =============================================================================
def _find_filecheck_binary() -> str | None:
    """Find the FileCheck binary, preferring Triton's bundled version."""
    # Try Triton's bundled FileCheck first
    try:
        import triton

        triton_dir = os.path.dirname(triton.__file__)
        bundled_path = os.path.join(triton_dir, "backends", "amd", "bin", "FileCheck")
        if os.path.isfile(bundled_path) and os.access(bundled_path, os.X_OK):
            return bundled_path
    except ImportError as e:
        logger.debug(f"Triton import failed: {e}")

    # Check environment variable
    env_path = os.environ.get("FILECHECK_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    # Try system PATH
    system_path = shutil.which("FileCheck")
    if system_path:
        return system_path

    return None


FILECHECK_BINARY_PATH = _find_filecheck_binary()
logger.debug(f"FILECHECK_BINARY_PATH is {FILECHECK_BINARY_PATH}")
FILECHECK_AVAILABLE = FILECHECK_BINARY_PATH is not None

if not FILECHECK_AVAILABLE:
    logger.warning(
        "FileCheck binary not found. Procedure detection will not work. "
        "Set FILECHECK_PATH env var or install Triton (includes FileCheck)."
    )


# =============================================================================
# PROCEDURE CHECK DATACLASSES
# =============================================================================
@dataclass
class ProcedureCheckResult:
    """Result of a single procedure check using FileCheck patterns."""

    procedure_name: str
    detected: bool = False
    check_pattern: str = ""
    match_details: list[str] = field(default_factory=list)
    error_message: str | None = None
    message: str = ""
    module_attributes: str | None = None
    # Dynamic attributes extracted based on display_attributes config from JSON
    attributes: dict[str, Any] = field(default_factory=dict)
    # Display attributes configuration from JSON
    display_attributes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProcedureCheckConfig:
    """Configuration for a procedure check."""

    name: str
    patterns: str  # Raw FileCheck pattern string (e.g., "CHECK: foo\nCHECK-NOT: bar")
    description: str = ""


# =============================================================================
# FILECHECK PATTERN MATCHING
# =============================================================================
def run_filecheck(
    content: str,
    config: ProcedureCheckConfig,
    display_attributes: list[dict[str, Any]] | None = None,
) -> ProcedureCheckResult:
    """
    Run FileCheck patterns on the given content to detect a procedure.

    Uses LLVM FileCheck binary directly with the raw pattern string.

    Args:
        content: IR content to check.
        config: Procedure check configuration with patterns.
        display_attributes: List of display attribute configs from JSON.
            Controls which metadata attributes are extracted from the IR.
    """
    if not content:
        return ProcedureCheckResult(
            procedure_name=config.name,
            detected=False,
            check_pattern=config.patterns,
            error_message="No content provided",
            display_attributes=display_attributes or [],
        )

    # Build match_details from raw pattern string for UI display
    match_details: list[str] = []
    for line in config.patterns.strip().split("\n"):
        line = line.strip()
        if line and line.startswith("CHECK"):
            match_details.append(line)

    def _build_result(
        detected: bool,
        check_pattern: str,
        error_message: str | None = None,
        message: str = "",
    ) -> ProcedureCheckResult:
        """Helper to build a ProcedureCheckResult with metadata extraction."""
        metadata = extract_ir_metadata(content, display_attributes) if content else {}
        # Separate module_attributes from dynamic attributes
        module_attributes = metadata.pop("module_attributes", None)
        return ProcedureCheckResult(
            procedure_name=config.name,
            detected=detected,
            check_pattern=check_pattern,
            match_details=match_details,
            error_message=error_message,
            message=message,
            module_attributes=module_attributes,
            attributes=metadata,
            display_attributes=display_attributes or [],
        )

    if FILECHECK_AVAILABLE:
        try:
            # Add semicolon prefix to each CHECK line for FileCheck format
            check_lines = []
            for line in config.patterns.strip().split("\n"):
                line = line.strip()
                if line and line.startswith("CHECK"):
                    check_lines.append(f"; {line}")
            check_text = "\n".join(check_lines)

            with tempfile.TemporaryDirectory() as tempdir:
                input_file = os.path.join(tempdir, "input.mlir")
                with open(input_file, "w") as f:
                    f.write(content)

                check_file = os.path.join(tempdir, "checks.txt")
                with open(check_file, "w") as f:
                    f.write(check_text)

                result = subprocess.run(
                    [
                        FILECHECK_BINARY_PATH,
                        check_file,
                        "--input-file",
                        input_file,
                        "--dump-input-context=50",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=FILECHECK_TIMEOUT_SECONDS,
                )

                if result.returncode == 0:
                    return _build_result(
                        detected=True,
                        check_pattern=check_text,
                    )
                else:
                    error_output = result.stdout + result.stderr
                    return _build_result(
                        detected=False,
                        check_pattern=check_text,
                        error_message=(
                            f"{error_output[:500]}... [truncated]"
                            if error_output
                            else None
                        ),
                    )

        except subprocess.TimeoutExpired:
            return _build_result(
                detected=False,
                check_pattern=config.patterns,
                error_message="FileCheck timed out",
            )
        except Exception as e:
            logger.warning(f"FileCheck error for {config.name}: {e}")
            return _build_result(
                detected=False,
                check_pattern=config.patterns,
                error_message=f"FileCheck error: {e}",
            )
    else:
        return ProcedureCheckResult(
            procedure_name=config.name,
            detected=False,
            check_pattern=config.patterns,
            error_message="FileCheck binary not available. Set FILECHECK_PATH env var or install Triton.",
            display_attributes=display_attributes or [],
        )


def extract_ir_metadata(
    ir_content: str,
    display_attributes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Extract metadata from IR content based on display_attributes configuration.

    Each display_attribute specifies how to extract its value:
    - source "module_attrs" + extract_rule "regex": regex search on IR content
    - source "ir_content" + extract_rule "count": count occurrences of extract_pattern
    - source "ir_content" + extract_rule "regex": regex search, return capture group
    - source "ir_content" + extract_rule "dot_shape": extract from tt.dot tensor shape
    - source "computed" + compute_rule: named computation from other attributes

    Args:
        ir_content: The IR content string to extract metadata from.
        display_attributes: List of display attribute configs from JSON.

    Returns:
        Dictionary of extracted metadata values.
    """
    if not ir_content:
        return {}

    metadata: dict[str, Any] = {}

    # Always extract module attributes string (for display)
    module_attrs_match = re.search(
        r"module\s+attributes\s*\{([^}]+)\}", ir_content, re.MULTILINE
    )
    if module_attrs_match:
        metadata["module_attributes"] = module_attrs_match.group(1).strip()

    if not display_attributes:
        return metadata

    # Index attributes by key for dependency resolution
    attr_by_key: dict[str, dict[str, Any]] = {}
    for attr in display_attributes:
        attr_by_key[attr.get("key", "")] = attr

    # Lazy-cache for dot_shape extraction (run once, use for multiple keys)
    _dot_shape_cache: dict[str, Any] | None = None

    def _get_dot_shape() -> dict[str, Any]:
        nonlocal _dot_shape_cache
        if _dot_shape_cache is None:
            _dot_shape_cache = _extract_dot_shape(ir_content)
        return _dot_shape_cache

    # Pass 1: Extract non-computed attributes
    for attr in display_attributes:
        source = attr.get("source", "")
        key = attr.get("key", "")
        extract_rule = attr.get("extract_rule", "")
        extract_pattern = attr.get("extract_pattern", "")

        if source == "computed":
            continue  # handled in pass 2

        if source == "module_attrs":
            if extract_rule == "regex" and extract_pattern:
                match = re.search(extract_pattern, ir_content)
                if match:
                    value = match.group(1)
                    if attr.get("type") == "number":
                        metadata[key] = int(value)
                    else:
                        metadata[key] = value

        elif source == "ir_content":
            if extract_rule == "count" and extract_pattern:
                metadata[key] = ir_content.count(extract_pattern)
            elif extract_rule == "regex" and extract_pattern:
                match = re.search(extract_pattern, ir_content)
                if match:
                    group_idx = attr.get("extract_group", 1)
                    value = match.group(group_idx)
                    if attr.get("type") == "number":
                        metadata[key] = int(value)
                    else:
                        metadata[key] = value
            elif extract_rule == "dot_shape":
                extract_field = attr.get("extract_field", key)
                dot_shape = _get_dot_shape()
                if extract_field in dot_shape:
                    metadata[key] = dot_shape[extract_field]

    # Pass 2: Resolve computed attributes (may depend on pass 1 results)
    for attr in display_attributes:
        if attr.get("source") != "computed":
            continue
        key = attr.get("key", "")
        if key in metadata:
            continue

        compute_rule = attr.get("compute_rule", "")
        # Ensure dependencies are resolved first
        compute_from = attr.get("compute_from", [])
        for dep_key in compute_from:
            if dep_key not in metadata:
                if dep_key in attr_by_key:
                    # Dependency is a declared display attribute - extract it
                    dep_attr = attr_by_key[dep_key]
                    _extract_single_attr(dep_attr, ir_content, metadata, _get_dot_shape)
                else:
                    # Dependency is not a declared attribute - try dot_shape cache
                    # (e.g. input_dtype needed for tile_size_bits computation)
                    dot_shape = _get_dot_shape()
                    if dep_key in dot_shape:
                        metadata[dep_key] = dot_shape[dep_key]

        value = _run_compute_rule(compute_rule, metadata, ir_content)
        if value is not None:
            metadata[key] = value

    return metadata


def _extract_single_attr(
    attr: dict[str, Any],
    ir_content: str,
    metadata: dict[str, Any],
    get_dot_shape: Any,
) -> None:
    """Extract a single attribute value and store in metadata. Used for dependency resolution."""
    key = attr.get("key", "")
    if key in metadata:
        return
    source = attr.get("source", "")
    extract_rule = attr.get("extract_rule", "")
    extract_pattern = attr.get("extract_pattern", "")

    if source == "module_attrs" and extract_rule == "regex" and extract_pattern:
        match = re.search(extract_pattern, ir_content)
        if match:
            value = match.group(1)
            metadata[key] = int(value) if attr.get("type") == "number" else value
    elif source == "ir_content":
        if extract_rule == "count" and extract_pattern:
            metadata[key] = ir_content.count(extract_pattern)
        elif extract_rule == "regex" and extract_pattern:
            match = re.search(extract_pattern, ir_content)
            if match:
                group_idx = attr.get("extract_group", 1)
                value = match.group(group_idx)
                metadata[key] = int(value) if attr.get("type") == "number" else value
        elif extract_rule == "dot_shape":
            extract_field = attr.get("extract_field", key)
            dot_shape = get_dot_shape()
            if extract_field in dot_shape:
                metadata[key] = dot_shape[extract_field]


def _run_compute_rule(
    rule: str, metadata: dict[str, Any], ir_content: str
) -> Any | None:
    """Run a named computation rule to derive an attribute value."""
    if rule == "pp_clusters":
        return _compute_pp_clusters(metadata, ir_content)
    elif rule == "tile_size_bits":
        return _compute_tile_size_bits(metadata)
    return None


def _compute_pp_clusters(metadata: dict[str, Any], ir_content: str) -> int | None:
    """Compute number of PP clusters from warps and barrier counts.

    Extracts dependencies from IR content if not already in metadata.
    """
    num_warps = metadata.get("num_warps")
    if num_warps is None:
        warps_match = re.search(r'"ttg\.num-warps"\s*=\s*(\d+)', ir_content)
        num_warps = int(warps_match.group(1)) if warps_match else 0

    cond_barrier_count = metadata.get("cond_barrier_count")
    if cond_barrier_count is None:
        cond_barrier_count = ir_content.count("amdgpu.cond_barrier")

    has_cond_barrier = cond_barrier_count > 0
    if num_warps == 4 and not has_cond_barrier:
        return 1
    elif num_warps == 8 and has_cond_barrier:
        dot_count = metadata.get("dot_count")
        if dot_count is None:
            dot_count = ir_content.count("tt.dot")
        if dot_count >= 4:
            return 4
        else:
            return 2
    return None


def _compute_tile_size_bits(metadata: dict[str, Any]) -> int | None:
    """Compute tile size in bits from tile dimensions and input dtype."""
    tile_m = metadata.get("tile_m")
    tile_n = metadata.get("tile_n")
    tile_k = metadata.get("tile_k")
    input_dtype = metadata.get("input_dtype")
    if tile_m is None or tile_n is None or tile_k is None or input_dtype is None:
        return None
    input_bits = _get_dtype_bits(input_dtype)
    if input_bits is None:
        return None
    return tile_m * tile_n * tile_k * input_bits


def _extract_dot_shape(ir_content: str) -> dict[str, Any]:
    """
    Extract tile dimensions from tt.dot tensor types in IR content.

    Parses the tt.dot operation signature to extract:
    - tile_m, tile_n, tile_k: tile dimensions
    - input_dtype, output_dtype: data types

    Returns:
        Dictionary containing dot shape attributes.
    """
    info: dict[str, Any] = {}

    # Pattern: tt.dot ... : tensor<MxKxdtype, #encoding> * tensor<KxNxdtype, #encoding> -> tensor<MxNxresult_dtype, #encoding>
    dot_pattern = re.search(
        r"tt\.dot[^:]*:\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9]+)[^*]*\*\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9]+)[^-]*->\s*tensor<(\d+)x(\d+)x([a-zA-Z0-9]+)",
        ir_content,
    )
    if dot_pattern:
        a_k = int(dot_pattern.group(2))
        input_dtype = dot_pattern.group(3).strip()
        out_m = int(dot_pattern.group(7))
        out_n = int(dot_pattern.group(8))
        output_dtype = dot_pattern.group(9).strip()

        info["tile_m"] = out_m
        info["tile_n"] = out_n
        info["tile_k"] = a_k
        info["input_dtype"] = input_dtype
        info["output_dtype"] = output_dtype

    return info


def _get_dtype_bits(dtype_str: str) -> int | None:
    """Get the number of bits for a given data type string."""
    dtype_bits_map = {
        "f16": 16,
        "bf16": 16,
        "f32": 32,
        "f64": 64,
        "fp16": 16,
        "bfloat16": 16,
        "fp32": 32,
        "fp64": 64,
        "i8": 8,
        "i16": 16,
        "i32": 32,
        "i64": 64,
        "int8": 8,
        "int16": 16,
        "int32": 32,
        "int64": 64,
        "f8e4m3fnuz": 8,
        "f8e5m2fnuz": 8,
        "f8e4m3fn": 8,
        "f8e5m2": 8,
    }
    # Clean up the dtype string and look for matches
    dtype_clean = dtype_str.strip().lower()
    for key, bits in dtype_bits_map.items():
        if key in dtype_clean:
            return bits
    return None


def find_procedures_with_patterns(
    procedure_configs: list[dict[str, Any]],
    ir_key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
    ttir_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Check multiple procedures with custom FileCheck patterns in IR content."""
    ir_content = load_ir_contents(ir_key, file_content, file_path)

    if not ir_content:
        return {
            cfg["name"]: {
                "procedure_name": cfg["name"],
                "detected": False,
                "error_message": "No IR content available",
                "display_attributes": cfg.get("display_attributes", []),
            }
            for cfg in procedure_configs
        }

    # Extract dot shape info from TTIR content if ttir_key is provided
    # (TTIR may have cleaner dot shape info than TTGIR)
    tile_info = {}
    if ttir_key:
        ttir_content = load_ir_contents(ttir_key, file_content, file_path)
        if ttir_content:
            tile_info = _extract_dot_shape(ttir_content)

    results = {}
    for cfg in procedure_configs:
        display_attrs = cfg.get("display_attributes", [])
        config = ProcedureCheckConfig(
            name=cfg["name"],
            patterns=cfg.get("patterns", f"CHECK: {cfg['name']}"),
            description=cfg.get("description", ""),
        )
        result = run_filecheck(ir_content, config, display_attributes=display_attrs)
        result_dict = asdict(result)
        result_dict["detection_method"] = "filecheck"
        # Include message from config if available
        if cfg.get("message"):
            result_dict["message"] = cfg["message"]
        # Merge tile_info into the dynamic attributes dict
        if tile_info:
            for key, value in tile_info.items():
                if key not in result_dict["attributes"]:
                    result_dict["attributes"][key] = value
        results[cfg["name"]] = result_dict

    return results


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


def _analyze_buffer_ops(
    ttgir_key: str,
    amdgcn_key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Analyze AMD buffer operations from TTGIR and AMDGCN content."""
    io_counts = {}
    ttgir_bufferops_info = process_amd_ttgir_bufferops(
        ttgir_key, file_content, file_path
    )
    gcn_bufferops_info = process_amd_gcn_bufferops(amdgcn_key, file_content, file_path)
    if ttgir_bufferops_info:
        io_counts["amd_ttgir_bufferops_count"] = ttgir_bufferops_info
    if gcn_bufferops_info:
        io_counts["amd_gcn_bufferops_count"] = gcn_bufferops_info
    return io_counts


def _analyze_loop_schedules(
    ttir_key: str,
    ttgir_key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
    payload: dict[str, Any],
    source_mappings: dict[str, Any],
) -> list[dict]:
    """Generate loop schedule information from TTIR and TTGIR content."""
    python_source_content = None
    python_source_start_line = 1
    python_source_info = payload.get("python_source")
    if python_source_info:
        python_source_content = python_source_info.get("code")
        python_source_start_line = python_source_info.get("start_line", 1)

    return generate_loop_schedule(
        ttir_key,
        ttgir_key,
        file_content,
        file_path,
        source_mappings,
        python_source_content,
        python_source_start_line,
    )


def _generate_ir_analysis(
    entry: str, procedure_checks: list[dict[str, Any]] | None = None
):
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
        io_counts = _analyze_buffer_ops(ttgir_key, amdgcn_key, file_content, file_path)
        if io_counts:
            ir_analysis["io_counts"] = io_counts
    if ttir_key and ttgir_key:
        loop_schedule = _analyze_loop_schedules(
            ttir_key, ttgir_key, file_content, file_path, payload, source_mappings
        )
        if loop_schedule:
            ir_analysis["loop_schedules"] = loop_schedule

    # Add FileCheck-based procedure detection if procedure_checks are specified
    if procedure_checks and ttgir_key:
        procedure_results = find_procedures_with_patterns(
            procedure_checks,
            ttgir_key,
            file_content,
            file_path,
            ttir_key=ttir_key,
        )
        if procedure_results:
            ir_analysis["procedure_checks"] = procedure_results

    return ir_analysis
