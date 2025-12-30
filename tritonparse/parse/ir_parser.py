#  Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List

logger = logging.getLogger("SourceMapping")

# the definition of the #loc directive. they are in the bottom of the IR files
# Example:#loc2 = loc("/tmp/torchinductor_yhao/yp/abcdef.py":20:28)
# Note: This should only match numbered locs like #loc1, #loc2, not bare #loc
LOC_PATTERN = re.compile(r'#loc(\d+) = loc\("([^"]+)":(\d+):(\d+)\)')

# the reference to the #loc directive. they are in the end of lines of the IR files
# Example: loc(#loc2)
CODE_LOC_PATTERN = re.compile(r".*loc\(#loc(\d*)\)\s*$")

# this pattern is used in the first function arguments line.
DIRECT_FILE_PATTERN = re.compile(r'.*loc\("([^"]+)":(\d+):(\d+)\)')

# the definition of the PTX loc directive.
# Example: .loc 1 0 50 // abcdef.py:0:50
PTX_LOC_PATTERN = re.compile(
    r"^\s*\.loc\s+\d+\s+(\d+)\s+(\d+)\s+//\s*(.+?):(\d+):(\d+)"
)

# the definition of the AMDGCN loc directive.
# Example: .loc	1 32 30                         ; abcd.py:32:30
# .loc	1 32 46 is_stmt 0               ; abcd.py:32:46
AMDGCN_LOC_PATTERN = re.compile(
    r".*loc\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+[^;]*)?;\s*(.+?):(\d+):(\d+)"
)


# alias loc definitions in TTGIR/TTIR
# Example: #loc16 = loc("pid"(#loc2))
# Example: #loc13 = loc("x_ptr"(#loc)) - bare #loc without number
ALIAS_WITH_NAME_PATTERN = re.compile(
    r'#loc(\d+)\s*=\s*loc\("([^"]+)"\s*\(\s*#loc(\d*)\s*\)\s*\)'
)

# Example: #loc20 = loc(#loc16)
ALIAS_SIMPLE_PATTERN = re.compile(r"#loc(\d+)\s*=\s*loc\(\s*#loc(\d*)\s*\)")

# Callsite loc definitions in TTIR/TTGIR
# Example: #loc220 = loc(callsite(#loc57 at #loc190))
# Captures: loc_id, callee_loc_id, caller_loc_id
# Note: Uses (\d*) to match optional numbers (for bare #loc references)
CALLSITE_PATTERN = re.compile(
    r"#loc(\d+)\s*=\s*loc\(\s*callsite\(\s*#loc(\d*)\s+at\s+#loc(\d*)\s*\)\s*\)"
)


def extract_loc_definitions(ir_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Extracts location definitions from the given IR content.

    This function searches for #loc directives in the provided IR content string.
    It identifies the main #loc directive, which is a special case located at the top
    of the IR files, and any subsequent #loc directives that define source file locations.

    Args:
        ir_content (str): The content of the IR file as a string.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping location IDs to their corresponding
        file names, line numbers, and column numbers.
    """
    locations = {}
    # The first #loc directive is a special case. It locates at the top of the IR files
    # Store it with empty string "" as key to avoid conflict with #loc1
    main_match = re.search(r'#loc = loc\("([^"]+)":(\d+):(\d+)\)', ir_content)
    if main_match:
        locations[""] = {
            "file": main_match.group(1),
            "line": int(main_match.group(2)),
            "column": int(main_match.group(3)),
        }
    # #loc1 = loc(unknown) is another special case. We ignore it.
    for loc_id, filename, line, col in LOC_PATTERN.findall(ir_content):
        key = loc_id
        locations[key] = {"file": filename, "line": int(line), "column": int(col)}

    # Handle alias-style loc definitions that reference another #loc
    # Build alias map first: alias_id -> target_id
    alias_map: Dict[str, str] = {}
    for m in ALIAS_WITH_NAME_PATTERN.finditer(ir_content):
        alias_id, _name, target_id = m.groups()
        # Empty target_id means bare #loc, map to "" (main loc key)
        alias_map[alias_id] = target_id or ""
    for m in ALIAS_SIMPLE_PATTERN.finditer(ir_content):
        alias_id, target_id = m.groups()
        # Empty target_id means bare #loc, map to "" (main loc key)
        alias_map[alias_id] = target_id or ""

    # Build definition line map and alias name map by scanning lines
    def_line_map: Dict[str, int] = {}
    alias_name_map: Dict[str, str] = {}
    main_loc_line: int = 0
    for i, line in enumerate(ir_content.split("\n"), start=1):
        if m := ALIAS_WITH_NAME_PATTERN.search(line):
            alias_id, name, target_id = m.groups()
            def_line_map[alias_id] = i
            alias_name_map[alias_id] = name
            # ensure alias map is populated even if only found in line scan
            # Empty target_id means bare #loc, map to "" (main loc key)
            alias_map.setdefault(alias_id, target_id or "")
        elif m := ALIAS_SIMPLE_PATTERN.search(line):
            alias_id, target_id = m.groups()
            def_line_map[alias_id] = i
            # Empty target_id means bare #loc, map to "" (main loc key)
            alias_map.setdefault(alias_id, target_id or "")
        if m2 := LOC_PATTERN.search(line):
            base_id, _fn, _ln, _col = m2.groups()
            def_line_map[base_id] = i
        if re.search(r'#loc\s*=\s*loc\("[^"]+":\d+:\d+\)', line):
            # main #loc = loc("file":line:col) without id
            main_loc_line = main_loc_line or i

    # Resolve aliases to base locations (file/line/column)
    resolving_stack = set()

    def resolve_alias(current_id: str) -> Dict[str, Any]:
        # Already a concrete location
        if current_id in locations:
            return locations[current_id]
        # Detect cycles
        if current_id in resolving_stack:
            return {}
        resolving_stack.add(current_id)
        parent_id = alias_map.get(current_id)
        result: Dict[str, Any] = {}
        if parent_id is not None:
            base = resolve_alias(parent_id)
            if base:
                # copy to avoid sharing the same dict by reference
                result = {
                    "file": base.get("file"),
                    "line": base.get("line"),
                    "column": base.get("column"),
                }
                locations[current_id] = result
        resolving_stack.remove(current_id)
        return result

    # Resolve aliases and attach alias metadata
    for alias_id, target_id in alias_map.items():
        if alias_id not in locations:
            resolve_alias(alias_id)

    # Collect callsite definitions
    callsite_defs = []
    for i, line in enumerate(ir_content.split("\n"), start=1):
        if m := CALLSITE_PATTERN.search(line):
            loc_id, callee_id, caller_id = m.groups()
            # Empty strings map to main loc key ""
            callsite_defs.append((loc_id, callee_id or "", caller_id or "", i))

    # Resolve callsite definitions
    # A callsite inherits the location from its callee (the code being called)
    # and stores a reference to its caller (the code doing the calling)
    for loc_id, callee_id, caller_id, def_line in callsite_defs:
        if loc_id not in locations:  # Avoid overwriting existing definitions
            if callee_id in locations:
                # Inherit location info from callee
                callee_info = locations[callee_id]
                locations[loc_id] = {
                    "file": callee_info["file"],
                    "line": callee_info["line"],
                    "column": callee_info["column"],
                    "def_line": def_line,
                    "is_callsite": True,
                    "callsite_callee": callee_id,
                    "callsite_caller": caller_id,
                }
            else:
                logger.warning(
                    f"Callsite #loc{loc_id} references undefined callee #loc{callee_id}"
                )
                # Note: We don't add this callsite to locations since callee is missing

    # Verify caller references (warning only, don't block)
    for loc_id, _callee_id, caller_id, _def_line in callsite_defs:
        if loc_id in locations and caller_id and caller_id not in locations:
            logger.warning(
                f"Callsite #loc{loc_id} references undefined caller #loc{caller_id}"
            )

    # Attach definition line and alias metadata
    for k, v in def_line_map.items():
        if k in locations:
            locations[k]["def_line"] = v
    for alias_id, target_id in alias_map.items():
        if alias_id in locations:
            locations[alias_id]["alias_of"] = target_id
            if alias_id in alias_name_map:
                locations[alias_id]["alias_name"] = alias_name_map[alias_id]

    # Attach definition line metadata
    for k, v in def_line_map.items():
        if k in locations:
            locations[k]["def_line"] = v
    if main_loc_line and "" in locations:
        locations[""]["def_line"] = main_loc_line
    return locations


def extract_code_locations(ir_content: str) -> Dict[int, str]:
    """
    Extracts code location mappings from the given IR content.

    This function scans through the provided IR content line by line and identifies
    lines that contain location references. It uses regular expressions to match
    both the #loc directives and direct file references. The function returns a
    dictionary mapping line numbers to their corresponding location identifiers.
    Limitations:
        For the first function arguments line, it may use some #loc(file:line:col), DIRECT_FILE_PATTERN, we only use the first location reference.
    Args:
        ir_content (str): The content of the IR file as a string.

    Returns:
        Dict[int, str]: A dictionary mapping line numbers to location identifiers,
        which can be either a #loc identifier or a direct file reference.
    """
    line_to_loc = {}
    for i, line in enumerate(ir_content.split("\n"), start=1):
        if m := CODE_LOC_PATTERN.search(line):
            line_to_loc[i] = m.group(1) or "0"
        elif m := DIRECT_FILE_PATTERN.search(line):
            file_path, ln, col = m.groups()
            line_to_loc[i] = f"direct:{file_path}:{ln}:{col}"
    return line_to_loc


def extract_ptx_amdgcn_mappings(
    content: str, other_mappings: List[Any] = None, ir_type: str = "ptx"
) -> Dict[str, Dict[str, Any]]:
    """
    Extract mappings from PTX code where `.loc` directives provide source file and line info.
    This function only processes code between the function begin and end markers (e.g., "// -- Begin function" and "// -- End function"). The PTX source code line mapping is quite different from that of other IRs. It segments the PTX code using the .loc directive, where each .loc directive provides information for mapping to a source code line.

    This function:
    1. Identifies the function boundary in PTX code
    2. Only processes code within the function boundary
    3. Maps PTX lines with `.loc` directives to source files and line numbers
    4. Associates subsequent code lines with the most recent `.loc` directive

    Args:
        ptx_content: The content of the PTX file

    Returns:
        Dictionary mapping PTX line numbers to source location information
    """
    mappings = {}
    current_mapping = None

    # Mark function scope
    function_start_line = 0
    function_end_line = 0
    # filename: {file_path, ...}
    referenced_files = defaultdict(set)
    if other_mappings is None:
        other_mappings = []
    for other in other_mappings:
        for _, info in other.items():
            if "file" in info:
                file_name = os.path.basename(info["file"])
                referenced_files[file_name].add(info["file"])

    def get_file_path(filename: str) -> str:
        file_path = filename
        if not os.path.isabs(filename):
            logger.debug(
                f"Filename '{filename}' does not contain a path. Attempting to resolve."
            )
            # Attempt to resolve the filename to a full path using referenced_files
            if filename in referenced_files:
                if len(referenced_files[filename]) > 1:
                    logger.debug(
                        f"Filename '{filename}' has multiple file paths. Using the first one."
                    )
                file_path = list(referenced_files[filename])[0]
                logger.debug(f"Resolved filename '{filename}' to {file_path}")
            else:
                logger.debug(f"Filename '{filename}' not found in referenced files.")
        return file_path

    # Regular expressions to match function start and end markers
    # @TODO: need to double check if the PTX content only contains one function
    begin_func_pattern = re.compile(
        r"(?:(?://|;)\s*(?:\.globl\s+\S+\s+)?|\.globl\s+\S+\s+;\s*)--\s*Begin function"
    )
    end_func_pattern = re.compile(r"(?://|;)\s*--\s*End function")

    # First scan: find function boundaries
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        if begin_func_pattern.search(line) and function_start_line == 0:
            function_start_line = i
        elif end_func_pattern.search(line) and function_start_line > 0:
            function_end_line = i
            break

    # If no function boundaries are found, return empty mapping
    if function_start_line == 0 or function_end_line == 0:
        logger.warning(
            f"Could not identify {ir_type} function boundaries. No {ir_type} mappings generated."
        )
        return mappings

    logger.debug(
        f"Processing {ir_type} function from line {function_start_line} to {function_end_line}"
    )

    is_ptx = ir_type == "ptx"
    is_amdgcn = ir_type == "amdgcn"

    tmp_loc_pattern = PTX_LOC_PATTERN if is_ptx else AMDGCN_LOC_PATTERN
    # Second scan: process code within function body
    # pay attention to the line number, it starts from 0 but the function_start_line starts from 1
    for i, line in enumerate(
        lines[function_start_line:function_end_line], start=function_start_line + 1
    ):
        try:
            # Check .loc directive line
            match = tmp_loc_pattern.match(line)
            if match:
                if is_ptx:
                    py_line, py_col, filename, _, _ = match.groups()
                elif is_amdgcn:
                    py_file_index, py_line, py_col, filename, _, _ = match.groups()
                else:
                    logger.error(f"Unknown IR type: {ir_type}")
                    raise ValueError(f"Unknown IR type: {ir_type}")
                file_path = get_file_path(filename)
                # Create new mapping
                current_mapping = {
                    "file": file_path,
                    "line": int(py_line),
                    "column": int(py_col),
                    f"{ir_type}_line": i,
                }
                # Store mapping
                mappings[str(i)] = current_mapping
            elif current_mapping:
                # For lines without their own .loc after .loc directive, associate with the nearest .loc mapping
                # Only process non-empty, non-comment meaningful code lines
                line_content = line.strip()
                if line_content and not (
                    (is_ptx and line_content.startswith("//"))
                    or (is_amdgcn and line_content.startswith(";"))
                ):
                    mappings[str(i)] = {
                        "file": current_mapping["file"],
                        "line": current_mapping["line"],
                        "column": current_mapping["column"],
                        f"{ir_type}_line": i,
                    }
        except Exception as e:
            logger.error(f"Error processing line {i}: {e}")
            logger.error(f"Line content: {line}")
            raise e
    return mappings
