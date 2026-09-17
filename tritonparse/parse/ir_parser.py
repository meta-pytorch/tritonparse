#  Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from tritonparse.tp_logger import get_logger

logger = get_logger("SourceMapping")

# the definition of the #loc directive. they are in the bottom of the IR files
# Example:#loc2 = loc("/tmp/torchinductor_yhao/yp/abcdef.py":20:28)
# Note: This should only match numbered locs like #loc1, #loc2, not bare #loc
LOC_PATTERN = re.compile(r'#loc(\d+) = loc\("([^"]+)":(\d+):(\d+)\)')

# the reference to the #loc directive. they are in the end of lines of the IR files
# Example: loc(#loc2)
CODE_LOC_PATTERN = re.compile(r".*loc\(#loc(\d*)\)\s*$")

# this pattern is used in the first function arguments line.
DIRECT_FILE_PATTERN = re.compile(r'.*loc\("([^"]+)":(\d+):(\d+)\)')

# LLVM IR debug metadata (LLIR).  Unlike MLIR's `#loc`, LLVM records source
# locations as `!dbg !N` references into `!N = !DILocation(...)` nodes.
# Example:
#   %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !11
#   !11 = !DILocation(line: 208, column: 12, scope: !4)
# `scope` resolves to a DIFile through DISubprogram/DILexicalBlockFile, and
# `inlinedAt` chains an inlined frame to its call site -- `tl.sum` nests two
# deep (standard.py -> standard.py -> the user's kernel).
LLVM_DIFILE_PATTERN = re.compile(
    r'^!(\d+)\s*=\s*!DIFile\(filename:\s*"([^"]*)",\s*directory:\s*"([^"]*)"',
    re.MULTILINE,
)
LLVM_DISCOPE_PATTERN = re.compile(
    r"^!(\d+)\s*=\s*(?:distinct\s+)?"
    r"!DI(?:Subprogram|LexicalBlockFile|LexicalBlock)\(.*?\bfile:\s*!(\d+)",
    re.MULTILINE,
)
# `file:` is optional on DILexicalBlock/DILexicalBlockFile -- such a scope
# inherits its file from the parent `scope:`, so resolution has to walk up
# rather than do a single lookup.
LLVM_DISCOPE_PARENT_PATTERN = re.compile(
    r"^!(\d+)\s*=\s*(?:distinct\s+)?"
    r"!DI(?:LexicalBlockFile|LexicalBlock)\(.*?\bscope:\s*!(\d+)",
    re.MULTILINE,
)
# `!dbg` on a `define` line points at the DISubprogram, not a DILocation.
# It carries the kernel's own file and def line.
LLVM_DISUBPROGRAM_PATTERN = re.compile(
    r"^!(\d+)\s*=\s*(?:distinct\s+)?!DISubprogram\((?=.*?\bfile:\s*!(\d+))"
    r"(?=.*?\bline:\s*(\d+))",
    re.MULTILINE,
)
LLVM_DILOCATION_PATTERN = re.compile(
    r"^!(\d+)\s*=\s*!DILocation\(line:\s*(\d+)"
    r"(?:,\s*column:\s*(\d+))?,\s*scope:\s*!(\d+)"
    r"(?:,\s*inlinedAt:\s*!(\d+))?",
    re.MULTILINE,
)
LLVM_DBG_REF_PATTERN = re.compile(r"!dbg\s+!(\d+)")

# the definition of the PTX loc directive.
# Example: .loc 1 0 50 // abcdef.py:0:50
PTX_LOC_PATTERN = re.compile(
    # The leading index is the .loc's own file; it identifies this frame as the
    # target of another directive's `inlined_at`, so the chain can be walked.
    r"^\s*\.loc\s+(\d+)\s+(\d+)\s+(\d+)"
    # Inlined code carries extra fields before the comment, e.g.
    #   .loc 2 43 13, function_name $L__info_string0, inlined_at 1 288 21 // f.py:43:13
    # Requiring "//" straight after the column silently skipped every one of
    # them, leaving PTX with no entries at all for inlined code.
    r"(?:\s*,[^/]*?\binlined_at\s+(\d+)\s+(\d+)\s+(\d+))?"
    r"[^/]*//\s*(.+?):(\d+):(\d+)"
)
# `.file N "path"` -- resolves the file index used by `inlined_at`.
PTX_FILE_PATTERN = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"', re.MULTILINE)

# the definition of the AMDGCN loc directive.
# Example: .loc	1 32 30                         ; abcd.py:32:30
# .loc	1 32 46 is_stmt 0               ; abcd.py:32:46
AMDGCN_LOC_PATTERN = re.compile(
    r".*loc\s+(\d+)\s+(\d+)\s+(\d+)(?:\s+[^;]*)?;\s*(.+?):(\d+):(\d+)"
)

# the definition of the SASS source mapping pattern.
# Example: //## File "/path/to/source.py", line 188
SASS_LOC_PATTERN = re.compile(r'//## File "([^"]+)", line (\d+)')

# the definition of the SASS PC offset pattern.
# Example: /*0000*/ LDC R1, c[0x0][0x28] ;
SASS_PC_PATTERN = re.compile(r".*\/\*([0-9a-fA-F]+)\*\/.*")


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

    # Resolve each callsite to the OUTERMOST frame of its chain. `file`/`line`
    # describe the callee -- the inlined library code actually emitted -- so on
    # their own they never point at the user's kernel. The root does.
    for info in locations.values():
        if not info.get("is_callsite"):
            continue
        cur, seen = info, set()
        while cur.get("is_callsite"):
            nxt = cur.get("callsite_caller")
            # `is None` rather than falsy: "" is the key of the bare `#loc`, so
            # `loc(callsite(#locN at #loc))` is a real caller and must resolve.
            if nxt is None or nxt in seen or nxt not in locations:
                break
            seen.add(nxt)
            cur = locations[nxt]
        # Only a frame that is itself NOT a callsite is the root of the chain.
        # The loop also exits by `break` -- on a missing, cyclic or unresolvable
        # caller -- and there `cur` is still a callsite, so its file/line are a
        # callee: a library line. Recording that as `inlined_at_*` would be the
        # exact mis-attribution this pass exists to remove. A broken chain means
        # the call site is unknown, so the entry stays plain, matching what the
        # PTX path does when a `.file` index will not resolve.
        if cur is not info and not cur.get("is_callsite"):
            info["inlined_at_file"] = cur["file"]
            info["inlined_at_line"] = cur["line"]

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


def _iter_sass_instructions(sass_content: str):
    """
    Iterate over SASS content, yielding source-mapped entries.

    Parses SASS text line by line, tracking the current source location from
    ``//## File`` comments and matching SASS instruction lines that contain
    ``/*hex_offset*/`` patterns.

    Lines referencing ``.nv_debug_ptx_txt`` are skipped so they never become
    the active source location.

    Inlined call stacks::

        //## File "standard.py", line 170 inlined at "standard.py", line 194
        //## File "standard.py", line 194 inlined at "kernel.py", line 227
        //## File "kernel.py", line 227 inlined at "kernel.py", line 593
        //## File "kernel.py", line 593
                /*0bc0*/   INSTR ;

    ``nvdisasm`` emits one ``//## File`` comment per inlined frame, ordered
    innermost-first. The instruction's true source is the innermost frame
    (the first comment, here ``standard.py:170``); the following comments are
    the outer call sites. The instruction is therefore attributed to the
    first comment of its block, not the last.

    Yields:
        Tuple of (line_num, pc_hex, source_info):
        - line_num: 1-based line number in the SASS text
        - pc_hex: hex offset string (e.g., "0180"), or None for //## File
          comment lines
        - source_info: dict with {file, line, column}. For instruction lines
          this is the innermost frame of the preceding ``//## File`` block;
          for comment lines it is that comment's own literal location.
    """
    # Source location attributed to the next instruction line: the innermost
    # frame of the current //## File block.
    instr_source_info = None
    # True while the current block has not yet captured its innermost frame.
    # Once set, later //## File comments in the same block (outer call sites)
    # must not override the instruction's attribution.
    awaiting_innermost = True
    # Last frame seen in the current block. nvdisasm orders frames
    # innermost-first, so the last one is the outermost call site -- the line in
    # the user's kernel. The innermost is what the instruction *is*; the
    # outermost is where the user wrote it.
    outermost_source_info = None
    lines = sass_content.split("\n")

    for line_num, line in enumerate(lines, 1):
        if ".nv_debug_ptx_txt" in line:
            continue

        match = SASS_LOC_PATTERN.match(line.strip())
        if match:
            file_path, source_line = match.groups()
            comment_source_info = {
                "file": file_path,
                "line": int(source_line),
                "column": 0,
            }
            if awaiting_innermost:
                instr_source_info = comment_source_info
                awaiting_innermost = False
            outermost_source_info = comment_source_info
            # The comment line itself maps to its own literal location.
            yield line_num, None, comment_source_info

        elif instr_source_info:
            pc_match = SASS_PC_PATTERN.match(line)
            if pc_match:
                info = instr_source_info
                if outermost_source_info is not instr_source_info:
                    # More than one frame: this instruction came from inlined
                    # code. Keep the innermost as file/line (unchanged
                    # behaviour) and record the call site alongside it.
                    info = dict(instr_source_info)
                    info["is_callsite"] = True
                    info["inlined_at_file"] = outermost_source_info["file"]
                    info["inlined_at_line"] = outermost_source_info["line"]
                yield line_num, pc_match.group(1), info
                # The next //## File comment begins a new inline stack.
                awaiting_innermost = True


def extract_sass_mappings(sass_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract source mappings from SASS content.

    Key = SASS text line number (str). Used by tritonparse UI and cross-IR
    mapping.

    SASS format::

        Function:kernel_name
                //## File "/path/to/source.py", line 188
                //## File ".nv_debug_ptx_txt", line 19    # Skip this line
                        /*0000*/                   MOV R1, c[0x0][0x28] ;

    Args:
        sass_content: The content of the SASS file as a string.

    Returns:
        A dictionary mapping SASS text line numbers (str) to their
        corresponding source file, line numbers, and column numbers.
    """
    mappings = {}
    for line_num, _pc_hex, source_info in _iter_sass_instructions(sass_content):
        entry = {
            "file": source_info["file"],
            "line": source_info["line"],
            "column": source_info["column"],
            "sass_line": line_num,
        }
        if source_info.get("is_callsite"):
            entry["is_callsite"] = True
            entry["inlined_at_file"] = source_info["inlined_at_file"]
            entry["inlined_at_line"] = source_info["inlined_at_line"]
        mappings[str(line_num)] = entry
    return mappings


def extract_sass_pc_mappings(sass_content: str) -> Dict[int, Dict[str, Any]]:
    """
    Extract PC-offset-keyed source mappings from SASS content.

    Key = PC offset (int). Designed for trace-analysis tools (e.g. CUTracer)
    that record per-instruction PC offsets and need to resolve them back to
    Python source locations.

    ``//## File`` comment lines (which have no PC offset) are skipped.

    Args:
        sass_content: The SASS text produced by ``nvdisasm -c -gi``.

    Returns:
        A dictionary mapping PC offset integers to source location info.
        Example::

            {
                0:   {"file": "/path/to/source.py", "line": 267, "column": 0, "sass_line": 16},
                16:  {"file": "/path/to/source.py", "line": 267, "column": 0, "sass_line": 17},
                304: {"file": "/path/to/standard.py", "line": 41, "column": 0, "sass_line": 37},
            }
    """
    mappings = {}
    for line_num, pc_hex, source_info in _iter_sass_instructions(sass_content):
        if pc_hex is None:
            continue
        # Use int as key to normalize hex format differences across tools:
        # nvdisasm outputs "0180", CUTracer trace records "0x180".
        # Converting to int (e.g., 384) makes matching straightforward —
        # consumers just do int(their_hex_string, 16) without worrying
        # about zero-padding or "0x" prefixes.
        mappings[int(pc_hex, 16)] = {
            "file": source_info["file"],
            "line": source_info["line"],
            "column": source_info["column"],
            "sass_line": line_num,
        }
    return mappings


def extract_llvm_dbg_mappings(llir_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract source mappings from LLVM IR debug metadata.

    LLVM IR does not use MLIR's ``#loc`` directives, so ``extract_loc_definitions``
    finds nothing in a ``.llir`` file.  Locations live in metadata instead::

        %5 = ... , !dbg !11
        !11 = !DILocation(line: 208, column: 12, scope: !4)
        !4  = distinct !DISubprogram(name: "k_reduce", file: !1, ...)
        !1  = !DIFile(filename: "kernel.py", directory: "/path")

    ``inlinedAt`` marks a location inside an inlined callee and points at the
    call site.  Following tritonparse's existing callsite convention (see
    ``extract_loc_definitions``), the entry's ``file``/``line`` describe the
    **callee** -- the code actually emitted -- ``callsite_caller`` is the
    *immediate* caller so the chain can be walked one frame at a time, and
    ``inlined_at_file``/``inlined_at_line`` give the root of the chain, which
    is the line in the user's kernel.

    NOTE: nothing consumes ``inlined_at_*`` yet.  ``create_python_mapping``
    keys purely on ``info["line"]``, so an inlined LLIR line currently lands on
    the library file and does not reach the user's kernel in the UI -- the same
    behaviour TTIR/TTGIR already have.  Closing that gap means emitting
    ``inlined_at_*`` from the MLIR and SASS paths too and teaching
    ``create_python_mapping`` to prefer it; doing it for LLIR alone would make
    LLIR behave differently from every other stage.  Deliberately left as a
    separate change -- not an oversight.  A follow-up extends ``inlined_at_*``
    to the MLIR and SASS paths and teaches ``create_python_mapping`` to prefer
    it, which closes the gap for every stage at once.

    Args:
        llir_content: The contents of the ``.llir`` file.

    Returns:
        Dictionary mapping LLIR line numbers (as strings) to source locations.
    """
    if not llir_content:
        return {}

    files = {
        m.group(1): os.path.join(m.group(3), m.group(2))
        for m in LLVM_DIFILE_PATTERN.finditer(llir_content)
    }
    scope_to_file = {
        m.group(1): m.group(2) for m in LLVM_DISCOPE_PATTERN.finditer(llir_content)
    }
    scope_to_parent = {
        m.group(1): m.group(2)
        for m in LLVM_DISCOPE_PARENT_PATTERN.finditer(llir_content)
    }
    subprograms = {
        m.group(1): {"file": m.group(2), "line": int(m.group(3))}
        for m in LLVM_DISUBPROGRAM_PATTERN.finditer(llir_content)
    }
    locations = {
        m.group(1): {
            "line": int(m.group(2)),
            "column": int(m.group(3) or 0),
            "scope": m.group(4),
            "inlined_at": m.group(5),
        }
        for m in LLVM_DILOCATION_PATTERN.finditer(llir_content)
    }
    if not locations:
        logger.debug("No !DILocation metadata found in LLIR")
        return {}
    logger.debug(f"Found {len(locations)} !DILocation nodes")

    def file_of_scope(scope_id: Optional[str]) -> Optional[str]:
        """Resolve a scope to a file, walking up `scope:` when `file:` is absent."""
        seen = set()
        while scope_id is not None and scope_id not in seen:
            seen.add(scope_id)
            file_id = scope_to_file.get(scope_id)
            if file_id is not None and file_id in files:
                return files[file_id]
            scope_id = scope_to_parent.get(scope_id)
        return None

    def file_of(loc: Dict[str, Any]) -> Optional[str]:
        return file_of_scope(loc["scope"])

    def root_of(loc_id: str):
        """Walk `inlinedAt` to the outermost call site (the user's kernel)."""
        cur = locations.get(loc_id)
        cur_id = loc_id
        depth = 0
        while cur and cur["inlined_at"] and depth < 32:
            nxt = locations.get(cur["inlined_at"])
            if nxt is None:
                break
            cur_id, cur, depth = cur["inlined_at"], nxt, depth + 1
        return cur_id, cur

    mappings: Dict[str, Dict[str, Any]] = {}
    for lineno, text in enumerate(llir_content.split("\n"), 1):
        m = LLVM_DBG_REF_PATTERN.search(text)
        if not m:
            continue
        loc_id = m.group(1)
        loc = locations.get(loc_id)
        if loc is None:
            # `!dbg` on a `define` line points at the DISubprogram rather than a
            # DILocation.  Map it to the kernel's def line, mirroring the
            # `kind: "loc_def"` entries extract_loc_definitions emits.
            sub = subprograms.get(loc_id)
            if sub is None:
                continue
            sub_file = files.get(sub["file"])
            if sub_file is None:
                continue
            mappings[str(lineno)] = {
                "file": sub_file,
                "line": sub["line"],
                "column": 0,
                "llir_line": lineno,
                "kind": "subprogram",
            }
            continue

        loc_file = file_of(loc)
        if loc_file is None:
            # The cross-stage join key is f"{file}:{line}:{column}"
            # (parse/mapper.py::create_ir_mapping), so emitting file="" here
            # would produce ":line:col" and false-match any other stage entry
            # that also failed to resolve.  Drop the entry instead.
            logger.debug(
                f"LLIR line {lineno}: scope !{loc['scope']} resolves to no DIFile; "
                "skipping"
            )
            continue

        entry = {
            "file": loc_file,
            "line": loc["line"],
            "column": loc["column"],
            "llir_line": lineno,
        }
        if loc["inlined_at"]:
            _, root = root_of(loc_id)
            entry["is_callsite"] = True
            entry["callsite_callee"] = loc_id
            # The IMMEDIATE caller, matching extract_loc_definitions, where a
            # callsite's caller may itself be a callsite so consumers can walk
            # the chain one frame at a time.  The root is below.
            entry["callsite_caller"] = loc["inlined_at"]
            if root is not None:
                root_file = file_of(root)
                if root_file is not None:
                    entry["inlined_at_file"] = root_file
                    entry["inlined_at_line"] = root["line"]
        mappings[str(lineno)] = entry

    logger.debug(f"Mapped {len(mappings)} LLIR lines from debug metadata")
    return mappings


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
    content: str, other_mappings: List[Any] | None = None, ir_type: str = "ptx"
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
    # index -> path, for resolving `inlined_at <index> <line> <col>`
    ptx_files = {m.group(1): m.group(2) for m in PTX_FILE_PATTERN.finditer(content)}

    # `inlined_at` names the IMMEDIATE caller, not the root of the chain. For
    # `tl.sum`, ptxas emits
    #     .loc 2 273 12, ... inlined_at 2 313 12     (standard.py -> standard.py)
    #     .loc 2 313 12, ... inlined_at 1   8  9     (standard.py -> the kernel)
    # so taking the field verbatim files the inner frame under standard.py:313 --
    # a library line -- while MLIR, SASS and LLIR all walk to the root and file
    # it under the kernel line. Build frame -> caller here so PTX can walk too.
    # (the pattern is line-anchored without re.MULTILINE, so scan line by line)
    ptx_inline_parent: Dict[tuple, tuple] = {}
    if is_ptx:
        for _l in lines:
            _m = PTX_LOC_PATTERN.match(_l)
            if _m and _m.group(4) is not None:
                ptx_inline_parent[_m.group(1, 2, 3)] = _m.group(4, 5, 6)
    # Second scan: process code within function body
    # pay attention to the line number, it starts from 0 but the function_start_line starts from 1
    for i, line in enumerate(
        lines[function_start_line:function_end_line], start=function_start_line + 1
    ):
        try:
            # Check .loc directive line
            match = tmp_loc_pattern.match(line)
            if match:
                inl_file_idx = inl_line = None
                if is_ptx:
                    (
                        own_file_idx,
                        py_line,
                        py_col,
                        inl_file_idx,
                        inl_line,
                        inl_col,
                        filename,
                        _,
                        _,
                    ) = match.groups()
                    if inl_line is not None:
                        # Walk to the OUTERMOST frame, as the other parsers do.
                        seen = {(own_file_idx, py_line, py_col)}
                        cur = (inl_file_idx, inl_line, inl_col)
                        while cur in ptx_inline_parent and cur not in seen:
                            seen.add(cur)
                            cur = ptx_inline_parent[cur]
                        inl_file_idx, inl_line, _ = cur
                elif is_amdgcn:
                    py_file_index, py_line, py_col, filename, _, _ = match.groups()
                else:
                    logger.error(f"Unknown IR type: {ir_type}")
                    raise ValueError(f"Unknown IR type: {ir_type}")
                if int(py_line) == 0:
                    # `.loc <f> 0 <c>` is the DWARF encoding for "no line
                    # information". Taking it literally invents a mapping to
                    # python line 0, which does not exist -- the entry looks
                    # mapped in the UI and resolves to nothing. Drop it, and
                    # stop attributing following instructions to a stale .loc.
                    current_mapping = None
                    continue
                file_path = get_file_path(filename)
                # Create new mapping
                current_mapping = {
                    "file": file_path,
                    "line": int(py_line),
                    "column": int(py_col),
                    f"{ir_type}_line": i,
                }
                inl_path = ptx_files.get(inl_file_idx) if inl_line is not None else None
                if inl_path:
                    # Inlined: file/line describe the callee, matching the other
                    # parsers. The call site is the line the user wrote.
                    #
                    # Both fields are set together or not at all. Recording a
                    # caller line beside a callee file would make
                    # `create_python_mapping` file the entry under the caller
                    # line without anything having checked the file it came
                    # from. An unresolvable `.file` index means we do not know
                    # the call site, so the entry stays a plain one.
                    current_mapping["is_callsite"] = True
                    current_mapping["inlined_at_file"] = inl_path
                    current_mapping["inlined_at_line"] = int(inl_line)
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
                    inherited = {
                        "file": current_mapping["file"],
                        "line": current_mapping["line"],
                        "column": current_mapping["column"],
                        f"{ir_type}_line": i,
                    }
                    if current_mapping.get("is_callsite"):
                        inherited["is_callsite"] = True
                        inherited["inlined_at_file"] = current_mapping[
                            "inlined_at_file"
                        ]
                        inherited["inlined_at_line"] = current_mapping[
                            "inlined_at_line"
                        ]
                    mappings[str(i)] = inherited
        except Exception as e:
            logger.error(f"Error processing line {i}: {e}")
            logger.error(f"Line content: {line}")
            raise e
    return mappings


# =============================================================================
# PARSER WRAPPER FUNCTIONS
# =============================================================================
# These wrapper functions serve two purposes:
# 1. Unify the parser signatures to match the (content, mappings, ir_type) interface
# 2. Preserve the original logic from generate_source_mappings (which combined
#    extract_loc_definitions + extract_code_locations for generic IR types)
#
# Note: We use wrappers instead of modifying existing function signatures to
# maintain backward compatibility with other code that may call these functions.
# =============================================================================


def _parse_generic_loc(
    ir_content: str,
    other_mappings: Optional[List[Any]] = None,
    ir_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parser for generic IR formats (TTIR/TTGIR/LLIR) with #loc directives.

    This combines extract_loc_definitions + extract_code_locations to preserve
    the original generate_source_mappings behavior for these IR types.

    Args:
        ir_content: The IR content
        other_mappings: Other mappings (not used for generic_loc)
        ir_type: The IR type (e.g., "ttir", "ttgir", "llir")

    Returns:
        Dictionary mapping line numbers to source locations
    """

    loc_defs = extract_loc_definitions(ir_content)
    logger.debug(f"Found {len(loc_defs)} #loc definitions")

    loc_refs = extract_code_locations(ir_content)
    logger.debug(f"Found {len(loc_refs)} loc references")

    mappings: Dict[str, Dict[str, Any]] = {}
    for ln, loc_id in loc_refs.items():
        if loc_id.startswith("direct:"):
            _, file_path, line, col = loc_id.split(":", 3)
            mappings[str(ln)] = {
                "file": file_path,
                "line": int(line),
                "column": int(col),
                f"{ir_type}_line": ln,
            }
        elif loc_id in loc_defs:
            info = loc_defs[loc_id]
            entry = {
                "file": info["file"],
                "line": info["line"],
                "column": info["column"],
                f"{ir_type}_line": ln,
            }
            # Propagate callsite metadata if present
            if info.get("is_callsite"):
                entry["is_callsite"] = True
                entry["callsite_callee"] = info["callsite_callee"]
                entry["callsite_caller"] = info["callsite_caller"]
                if "inlined_at_line" in info:
                    entry["inlined_at_file"] = info["inlined_at_file"]
                    entry["inlined_at_line"] = info["inlined_at_line"]
            # Propagate alias metadata if present
            if "alias_name" in info:
                entry["alias_name"] = info["alias_name"]
            if "alias_of" in info:
                entry["loc_id"] = loc_id
            mappings[str(ln)] = entry

    # Add separate entries for loc definition lines
    for loc_id, info in loc_defs.items():
        if "def_line" not in info:
            continue
        def_ln = info["def_line"]
        # Only create mapping if this line doesn't already have one
        if str(def_ln) not in mappings:
            entry = {
                "file": info["file"],
                "line": info["line"],
                "column": info["column"],
                f"{ir_type}_line": def_ln,
                "kind": "loc_def",
            }
            if "alias_name" in info:
                entry["alias_name"] = info["alias_name"]
            if "alias_of" in info:
                entry["loc_id"] = loc_id
            mappings[str(def_ln)] = entry

    return mappings


def _parse_llvm_dbg(
    ir_content: str,
    other_mappings: Optional[List[Any]] = None,
    ir_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parser for LLVM IR (LLIR), which carries source locations in debug metadata
    rather than MLIR ``#loc`` directives.

    Args:
        ir_content: The LLIR content
        other_mappings: Other mappings (not used; LLVM DIFile nodes are absolute)
        ir_type: The IR type (not used, kept for signature consistency)

    Returns:
        Dictionary mapping line numbers to source locations
    """
    return extract_llvm_dbg_mappings(ir_content)


def _parse_ptx_loc(
    ir_content: str,
    other_mappings: Optional[List[Any]] = None,
    ir_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parser for PTX IR format.

    Args:
        ir_content: The PTX content
        other_mappings: Other mappings for file path resolution (from TTIR/TTGIR)
        ir_type: The IR type (not used, kept for signature consistency)

    Returns:
        Dictionary mapping line numbers to source locations
    """
    return extract_ptx_amdgcn_mappings(ir_content, other_mappings, "ptx")


def _parse_amdgcn_loc(
    ir_content: str,
    other_mappings: Optional[List[Any]] = None,
    ir_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parser for AMDGCN assembly format.

    Args:
        ir_content: The AMDGCN content
        other_mappings: Other mappings for file path resolution (from TTIR/TTGIR)
        ir_type: The IR type (not used, kept for signature consistency)

    Returns:
        Dictionary mapping line numbers to source locations
    """
    return extract_ptx_amdgcn_mappings(ir_content, other_mappings, "amdgcn")


def _parse_sass_loc(
    ir_content: str,
    other_mappings: Optional[List[Any]] = None,
    ir_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parser for NVIDIA SASS assembly format.

    Args:
        ir_content: The SASS content
        other_mappings: Other mappings (not used for SASS, kept for signature consistency)
        ir_type: The IR type (not used, kept for signature consistency)

    Returns:
        Dictionary mapping line numbers to source locations
    """
    return extract_sass_mappings(ir_content)


def _parse_none(
    ir_content: str,
    other_mappings: Optional[List[Any]] = None,
    ir_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Placeholder parser for stages that don't support source mapping (e.g., CUBIN).

    Args:
        ir_content: The content (not used)
        other_mappings: Other mappings (not used)
        ir_type: The IR type (not used)

    Returns:
        Empty dictionary
    """
    return {}
