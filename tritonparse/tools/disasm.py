#  Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import re
import subprocess

# Regex patterns for nvdisasm output
NVDISASM_FNAME_RE = re.compile(r"^\s*\.global\s+(\w+)")


def path_to_nvdisasm():
    from triton import knobs

    return knobs.nvidia.nvdisasm.path


def is_nvdisasm_available():
    try:
        if path_to_nvdisasm():
            return True
        else:
            return False
    except RuntimeError:
        return False


def extract(file_path):
    """Extract SASS from CUBIN using nvdisasm.

    nvdisasm output is much cleaner than cuobjdump:
    - Single line per instruction (no encoding lines)
    - Labels are already symbolized (.L_x_0 instead of addresses)
    - Source line information is included
    - No need for complex address remapping

    nvdisasm Documentation:
    https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"file not found: {file_path}")
    nvdisasm = path_to_nvdisasm()
    args = [nvdisasm, "-c", "-gp", "-g", "-gi", file_path]
    sass_str = subprocess.check_output(args)
    sass_lines = sass_str.splitlines()
    line_idx = 0

    while line_idx < len(sass_lines):
        line = sass_lines[line_idx].decode()

        # Find function definition (.global function_name)
        while NVDISASM_FNAME_RE.match(line) is None:
            line_idx += 1
            if line_idx >= len(sass_lines):
                return None
            line = sass_lines[line_idx].decode()

        # Extract function name
        match = NVDISASM_FNAME_RE.match(line)
        if match is None:
            return None
        fname = match.group(1)
        ret = f"Function:{fname}\n"

        # Find the actual start of function content (.text.kernel_name:)
        text_section_pattern = f".text.{fname}:"
        line_idx += 1
        while line_idx < len(sass_lines):
            line = sass_lines[line_idx].decode().strip()
            if line == text_section_pattern:
                line_idx += 1  # Move past the .text.kernel_name: line
                break
            line_idx += 1

        # Process all lines until next .headerflags or end of file
        while line_idx < len(sass_lines):
            line = sass_lines[line_idx].decode().rstrip()

            # Stop if we encounter next function's headerflags
            if line.strip().startswith(".headerflags"):
                break
            ret += line + "\n"
            line_idx += 1

        ret += "\n"
        return ret
