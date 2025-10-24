#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Function extractor for reproducer utility functions.

This module extracts utility functions from utils.py and load_tensor.py
using AST parsing, and generates standalone code for reproducers.
"""

import ast
from pathlib import Path


def extract_utility_functions() -> str:
    """
    Extract all utility functions needed for the reproducer template.

    Uses AST parsing to extract functions and constants from source files
    without importing them (avoiding potential side effects).

    Returns:
        str: Complete Python code including imports and all utility functions.
    """
    # Prepare file paths
    base_dir = Path(__file__).parent
    utils_path = base_dir / "utils.py"
    load_tensor_path = base_dir.parent / "tools" / "load_tensor.py"

    # Parse source files
    utils_tree, utils_lines = _parse_source_file(utils_path)
    load_tensor_tree, load_tensor_lines = _parse_source_file(load_tensor_path)

    # Define what to extract (in dependency order)
    utils_function_names = [
        "_get_triton_tensor_types",
        "create_args_from_json_file",
        "create_args_from_json",
        "_apply_stride_and_offset",
        "_create_base_tensor",
        "_create_tensor",
        "_create_arg_from_info",
    ]

    load_tensor_function_names = [
        "load_tensor",
    ]

    # Extract content
    extracted_parts = []

    # Add required imports
    extracted_parts.append(_generate_imports())

    # Extract constant
    constant = _extract_assignment(
        utils_tree, utils_lines, "TRITON_KERNELS_CUSTOM_TYPES"
    )
    if constant:
        extracted_parts.append(constant)

    # Extract load_tensor functions
    extracted_parts.extend(
        _extract_functions(
            load_tensor_tree, load_tensor_lines, load_tensor_function_names
        )
    )

    # Extract utils functions
    extracted_parts.extend(
        _extract_functions(utils_tree, utils_lines, utils_function_names)
    )

    # Combine all parts
    return "\n\n".join(extracted_parts)


def _parse_source_file(file_path: Path) -> tuple[ast.Module, list[str]]:
    """
    Parse a Python source file and return its AST and source lines.

    Args:
        file_path: Path to the Python source file

    Returns:
        tuple: (AST tree, list of source code lines)

    Raises:
        FileNotFoundError: If the source file doesn't exist
        SyntaxError: If the source file has syntax errors
    """
    try:
        source_code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source_code, filename=str(file_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Source file not found: {file_path}") from e
    except SyntaxError as e:
        raise SyntaxError(f"Failed to parse {file_path}: {e}") from e

    lines = source_code.splitlines()
    return tree, lines


def _extract_assignment(
    tree: ast.Module, lines: list[str], var_name: str
) -> str | None:
    """
    Extract a module-level assignment statement by variable name.

    Args:
        tree: AST tree of the source file
        lines: Source code lines
        var_name: Name of the variable to extract

    Returns:
        Complete assignment statement source code, or None if not found

    Example:
        Extracts:
        TRITON_KERNELS_CUSTOM_TYPES = (
            importlib.util.find_spec("triton_kernels") is not None
            and importlib.util.find_spec("triton_kernels.tensor") is not None
        )
    """
    # Search only at module level
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    # Found it! Extract source code using line numbers
                    start_line = node.lineno - 1  # Convert to 0-based index
                    end_line = node.end_lineno  # Already suitable for slicing
                    assignment_lines = lines[start_line:end_line]
                    return "\n".join(assignment_lines)
    return None


def _extract_function(tree: ast.Module, lines: list[str], func_name: str) -> str | None:
    """
    Extract a function definition by name, including decorators.

    Args:
        tree: AST tree of the source file
        lines: Source code lines
        func_name: Name of the function to extract

    Returns:
        Complete function source code including decorators, or None if not found

    Example:
        Extracts:
        @lru_cache(maxsize=1)
        def _get_triton_tensor_types():
            '''Docstring'''
            ...
    """
    # Walk the entire tree (handles nested functions if needed)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # If function has decorators, start from the first decorator
            if node.decorator_list:
                start_line = node.decorator_list[0].lineno - 1
            else:
                start_line = node.lineno - 1

            end_line = node.end_lineno
            func_lines = lines[start_line:end_line]
            return "\n".join(func_lines)
    return None


def _extract_functions(
    tree: ast.Module, lines: list[str], func_names: list[str]
) -> list[str]:
    """
    Extract multiple functions from a source file.

    Args:
        tree: AST tree of the source file
        lines: Source code lines
        func_names: List of function names to extract

    Returns:
        List of function source codes in the same order as func_names

    Raises:
        ValueError: If any function is not found
    """
    extracted = []
    for func_name in func_names:
        func_source = _extract_function(tree, lines, func_name)
        if func_source is None:
            raise ValueError(
                f"Function '{func_name}' not found in source. "
                f"Available functions might have been renamed or removed."
            )
        extracted.append(func_source)
    return extracted


def _generate_imports() -> str:
    """
    Generate the import statements needed for the extracted functions.

    Returns:
        str: Import statements as a single string
    """
    imports = [
        "import gzip",
        "import hashlib",
        "import importlib",
        "import importlib.util",
        "import io",
        "import json",
        "import logging",
        "import sys",
        "from functools import lru_cache",
        "from pathlib import Path",
        "from typing import Union",
        "",
        "import torch",
    ]
    return "\n".join(imports)
