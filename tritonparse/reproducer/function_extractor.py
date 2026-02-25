#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Function extractor for reproducer utility functions.

This module extracts utility functions from utils.py and load_tensor.py
using AST parsing, and generates standalone code for reproducers.

It also provides utilities for extracting kernel functions with their
decorators (e.g., @triton.autotune) from source files.
"""

import ast
import importlib.resources
from pathlib import Path
from typing import Optional

from tritonparse.reproducer.utils import TRITON_COMPILE_PARAMS


def _read_source_from_package(package: str, resource: str) -> str:
    """
    Read source code from a package resource.

    Works in both normal filesystem and PAR (Python Archive) environments.

    Args:
        package: The package name (e.g., "tritonparse.reproducer")
        resource: The resource filename (e.g., "utils.py")

    Returns:
        str: The source code content
    """
    try:
        # Python 3.9+ API
        files = importlib.resources.files(package)
        return (files / resource).read_text(encoding="utf-8")
    except (TypeError, AttributeError):
        # Fallback for older Python versions
        return importlib.resources.read_text(package, resource)


def extract_utility_functions(embed_context: bool = False) -> str:
    """
    Extract all utility functions needed for the reproducer template.

    Uses AST parsing to extract functions and constants from source files
    without importing them (avoiding potential side effects).

    Args:
        embed_context: If True, exclude file-based loading functions
            (create_args_from_json_file) since the context is embedded
            directly in the script. Default: False.

    Returns:
        str: Complete Python code including imports and all utility functions.
    """
    # Read source files using importlib.resources (works in PAR environments)
    utils_source = _read_source_from_package("tritonparse.reproducer", "utils.py")
    load_tensor_source = _read_source_from_package(
        "tritonparse.tools", "load_tensor.py"
    )

    # Parse source files
    utils_tree, utils_lines = _parse_source_code(utils_source, "utils.py")
    load_tensor_tree, load_tensor_lines = _parse_source_code(
        load_tensor_source, "load_tensor.py"
    )

    # Define what to extract (in dependency order)
    # When embed_context=True, we don't need create_args_from_json_file
    # since the JSON is embedded directly in the script
    utils_function_names = [
        "_get_triton_tensor_types",
        "_get_tensor_descriptor_type",
        "create_args_from_json",
        "_apply_stride_and_offset",
        "_create_base_tensor",
        "_create_tensor",
        "_create_arg_from_info",
    ]

    # Only include file-based loading function when not embedding context
    if not embed_context:
        utils_function_names.insert(1, "create_args_from_json_file")

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

    # Extract TRITON_DTYPE_MAP constant
    dtype_map = _extract_assignment(utils_tree, utils_lines, "TRITON_DTYPE_MAP")
    if dtype_map:
        extracted_parts.append(dtype_map)

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


def _parse_source_code(
    source_code: str, filename: str = "<string>"
) -> tuple[ast.Module, list[str]]:
    """
    Parse Python source code and return its AST and source lines.

    Args:
        source_code: Python source code as a string
        filename: Filename to use for error messages (default: "<string>")

    Returns:
        tuple: (AST tree, list of source code lines)

    Raises:
        SyntaxError: If the source code has syntax errors
    """
    try:
        tree = ast.parse(source_code, filename=filename)
    except SyntaxError as e:
        raise SyntaxError(f"Failed to parse {filename}: {e}") from e

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
        "import os",
        "import sys",
        "from functools import lru_cache",
        "from pathlib import Path",
        "from typing import Union",
        "",
        "import torch",
        "import triton.language as tl",
    ]
    return "\n".join(imports)


def extract_function_with_decorators(
    file_path: str,
    function_name: str,
    source_repo_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Extract a function's source code including decorators from a file.

    Resolves production paths to local repo paths if source_repo_dir is provided.

    Args:
        file_path: Path to the source file (may be a production path).
        function_name: Name of the function to extract.
        source_repo_dir: Optional repo directory for path resolution.

    Returns:
        Function source with decorators, or None if not found.
    """
    source_path = Path(file_path)

    # Try to resolve the path if it doesn't exist
    if not source_path.exists() and source_repo_dir:
        source_repo_path = Path(source_repo_dir)
        if source_repo_path.exists():
            # Try progressively shorter path suffixes
            for i in range(1, len(source_path.parts)):
                new_path = source_repo_path / Path("/".join(source_path.parts[i:]))
                if new_path.exists():
                    source_path = new_path
                    break

    if not source_path.exists():
        return None

    try:
        source_content = source_path.read_text()
        tree, lines = _parse_source_code(source_content, str(source_path))
        return _extract_function(tree, lines, function_name)
    except Exception:
        return None


def extract_autotune_config_params(source_code: str) -> set[str]:
    """
    Extract parameter names provided by @triton.autotune configs.

    Finds all triton.Config(...) calls in the source and extracts the dict keys,
    which are the parameters the autotuner provides at runtime.

    Handles patterns like:
    - Direct dict literals: triton.Config({"BLOCK_M": 64, ...})
    - Variable references: config_kwargs = {...}; triton.Config(config_kwargs, ...)

    Args:
        source_code: Kernel source including decorators and any config helpers.

    Returns:
        Set of autotune-provided parameter names, or empty set if none found.
    """
    params: set[str] = set()

    try:
        tree, _ = _parse_source_code(source_code)
    except SyntaxError:
        return params

    # Build variable bindings for dict literals
    var_bindings = _build_variable_bindings(tree)

    for node in ast.walk(tree):
        # Look for triton.Config(...) calls
        if isinstance(node, ast.Call):
            if _is_triton_config_call(node):
                # Extract keys from dict argument (first positional or keyword args)
                params.update(_extract_config_dict_keys(node, var_bindings))

    return params


def _build_variable_bindings(tree: ast.AST) -> dict[str, ast.Dict]:
    """Build a mapping from variable names to dict literals.

    Scans the AST for simple assignments like:
        config_kwargs = {"BLOCK_M": 64, ...}

    This enables resolving variable references in triton.Config(config_kwargs, ...).

    Args:
        tree: The parsed AST.

    Returns:
        Dict mapping variable names to their ast.Dict values.
    """
    bindings: dict[str, ast.Dict] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # Handle: var = {...}
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Dict)
            ):
                bindings[node.targets[0].id] = node.value
    return bindings


def _is_triton_config_call(node: ast.Call) -> bool:
    """Check if an AST Call node is a triton.Config(...) call."""
    func = node.func
    # Handle: triton.Config(...)
    if isinstance(func, ast.Attribute) and func.attr == "Config":
        if isinstance(func.value, ast.Name) and func.value.id == "triton":
            return True
    # Handle: Config(...) when imported directly
    if isinstance(func, ast.Name) and func.id == "Config":
        return True
    return False


# Triton.Config-specific kwargs that aren't kernel params
# These are compile/runtime parameters that triton.Config accepts but are not
# user-defined kernel config parameters.
# Reference: triton.Config class in triton/runtime/autotuner.py
# https://github.com/triton-lang/triton/blob/main/python/triton/runtime/autotuner.py
_TRITON_CONFIG_KWARGS = set(TRITON_COMPILE_PARAMS) | {
    "pre_hook",
    # Warp specialization parameters that may appear in some Triton versions
    "minRegAutoWS",
    "maxRegAutoWS",
    "pingpongAutoWS",
}


def _extract_config_dict_keys(
    node: ast.Call, var_bindings: dict[str, ast.Dict]
) -> set[str]:
    """Extract keys from the dict argument of a triton.Config call.

    Handles direct dict literals and variable references to dict literals.

    Args:
        node: The AST Call node for triton.Config(...)
        var_bindings: Mapping from variable names to their dict literal values.

    Returns:
        Set of config parameter names.
    """
    keys: set[str] = set()

    # Check first positional argument
    if node.args:
        first_arg = node.args[0]
        # Handle direct dict literal
        if isinstance(first_arg, ast.Dict):
            keys.update(_extract_dict_keys(first_arg))
        # Handle variable reference to a dict
        elif isinstance(first_arg, ast.Name) and first_arg.id in var_bindings:
            keys.update(_extract_dict_keys(var_bindings[first_arg.id]))

    # Also check keyword arguments passed directly to Config
    for keyword in node.keywords:
        if keyword.arg is not None:
            # Skip triton.Config-specific kwargs that aren't kernel params
            if keyword.arg not in _TRITON_CONFIG_KWARGS:
                keys.add(keyword.arg)

    return keys


def _extract_dict_keys(dict_node: ast.Dict) -> set[str]:
    """Extract keys from an AST Dict node.

    Handles both quoted string keys ('BLOCK_SIZE') and unquoted identifier
    keys (BLOCK_SIZE) which are valid Python dict syntax.
    """
    keys: set[str] = set()
    for key in dict_node.keys:
        if key is not None:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
            elif isinstance(key, ast.Str):  # Python 3.7 compatibility
                keys.add(key.s)
            elif isinstance(key, ast.Name):
                # Handle unquoted identifier keys like {BLOCK_SIZE: 128}
                keys.add(key.id)
    return keys


def _is_autotune_decorator(node: ast.expr) -> bool:
    """Check if a decorator node is @triton.autotune(...) or @autotune(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    # Handle: @triton.autotune(...)
    if isinstance(func, ast.Attribute) and func.attr == "autotune":
        if isinstance(func.value, ast.Name) and func.value.id == "triton":
            return True
    # Handle: @autotune(...) when imported directly
    if isinstance(func, ast.Name) and func.id == "autotune":
        return True
    return False


def _get_configs_kwarg(node: ast.Call) -> ast.expr | None:
    """Extract the 'configs' argument AST node from a triton.autotune(...) call."""
    for kw in node.keywords:
        if kw.arg == "configs":
            return kw.value
    # Also check first positional argument
    if node.args:
        return node.args[0]
    return None


def _extract_config_values(
    call_node: ast.Call,
    var_bindings: dict[str, ast.Dict],
) -> dict[str, object]:
    """Extract all key-value pairs from a single triton.Config(...) call.

    Extracts both:
    - Dict params (first arg): {"BLOCK_M": 256, "BLOCK_N": 128, ...}
    - Compile params (keyword args): num_warps=4, num_stages=0, ...

    Non-constant values (e.g., pre_hook=func_ref) are skipped.
    """
    config: dict[str, object] = {}

    # Extract from first positional argument (dict)
    if call_node.args:
        first_arg = call_node.args[0]
        dict_node = None
        if isinstance(first_arg, ast.Dict):
            dict_node = first_arg
        elif isinstance(first_arg, ast.Name) and first_arg.id in var_bindings:
            dict_node = var_bindings[first_arg.id]

        if dict_node is not None:
            for key, val in zip(dict_node.keys, dict_node.values):
                k = None
                if key is not None:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        k = key.value
                    elif isinstance(key, ast.Name):
                        k = key.id
                if k is not None and isinstance(val, ast.Constant):
                    config[k] = val.value

    # Extract from keyword args (num_warps=4, num_stages=0, etc.)
    for kw in call_node.keywords:
        if kw.arg is not None and isinstance(kw.value, ast.Constant):
            config[kw.arg] = kw.value.value

    return config


def extract_kernel_autotune_configs(
    source_code: str,
    function_name: str,
) -> list[dict[str, object]] | None:
    """Extract compile param values from a kernel's @triton.autotune configs.

    Traces from the kernel's decorator to the exact configs it references,
    isolating from other kernels' configs in the same file.

    Handles three patterns for the configs= argument:
    - ast.Name (variable reference): configs=configs_var → traces to variable assignment
    - ast.List (inline list): configs=[triton.Config(...), ...] → parses directly
    - ast.Call (function call): configs=get_configs() → returns None (cannot resolve)

    Args:
        source_code: Full source file content.
        function_name: Target kernel function name.

    Returns:
        List of config dicts (each containing dict params + compile params),
        or None if extraction fails.
    """
    try:
        tree, _ = _parse_source_code(source_code)
    except SyntaxError:
        return None

    # Step 1: Locate the target function
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            func_node = node
            break
    if func_node is None:
        return None

    # Step 2: Find @triton.autotune decorator, extract configs= argument
    configs_ast_node = None
    for dec in func_node.decorator_list:
        if _is_autotune_decorator(dec):
            configs_ast_node = _get_configs_kwarg(dec)
            break
    if configs_ast_node is None:
        return None

    # Step 3: Resolve configs AST node to a list of triton.Config calls
    config_call_nodes: list[ast.Call] = []
    if isinstance(configs_ast_node, ast.List):
        # Inline: configs=[triton.Config(...), ...]
        config_call_nodes = [
            elt
            for elt in configs_ast_node.elts
            if isinstance(elt, ast.Call) and _is_triton_config_call(elt)
        ]
    elif isinstance(configs_ast_node, ast.Name):
        # Variable reference: configs=configs_var
        # Collect ALL triton.Config calls from ALL assignments to this variable.
        # Multiple assignments may exist (e.g., different config sets for different
        # kernels reusing the same variable name, or a cleanup `configs = []`).
        # We collect broadly here and let match_autotune_config disambiguate
        # using extracted_args.
        var_name = configs_ast_node.id
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if isinstance(stmt.value, ast.List):
                            config_call_nodes.extend(
                                elt
                                for elt in stmt.value.elts
                                if isinstance(elt, ast.Call)
                                and _is_triton_config_call(elt)
                            )
    elif isinstance(configs_ast_node, ast.Call):
        # Dynamic: configs=get_fwd_configs() — cannot resolve statically
        return None
    else:
        return None

    if not config_call_nodes:
        return None

    # Step 4: Extract key-value pairs from each Config
    var_bindings = _build_variable_bindings(tree)
    configs = []
    for call_node in config_call_nodes:
        config = _extract_config_values(call_node, var_bindings)
        if config:
            configs.append(config)

    return configs if configs else None


def match_autotune_config(
    configs: list[dict[str, object]],
    extracted_args: dict[str, object],
) -> dict[str, object] | None:
    """Match the selected autotune config and return its compile params.

    Strategy:
    1. If all configs agree on every compile param value, use them directly.
    2. Otherwise, match by comparing config dict params against extracted_args.

    Args:
        configs: List of config dicts from extract_kernel_autotune_configs.
        extracted_args: Extracted argument info from launch event.
            Format: {"BLOCK_M": {"type": "int", "value": 256}, ...}

    Returns:
        Dict of compile params (e.g., {"num_warps": 4, "num_stages": 0}),
        or None if matching fails.
    """
    if not configs:
        return None

    # Strategy 1: Check if all configs agree on compile params
    compile_params_result: dict[str, object] = {}
    all_agree = True
    for param in TRITON_COMPILE_PARAMS:
        values = {c[param] for c in configs if param in c}
        if len(values) == 1:
            compile_params_result[param] = values.pop()
        elif len(values) > 1:
            all_agree = False
            break

    if all_agree and compile_params_result:
        return compile_params_result

    # Strategy 2: Match by config dict params against extracted_args
    for config in configs:
        dict_params = {
            k: v
            for k, v in config.items()
            if k not in TRITON_COMPILE_PARAMS and k not in _TRITON_CONFIG_KWARGS
        }
        if not dict_params:
            continue
        match = all(
            (
                isinstance(extracted_args.get(k), dict)
                and extracted_args[k].get("value") == v
            )
            for k, v in dict_params.items()
        )
        if match:
            return {k: v for k, v in config.items() if k in TRITON_COMPILE_PARAMS}

    return None


def _is_constexpr_annotation(annotation: ast.expr) -> bool:
    """Check if an annotation is tl.constexpr or triton.language.constexpr."""
    if isinstance(annotation, ast.Attribute) and annotation.attr == "constexpr":
        if isinstance(annotation.value, ast.Name):
            return annotation.value.id in ("tl", "triton")
        if isinstance(annotation.value, ast.Attribute):
            return annotation.value.attr == "language"
    return False


def is_constexpr_param(param_name: str, source_code: str) -> bool:
    """
    Check if a parameter is annotated as tl.constexpr in the kernel signature.

    Args:
        param_name: Name of the parameter to check.
        source_code: Kernel source code containing the function signature.

    Returns:
        True if the parameter has a `: tl.constexpr` annotation, False otherwise.

    Example:
        >>> source = "def kernel(a_ptr, BLOCK_SIZE: tl.constexpr): pass"
        >>> is_constexpr_param("BLOCK_SIZE", source)
        True
        >>> is_constexpr_param("a_ptr", source)
        False
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                if arg.arg == param_name and arg.annotation is not None:
                    if _is_constexpr_annotation(arg.annotation):
                        return True
    return False
