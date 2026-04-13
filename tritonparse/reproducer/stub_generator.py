#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Generate stub @triton.jit functions for IR-based reproducers.

A stub kernel has the same function name and parameter signature as the original
kernel but a trivial body (``pass``).  When used with ``ir_override`` on a
``triton.Config``, the stub compiles through the frontend (producing a valid but
trivial TTIR) and the compilation output is replaced by the captured IR from the
original kernel.  This completely eliminates the need to copy the original
kernel's source code and its transitive dependencies.
"""

import ast
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from tritonparse.reproducer.function_extractor import _is_constexpr_annotation
from tritonparse.reproducer.ingestion.ndjson import ContextBundle
from tritonparse.tp_logger import logger


def generate_stub_source(kernel_name: str, source_code: str) -> str:
    """Generate a stub ``@triton.jit`` function from the original kernel source.

    Parses the original source to extract parameter names and ``tl.constexpr``
    annotations, then produces a stub with the same signature and a ``pass``
    body.  Does NOT include ``@triton.autotune`` — that is added separately
    by the placeholder replacer.

    Args:
        kernel_name: The kernel function name.
        source_code: The original kernel source code (from ``@triton.jit`` onward).

    Returns:
        Python source code defining the stub function.
    """
    params = extract_params_from_source(source_code)

    # Build parameter list
    param_strs: List[str] = []
    for name, is_constexpr in params:
        if is_constexpr:
            param_strs.append(f"{name}: tl.constexpr")
        else:
            param_strs.append(name)

    # Format as multi-line if many params
    if len(param_strs) > 3:
        params_block = "\n    " + ",\n    ".join(param_strs) + ",\n"
    else:
        params_block = ", ".join(param_strs)

    return f"@triton.jit\ndef {kernel_name}({params_block}):\n    pass"


def _find_ir_override_file(ir_dir: str) -> "str | None":
    """Find the captured TTIR file for ir_override.

    Returns the full path to the first ``.ttir`` file, or None.

    NOTE: This function is extracted by ``get_function_source()`` and embedded
    verbatim in generated reproducer scripts.  Keep imports minimal (only
    ``os`` and ``logging`` are available).
    """
    import logging
    import os

    for name in sorted(os.listdir(ir_dir)):
        if name.endswith(".ttir"):
            return os.path.join(ir_dir, name)
    logging.getLogger(__name__).warning(
        "captured_irs/ directory exists but contains no .ttir files. "
        "Stub kernel will run without ir_override (fallback mode)."
    )
    return None


def get_constexpr_values(context_bundle: ContextBundle) -> Dict[str, Any]:
    """Extract constexpr parameter names and their values from the context.

    Args:
        context_bundle: The context bundle from the captured launch event.

    Returns:
        Dictionary mapping constexpr parameter name to its value.
    """
    source_code = context_bundle.kernel_info.source_code
    params = extract_params_from_source(source_code)
    constexpr_names = {name for name, is_ce in params if is_ce}
    args = context_bundle.args

    result: Dict[str, Any] = {}
    for name in constexpr_names:
        arg_info = args.get(name)
        if arg_info is None:
            continue
        if isinstance(arg_info, dict):
            result[name] = arg_info.get("value")
        else:
            result[name] = arg_info

    return result


@lru_cache(maxsize=8)
def extract_params_from_source(
    source_code: str,
) -> Tuple[Tuple[str, bool], ...]:
    """Extract parameter names and constexpr annotations from kernel source.

    Results are cached since the same source is parsed multiple times during
    reproducer generation (stub generation, constexpr extraction, invocation).

    Returns:
        Tuple of ``(param_name, is_constexpr)`` pairs.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        logger.warning("Failed to parse kernel source for param extraction")
        return ()

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        params: List[Tuple[str, bool]] = []
        all_args = list(node.args.args) + list(node.args.kwonlyargs)
        for arg in all_args:
            is_ce = arg.annotation is not None and _is_constexpr_annotation(
                arg.annotation
            )
            params.append((arg.arg, is_ce))
        return tuple(params)

    return ()
