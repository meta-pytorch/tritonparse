from datetime import datetime
from pathlib import Path

from tritonparse.tp_logger import logger


def determine_output_paths(out_dir: str, kernel_name: str):
    """
    Determine output file paths for reproducer script and context data.

    Args:
        out_dir: Output directory path. If empty, uses default location.
        kernel_name: Name of the kernel for default directory naming.

    Returns:
        Tuple of (python_script_path, json_context_path) as Path objects.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_directory = Path(out_dir) / kernel_name
    output_directory.mkdir(parents=True, exist_ok=True)

    out_py_path = output_directory / f"repro_{timestamp}.py"
    temp_json_path = output_directory / f"repro_context_{timestamp}.json"

    return out_py_path, temp_json_path


def _generate_import_statements(kernel_info) -> tuple[str, str]:
    """
    Generate (sys.path insertion statement, import statement) for the kernel.

    Strategy:
    - Always add the kernel file's parent directory to sys.path.
    - If the filename (without .py) is a valid identifier, import using that
      module name: `from <stem> import <func> as imported_kernel_function`.
    - Otherwise, fall back to dynamic import via importlib.util and bind
      `imported_kernel_function` from the loaded module.
    """
    file_path = Path(kernel_info.file_path)
    function_name = kernel_info.function_name

    if not file_path or not function_name:
        raise ValueError("Kernel file path or function name missing from context.")

    # Always add the file's parent directory to sys.path
    sys_stmt = (
        "import sys; p = r'" + str(file_path.parent) + "';\n"
        "if p not in sys.path: sys.path.insert(0, p)"
    )

    module_name = file_path.with_suffix("").name
    if module_name.isidentifier():
        import_stmt = (
            f"from {module_name} import {function_name} as imported_kernel_function"
        )
        logger.info("Generated direct import statement: %s", import_stmt)
        return sys_stmt, import_stmt

    # Fallback: dynamic import when filename is not a valid identifier
    import_stmt = (
        "import importlib.util\n"
        f"_spec = importlib.util.spec_from_file_location('kernel_mod', r'{str(file_path)}')\n"
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_mod)\n"
        f"imported_kernel_function = getattr(_mod, '{function_name}')"
    )
    logger.info("Generated dynamic import for file: %s", file_path)
    return sys_stmt, import_stmt


def _parse_kernel_signature(kernel_source_code: str) -> tuple[list[str], list[str]]:
    """
    Parses a Triton kernel's source code to distinguish positional args
    from keyword args (those with default values).
    """
    signature_lines = []
    in_signature = False
    for line in kernel_source_code.splitlines():
        # Start capturing lines from 'def'
        if "def " in line:
            in_signature = True
        if in_signature:
            # Strip comments and leading/trailing whitespace
            clean_line = line.split("#")[0].strip()
            signature_lines.append(clean_line)
            # Stop capturing after the signature ends
            if "):" in line:
                break

    full_signature = "".join(signature_lines)
    # Extract content between the first '(' and the last '):'
    try:
        params_str = full_signature[
            full_signature.find("(") + 1 : full_signature.rfind("):")
        ]
    except IndexError as exc:
        raise ValueError("Could not parse kernel signature.") from exc

    # Clean up and split the parameters string
    params = [p.strip() for p in params_str.replace("\n", "").split(",") if p.strip()]

    positional_args = []
    keyword_args = []

    for param in params:
        if "=" in param:
            # Keyword arguments have a default value
            arg_name = param.split("=")[0].strip()
            keyword_args.append(arg_name)
        else:
            # Positional arguments do not have a default value
            arg_name = param.split(":")[0].strip()
            positional_args.append(arg_name)

    logger.debug("Parsed positional args: %s", positional_args)
    logger.debug("Parsed keyword args: %s", keyword_args)
    return positional_args, keyword_args
