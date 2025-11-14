#  Copyright (c) Meta Platforms, Inc. and affiliates.

import importlib
import importlib.util
import json
import logging
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import torch
from tritonparse.tools.load_tensor import load_tensor
from tritonparse.tp_logger import logger

TRITON_KERNELS_CUSTOM_TYPES = (
    importlib.util.find_spec("triton_kernels") is not None
    and importlib.util.find_spec("triton_kernels.tensor") is not None
)


@lru_cache(maxsize=1)
def _get_triton_tensor_types():
    mod = importlib.import_module("triton_kernels.tensor")
    return (
        mod.Tensor,
        mod.Storage,
        mod.StridedLayout,
    )


def create_args_from_json_file(json_path):
    """
    Load and parse a reproducer JSON file.

    Args:
        json_path (str): Path to the JSON file describing the kernel launch.

    Returns:
        tuple[list, dict]: Grid specification list and map of argument name to value.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return create_args_from_json(data)


def create_args_from_json(data):
    """
    Parse a reproducer JSON and build kernel grid and argument dictionary.

    Args:
        data (dict | list): JSON data describing the kernel launch.

    Returns:
        tuple[list, dict]: Grid specification list and map of argument name to value.
    """
    # Handle data format validation and extraction
    if isinstance(data, list):
        if len(data) != 1:
            print(
                f"Error: Expected single element list, got list with {len(data)} elements"
            )
            sys.exit(1)
        data = data[0]
    elif not isinstance(data, dict):
        print(f"Error: Expected list or dict, got {type(data)}")
        sys.exit(1)

    grid = data.get("grid", [])
    args_dict = {}
    extracted_args = data.get("extracted_args", {})

    for arg_name, arg_info in extracted_args.items():
        args_dict[arg_name] = _create_arg_from_info(arg_info)

    return grid, args_dict


def _apply_stride_and_offset(tensor, shape, stride, storage_offset):
    """
    Apply custom stride and storage offset to a tensor if needed.

    Args:
        tensor: The base contiguous tensor
        shape: The desired shape
        stride: The desired stride (or None for contiguous)
        storage_offset: The desired storage offset

    Returns:
        torch.Tensor: The strided tensor view or original tensor if contiguous
    """
    if stride is None:
        return tensor

    # Calculate expected contiguous stride
    expected_contiguous_stride = []
    s = 1
    for dim_size in reversed(shape):
        expected_contiguous_stride.insert(0, s)
        s *= dim_size

    # If stride matches contiguous stride and no storage offset, return as-is
    if tuple(stride) == tuple(expected_contiguous_stride) and storage_offset == 0:
        return tensor

    # Calculate required storage size
    if len(shape) > 0 and len(stride) > 0:
        max_offset = storage_offset
        for dim_stride, dim_size in zip(stride, shape):
            if dim_size > 0:
                max_offset += dim_stride * (dim_size - 1)
        storage_size = max_offset + 1
    else:
        storage_size = storage_offset + 1

    # Create larger storage tensor and create strided view
    storage_tensor = torch.empty(storage_size, dtype=tensor.dtype, device=tensor.device)

    # Create strided view
    strided_view = storage_tensor.as_strided(
        size=shape, stride=stride, storage_offset=storage_offset
    )

    # Copy data from the base tensor into the strided layout
    strided_view.copy_(tensor.flatten()[: strided_view.numel()].view(shape))

    return strided_view


def _create_base_tensor(arg_info) -> torch.Tensor:
    """
    Create a base tensor without stride/offset modifications.

    Args:
        arg_info (dict): Argument information including dtype, shape, device, etc.

    Returns:
        torch.Tensor: The created base tensor
    """
    if arg_info.get("blob_path"):
        return load_tensor(arg_info.get("blob_path"), arg_info.get("device"))

    # Extract basic tensor properties
    dtype_str = arg_info.get("dtype")
    try:
        torch_dtype = getattr(torch, dtype_str.split(".")[-1])
    except AttributeError:
        logging.error(f"Unsupported dtype: {dtype_str}. Defaulting to float32.")
        torch_dtype = torch.float32

    shape = arg_info.get("shape", [])
    device = arg_info.get("device", "cpu")
    # Normalize cuda device to cuda:0
    if isinstance(device, str) and device.startswith("cuda"):
        device = "cuda:0"

    # Extract statistical information if available
    mean = arg_info.get("mean")
    std = arg_info.get("std")
    min_val = arg_info.get("min")
    max_val = arg_info.get("max")
    has_stats = (
        mean is not None
        and std is not None
        and min_val is not None
        and max_val is not None
    )

    if arg_info.get("tensor_capture_error", False):
        logging.error(
            f"Error: Tensor '{arg_info.get('name', '')}' had capture error. Generating random tensor instead."
        )

    # Use a dummy tensor to check properties of the dtype
    tensor_props = torch.empty(0, dtype=torch_dtype)

    # Case 1: Floating point types
    if tensor_props.is_floating_point():
        if has_stats:
            # Generate tensor with statistical properties matching original data
            if std == 0 or min_val == max_val:
                # Constant tensor
                return torch.full(shape, mean, dtype=torch_dtype, device=device)
            # Generate normal distribution with mean and std, then clamp to [min, max]
            tensor = torch.randn(shape, dtype=torch.float32, device=device) * std + mean
            tensor = torch.clamp(tensor, min=min_val, max=max_val)
            return tensor.to(torch_dtype)
        else:
            # Fallback to original random generation
            if torch_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                tmp = torch.rand(shape, dtype=torch.float32, device=device)
                return tmp.to(torch_dtype)
            else:
                return torch.empty(shape, dtype=torch_dtype, device=device).random_()

    # Case 2: Integer types
    elif torch_dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ]:
        if has_stats and torch_dtype != torch.bool:
            # Generate tensor with statistical properties, then round for integers
            if std == 0 or min_val == max_val:
                # Constant tensor
                return torch.full(shape, int(mean), dtype=torch_dtype, device=device)
            tensor = torch.randn(shape, dtype=torch.float32, device=device) * std + mean
            tensor = torch.clamp(tensor, min=min_val, max=max_val)
            return torch.round(tensor).to(torch_dtype)
        else:
            # Fallback to original random generation
            return torch.empty(shape, dtype=torch_dtype, device=device).random_()

    # Case 3: Complex numbers need special handling
    elif tensor_props.is_complex():
        # Complex types: fallback to original logic for now
        # TODO: Could be improved to use statistical info if available
        float_dtype = torch.float32 if torch_dtype == torch.complex64 else torch.float64
        real_part = torch.rand(shape, dtype=float_dtype, device=device)
        imag_part = torch.rand(shape, dtype=float_dtype, device=device)
        return torch.complex(real_part, imag_part)

    # Case 4: Handle other unsigned integers (like uint32) which fail with random_()
    elif "uint" in str(torch_dtype):
        if has_stats:
            # Generate tensor with statistical properties for unsigned integers
            if std == 0 or min_val == max_val:
                return torch.full(shape, int(mean), dtype=torch_dtype, device=device)
            tensor = torch.randn(shape, dtype=torch.float32, device=device) * std + mean
            tensor = torch.clamp(tensor, min=min_val, max=max_val)
            return torch.round(tensor).to(torch_dtype)
        else:
            # Fallback to original random generation
            return torch.randint(0, 1000, shape, dtype=torch_dtype, device=device)

    # Case 5: If we don't know how to handle the type, raise an error
    else:
        raise NotImplementedError(
            f"Random data generation not implemented for dtype: {torch_dtype}"
        )


def _create_tensor(arg_info) -> torch.Tensor:
    """
    Create a tensor with stride and storage offset if needed.

    Args:
        arg_info (dict): Argument information including dtype, shape, stride, etc.

    Returns:
        torch.Tensor: The created tensor with applied stride/offset
    """
    tensor = _create_base_tensor(arg_info)

    # Apply stride and storage offset if needed
    shape = arg_info.get("shape", [])
    stride = arg_info.get("stride")
    storage_offset = arg_info.get("storage_offset", 0)
    return _apply_stride_and_offset(tensor, shape, stride, storage_offset)


def _create_arg_from_info(arg_info):
    """
    Recursively construct a kernel argument from its JSON schema.

    Args:
        arg_info (dict): JSON object describing a single argument, including
            fields like 'type', 'value', 'dtype', 'shape', 'device', etc.

    Returns:
        Any: The constructed Python object suitable for kernel invocation.

    Raises:
        RuntimeError: When required optional dependencies are missing.
        NotImplementedError: When a dtype or type is not supported yet.
    """
    arg_type = arg_info.get("type")

    if arg_type == "NoneType":
        return None

    if arg_type in ["int", "bool", "str", "float"]:
        return arg_info.get("value")

    elif arg_type == "tensor":
        return _create_tensor(arg_info)

    elif arg_type == "triton_kernels.tensor.Tensor":
        if not TRITON_KERNELS_CUSTOM_TYPES:
            raise RuntimeError(
                "Optional dependency 'triton_kernels.tensor' is not installed; cannot construct Tensor."
            )
        Tensor, Storage, StridedLayout = _get_triton_tensor_types()
        storage = _create_arg_from_info(arg_info.get("storage"))
        dtype_str = arg_info.get("dtype")
        torch_dtype = getattr(torch, dtype_str.split(".")[-1])
        return Tensor(
            storage=storage,
            shape=arg_info.get("shape"),
            shape_max=arg_info.get("shape_max"),
            dtype=torch_dtype,
        )

    elif arg_type == "triton_kernels.tensor.Storage":
        if not TRITON_KERNELS_CUSTOM_TYPES:
            raise RuntimeError(
                "Optional dependency 'triton_kernels.tensor' is not installed; cannot construct Storage."
            )
        Tensor, Storage, StridedLayout = _get_triton_tensor_types()
        data = _create_arg_from_info(arg_info.get("data"))
        layout = _create_arg_from_info(arg_info.get("layout"))
        return Storage(data=data, layout=layout)

    elif arg_type == "StridedLayout":
        if not TRITON_KERNELS_CUSTOM_TYPES:
            raise RuntimeError(
                "Optional dependency 'triton_kernels.tensor' is not installed; cannot construct StridedLayout."
            )
        Tensor, Storage, StridedLayout = _get_triton_tensor_types()
        return StridedLayout(shape=arg_info.get("initial_shape"))
    else:
        print(f"Warning: Unhandled argument type '{arg_type}'. Returning None.")
        return None


def determine_output_paths(out_dir: str, kernel_name: str, template: str):
    """
    Determine output file paths for reproducer script and context data.

    Args:
        out_dir: Output directory path. If empty, uses default location.
        kernel_name: Name of the kernel for default directory naming.
        template: Template name or path. If a path, extracts the filename.

    Returns:
        Tuple of (python_script_path, json_context_path) as Path objects.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_directory = Path(out_dir) / kernel_name
    output_directory.mkdir(parents=True, exist_ok=True)

    # Extract template name from path if needed
    template_name = (
        Path(template).stem if "/" in template or "\\" in template else template
    )

    filename_parts = ["repro"]
    if template != "example":
        filename_parts.append(template_name)
    filename_parts.append(timestamp)
    filename = "_".join(filename_parts) + ".py"
    out_py_path = output_directory / filename
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
        logger.debug("Generated direct import statement: %s", import_stmt)
        return sys_stmt, import_stmt

    # Fallback: dynamic import when filename is not a valid identifier
    import_stmt = (
        "import importlib.util\n"
        f"_spec = importlib.util.spec_from_file_location('kernel_mod', r'{str(file_path)}')\n"
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_mod)\n"
        f"imported_kernel_function = getattr(_mod, '{function_name}')"
    )
    logger.debug("Generated dynamic import for file: %s", file_path)
    return sys_stmt, import_stmt


def _parse_kernel_signature(kernel_source_code: str) -> tuple[list[str], list[str]]:
    """
    Parses a Triton kernel's source code to distinguish positional args
    from keyword args (those with default values).
    """
    signature_lines = []
    in_signature = False
    for line in kernel_source_code.splitlines():
        # Mark beginning of signature when function definition is found
        if line.strip().startswith("def "):
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


def _generate_invocation_snippet(
    positional_args: list[str], keyword_args: list[str]
) -> str:
    """Generates a single-line Python code snippet for kernel invocation."""
    # Prepare positional args for direct injection into the call
    pos_args_str = ", ".join([f'args_dict["{arg}"]' for arg in positional_args])

    # Prepare keyword args for direct injection
    kw_args_str = ", ".join([f'{arg}=args_dict["{arg}"]' for arg in keyword_args])

    # Combine them, ensuring proper comma separation
    all_args = []
    if pos_args_str:
        all_args.append(pos_args_str)
    if kw_args_str:
        all_args.append(kw_args_str)

    # Create the single-line call
    return f"imported_kernel_function[tuple(grid)]({', '.join(all_args)})"
