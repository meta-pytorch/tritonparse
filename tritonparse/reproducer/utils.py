import importlib
import importlib.util
import json
import sys
from functools import lru_cache

import torch

TRITON_KERNELS_CUSTOM_TYPES = (
    importlib.util.find_spec("triton_kernels.tensor") is not None
)


def create_args_from_json(json_path):
    """
    Creates a list of arguments for a kernel launch from a JSON file.

    Args:
        json_path (str): The path to the JSON file containing the kernel
                         launch information.

    Returns:
        tuple: A tuple containing the grid and a dictionary of arguments.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
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


@lru_cache(maxsize=1)
def _get_triton_tensor_types():
    mod = importlib.import_module("triton_kernels.tensor")
    return (
        getattr(mod, "Tensor"),
        getattr(mod, "Storage"),
        getattr(mod, "StridedLayout"),
    )


def _create_arg_from_info(arg_info):
    """
    Recursively creates a kernel argument from its JSON info dictionary.
    """
    arg_type = arg_info.get("type")

    if arg_type in ["int", "bool"]:
        return arg_info.get("value")

    elif arg_type == "tensor":
        dtype_str = arg_info.get("dtype")
        try:
            torch_dtype = getattr(torch, dtype_str.split(".")[-1])
        except AttributeError:
            torch_dtype = torch.float32

        shape = arg_info.get("shape", [])
        device = arg_info.get("device", "cpu")

        # Use a dummy tensor to check properties of the dtype
        tensor_props = torch.empty(0, dtype=torch_dtype)

        # Case 1: Floating point, signed integers, uint8, and bool are supported by random_()
        if tensor_props.is_floating_point() or torch_dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        ]:
            return torch.empty(shape, dtype=torch_dtype, device=device).random_()

        # Case 2: Complex numbers need special handling
        elif tensor_props.is_complex():
            float_dtype = (
                torch.float32 if torch_dtype == torch.complex64 else torch.float64
            )
            real_part = torch.rand(shape, dtype=float_dtype, device=device)
            imag_part = torch.rand(shape, dtype=float_dtype, device=device)
            return torch.complex(real_part, imag_part)

        # Case 3: Handle other unsigned integers (like uint32) which fail with random_()
        elif "uint" in str(torch_dtype):
            return torch.randint(0, 1000, shape, dtype=torch_dtype, device=device)

        # Case 4: If we don't know how to handle the type, raise an error
        else:
            raise NotImplementedError(
                f"Random data generation not implemented for dtype: {torch_dtype}"
            )

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
