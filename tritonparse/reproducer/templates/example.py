import hashlib
import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import torch

# {{KERNEL_IMPORT_PLACEHOLDER}}

TRITON_KERNELS_CUSTOM_TYPES = (
    importlib.util.find_spec("triton_kernels.tensor") is not None
)


@lru_cache(maxsize=1)
def _get_triton_tensor_types():
    """
    Import and cache Triton custom tensor types.

    Returns:
        tuple: (Tensor, Storage, StridedLayout) classes from triton_kernels.tensor.

    Raises:
        ImportError: If the optional module 'triton_kernels.tensor' is not available.
    """
    mod = importlib.import_module("triton_kernels.tensor")
    return (
        getattr(mod, "Tensor"),
        getattr(mod, "Storage"),
        getattr(mod, "StridedLayout"),
    )


def load_tensor(tensor_file_path: str, device: str = None) -> torch.Tensor:
    """
    Load a tensor from its file path and verify its integrity using the hash in the filename.

    Args:
        tensor_file_path (str): Direct path to the tensor .bin file. The filename should be
                               the hash of the file contents followed by .bin extension.
        device (str, optional): Device to load the tensor to (e.g., 'cuda:0', 'cpu').
                               If None, keeps the tensor on its original device.

    Returns:
        torch.Tensor: The loaded tensor (moved to the specified device if provided)

    Raises:
        FileNotFoundError: If the tensor file doesn't exist
        RuntimeError: If the tensor cannot be loaded
        ValueError: If the computed hash doesn't match the filename hash
    """
    blob_path = Path(tensor_file_path)

    if not blob_path.exists():
        raise FileNotFoundError(f"Tensor blob not found: {blob_path}")

    # Extract expected hash from filename (remove .bin extension)
    expected_hash = blob_path.stem

    # Compute actual hash of file contents
    with open(blob_path, "rb") as f:
        file_contents = f.read()
        computed_hash = hashlib.blake2b(file_contents).hexdigest()

    # Verify hash matches filename
    if computed_hash != expected_hash:
        raise ValueError(
            f"Hash verification failed: expected '{expected_hash}' but computed '{computed_hash}'"
        )

    try:
        # Load the tensor using torch.load (tensors are saved with torch.save)
        # If device is None, keep tensor on its original device, otherwise move to specified device
        tensor = torch.load(blob_path, map_location=device)
        return tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor from {blob_path}: {str(e)}") from e


def create_args_from_json(json_path):
    """
    Parse a reproducer JSON and build kernel grid and argument dictionary.

    Args:
        json_path (str): Path to the JSON file describing the kernel launch.

    Returns:
        tuple[list, dict]: Grid specification list and map of argument name to value.
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

    if arg_type in ["int", "bool"]:
        return arg_info.get("value")

    elif arg_type == "tensor":
        if arg_info.get("blob_path"):
            return load_tensor(arg_info.get("blob_path"), arg_info.get("device"))
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
        if tensor_props.is_floating_point():
            if torch_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                tmp = torch.rand(shape, dtype=torch.float32, device=device)
                return tmp.to(torch_dtype)
            else:
                return torch.empty(shape, dtype=torch_dtype, device=device).random_()
        elif torch_dtype in [
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
        # Case 4: If we don't know how to handle the type, raise an error
        else:
            raise NotImplementedError(
                f"Random data generation not implemented for dtype: {torch_dtype}"
            )

    else:
        print(f"Warning: Unhandled argument type '{arg_type}'. Returning None.")
        return None


if __name__ == "__main__":
    json_file = "{{JSON_PATH_PLACEHOLDER}}"
    grid, args_dict = create_args_from_json(json_file)

    print("Generated kernel arguments dictionary:")
    for name, arg in args_dict.items():
        print(f"  {name}: {arg}")
    print(f"Grid: {grid}")

    # {{KERNEL_INVOCATION_PLACEHOLDER}}

    torch.cuda.synchronize()
    print("Kernel launch finished.")
