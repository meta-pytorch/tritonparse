#!/usr/bin/env python3
#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Simple tensor loading utility for tritonparse saved tensors.
Usage:
import tritonparse.tools.load_tensor as load_tensor
tensor = load_tensor.load_tensor(tensor_file_path, device)
"""

import gzip
import hashlib
import io
from pathlib import Path
from typing import Union

import torch


def load_tensor(tensor_file_path: Union[str, Path], device: str = None) -> torch.Tensor:
    """
    Load a tensor from its file path and verify its integrity using the hash in the filename.

    Args:
        tensor_file_path (str | Path): Direct path to the tensor file. Supports both:
                               - .bin.gz: gzip-compressed tensor (hash is of uncompressed data)
                               - .bin: uncompressed tensor (for backward compatibility)
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

    # Detect compression by file extension
    is_compressed = blob_path.name.endswith(".bin.gz")

    # Read file contents (decompress if needed)
    try:
        with open(blob_path, "rb") as f:
            file_obj = gzip.GzipFile(fileobj=f, mode="rb") if is_compressed else f
            file_contents = file_obj.read()
    except (OSError, gzip.BadGzipFile) as e:
        if is_compressed:
            raise RuntimeError(f"Failed to decompress gzip file {blob_path}: {str(e)}")
        else:
            raise RuntimeError(f"Failed to read file {blob_path}: {str(e)}")

    # Extract expected hash from filename
    # abc123.bin.gz -> abc123 or abc123.bin -> abc123
    expected_hash = blob_path.name.removesuffix(".bin.gz" if is_compressed else ".bin")

    # Compute hash of uncompressed data
    computed_hash = hashlib.blake2b(file_contents).hexdigest()

    # Verify hash matches filename
    if computed_hash != expected_hash:
        raise ValueError(
            f"Hash verification failed: expected '{expected_hash}' but computed '{computed_hash}'"
        )

    try:
        # Load the tensor from memory buffer
        tensor = torch.load(io.BytesIO(file_contents), map_location=device)
        return tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load tensor from {blob_path}: {str(e)}")
