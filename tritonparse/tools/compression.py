#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Compression utilities for tritonparse trace files.

Provides transparent handling of compressed trace files,
supporting gzip (.bin.ndjson, .ndjson.gz, .gz) and zstd (.zst) formats.

Uses Python 3.14+ standard library zstd module.
"""

import gzip
import io
import zstd
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO, Union

# Magic numbers for compression format detection
GZIP_MAGIC = b"\x1f\x8b"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def detect_compression(filepath: Union[str, Path]) -> str:
    """Detect compression format using magic number detection."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic[:2] == GZIP_MAGIC:
            return "gzip"
        if magic == ZSTD_MAGIC:
            return "zstd"
    return "none"


def is_gzip_file(filepath: Union[str, Path]) -> bool:
    try:
        return detect_compression(filepath) == "gzip"
    except FileNotFoundError:
        return False


def is_zstd_file(filepath: Union[str, Path]) -> bool:
    try:
        return detect_compression(filepath) == "zstd"
    except FileNotFoundError:
        return False


@contextmanager
def open_compressed_file(filepath: Union[str, Path]) -> Iterator[TextIO]:
    """Open a file with automatic compression detection."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    compression = detect_compression(filepath)

    if compression == "gzip":
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            yield f
    elif compression == "zstd":
        dctx = zstd.ZstdDecompressor()
        with open(filepath, "rb") as binary_file:
            with dctx.stream_reader(binary_file, read_across_frames=True) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                    yield text_stream
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            yield f


def iter_lines(filepath: Union[str, Path]) -> Iterator[str]:
    """Iterate over lines with transparent compression handling."""
    with open_compressed_file(filepath) as f:
        for line in f:
            yield line.rstrip("\n\r")