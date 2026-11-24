#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Compression utilities for tritonparse trace files.

Provides transparent handling of compressed trace files,
supporting gzip (.bin.ndjson, .ndjson.gz, .gz) and zstd (.zst) formats.

This module uses magic number detection for reliability, which works
regardless of file extension.

Usage:
    from tritonparse.tools.compression import open_compressed_file, detect_compression

    # Auto-detect and open any trace file
    with open_compressed_file("trace.bin.ndjson") as f:
        for line in f:
            process(line)

    # Check compression type
    compression = detect_compression("trace.ndjson.zst")  # Returns "zstd"
"""

import gzip
import io
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO, Union

import zstandard as zstd
from tritonparse.tp_logger import logger

# Magic numbers for compression format detection
# gzip: 0x1F 0x8B (RFC 1952)
GZIP_MAGIC = b"\x1f\x8b"
# zstd: 0xFD2FB528 (little-endian), appears as 28 B5 2F FD in file
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def detect_compression(filepath: Union[str, Path]) -> str:
    """
    Detect compression format of a file using magic number detection.

    This method reads the first few bytes of the file to identify the
    compression format, which is more reliable than checking file extensions.

    Args:
        filepath: Path to the file to check

    Returns:
        Compression type: "gzip", "zstd", or "none"

    Raises:
        FileNotFoundError: If file does not exist
    """
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
    """
    Check if a file is gzip compressed.

    Args:
        filepath: Path to the file to check

    Returns:
        True if file is gzip compressed, False otherwise
    """
    try:
        return detect_compression(filepath) == "gzip"
    except FileNotFoundError:
        return False


def is_zstd_file(filepath: Union[str, Path]) -> bool:
    """
    Check if a file is zstd compressed.

    Args:
        filepath: Path to the file to check

    Returns:
        True if file is zstd compressed, False otherwise
    """
    try:
        return detect_compression(filepath) == "zstd"
    except FileNotFoundError:
        return False


class _TruncatedGzipTailReader(io.BufferedIOBase):
    """Treat a truncated trailing gzip member as end of file.

    TritonTraceHandler writes one gzip member per record, so a trace whose
    writer was killed mid-record, or that is read while the job is still
    writing, ends in an incomplete member. gzip raises EOFError there and the
    complete records before it would be lost. Plain-text and zstd traces
    already keep those records, so do the same for gzip.

    Whatever part of the truncated record gzip could still decode is passed
    through, so the stream ends with that partial line and no trailing
    newline. That matches what a truncated plain-text trace gives, and the
    parser drops the line as invalid JSON.

    Only EOFError is caught. A bad CRC or corrupt deflate data in any member
    still raises.
    """

    def __init__(self, gzip_file: gzip.GzipFile, filepath: Path):
        self._gzip_file = gzip_file
        self._filepath = filepath
        self._truncated = False
        self._warned = False

    def _hit_truncated_tail(self) -> None:
        self._truncated = True
        if not self._warned:
            self._warned = True
            logger.warning(
                f"{self._filepath} ends with a truncated gzip record; "
                "the records before it were read, the last one is incomplete."
            )

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return self._gzip_file.seekable()

    def tell(self) -> int:
        return self._gzip_file.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._truncated = False
        try:
            return self._gzip_file.seek(offset, whence)
        except EOFError:
            # Seeking forward decompresses up to the target, so it can run
            # into the truncated tail. Stop at the end of the readable data.
            self._hit_truncated_tail()
            return self._gzip_file.tell()

    def read1(self, size: int = -1) -> bytes:
        if self._truncated:
            return b""
        try:
            # read1 returns data one member at a time, so records decoded
            # before the truncated member have already been handed out.
            return self._gzip_file.read1(size)
        except EOFError:
            self._hit_truncated_tail()
            return b""

    def read(self, size: int = -1) -> bytes:
        chunks = []
        remaining = size if size is not None and size >= 0 else None
        while remaining is None or remaining > 0:
            chunk = self.read1(-1 if remaining is None else remaining)
            if not chunk:
                break
            chunks.append(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        return b"".join(chunks)


class _ZstdFramesReader(io.BufferedIOBase):
    """Read concatenated zstd frames and warn if the last one is cut off.

    TritonTraceHandler writes one zstd frame per record. A trace whose writer
    was killed mid-record, or that is read while the job is still writing,
    ends in an incomplete frame. The complete records before it are read as
    usual, the same as with ``stream_reader(read_across_frames=True)``, but
    that reader drops the incomplete frame without saying so. Decoding one
    frame at a time lets us tell that case apart and log one warning, as the
    gzip path does.

    Whatever part of the truncated frame zstd could already decode is passed
    through, so the stream can end with a partial line, as with gzip.
    Corrupt data still raises ``zstd.ZstdError``.
    """

    _READ_SIZE = zstd.DECOMPRESSION_RECOMMENDED_INPUT_SIZE
    # A finished frame hands back the rest of the input it was given as a
    # copy. Records are usually small, so feeding small pieces keeps that
    # copy short; this was faster than 16 KiB or a whole read.
    _FEED_SIZE = 2 * 1024

    def __init__(self, binary_file: BinaryIO, filepath: Path):
        self._file = binary_file
        self._filepath = filepath
        self._dctx = zstd.ZstdDecompressor()
        # Decompressor for the frame being read; None between frames.
        self._frame = None
        # Compressed bytes read from the file; the part from _in_pos on has
        # not been decoded yet.
        self._input = memoryview(b"")
        self._in_pos = 0
        # Decoded bytes; the part from _pos on has not been returned yet.
        self._output = b""
        self._pos = 0
        self._at_end = False

    def readable(self) -> bool:
        return True

    def _fill(self) -> None:
        """Decode until there is output or the file is exhausted."""
        while self._pos == len(self._output) and not self._at_end:
            if self._in_pos == len(self._input):
                self._input = memoryview(self._file.read(self._READ_SIZE))
                self._in_pos = 0
                if not self._input:
                    self._at_end = True
                    if self._frame is not None:
                        logger.warning(
                            f"{self._filepath} ends with a truncated zstd record; "
                            "the records before it were read, the last one is "
                            "incomplete."
                        )
                    return
            if self._frame is None:
                self._frame = self._dctx.decompressobj()
            data = self._input[self._in_pos : self._in_pos + self._FEED_SIZE]
            self._output = self._frame.decompress(data)
            self._pos = 0
            self._in_pos += len(data)
            if self._frame.eof:
                # Bytes past the end of this frame belong to the next one.
                self._in_pos -= len(self._frame.unused_data)
                self._frame = None

    def read1(self, size: int = -1) -> bytes:
        self._fill()
        end = len(self._output)
        if size is not None and size >= 0:
            end = min(end, self._pos + size)
        out = self._output[self._pos : end]
        self._pos = end
        return out

    def read(self, size: int = -1) -> bytes:
        chunks = []
        remaining = size if size is not None and size >= 0 else None
        while remaining is None or remaining > 0:
            chunk = self.read1(-1 if remaining is None else remaining)
            if not chunk:
                break
            chunks.append(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        return b"".join(chunks)


@contextmanager
def open_compressed_file(filepath: Union[str, Path]) -> Iterator[TextIO]:
    """
    Open a file with automatic compression detection and handling.

    This context manager transparently handles gzip, zstd, and plain text files.
    Compression format is detected using magic numbers, not file extensions.

    Args:
        filepath: Path to the file to open

    Yields:
        Text stream for reading the file contents

    Raises:
        FileNotFoundError: If file does not exist

    Example:
        >>> with open_compressed_file("trace.bin.ndjson") as f:
        ...     for line in f:
        ...         record = json.loads(line)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    compression = detect_compression(filepath)

    if compression == "gzip":
        # GzipFile handles gzip member concatenation automatically
        with gzip.GzipFile(filepath, "rb") as gzip_file:
            reader = _TruncatedGzipTailReader(gzip_file, filepath)
            with io.TextIOWrapper(reader, encoding="utf-8") as f:
                yield f
    elif compression == "zstd":
        with open(filepath, "rb") as binary_file:
            reader = _ZstdFramesReader(binary_file, filepath)
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                yield text_stream
    else:
        # Plain text file
        with open(filepath, "r", encoding="utf-8") as f:
            yield f


def iter_lines(filepath: Union[str, Path]) -> Iterator[str]:
    """
    Iterate over lines in a file, handling compression transparently.

    This is a memory-efficient way to process large trace files line by line.

    Args:
        filepath: Path to the file

    Yields:
        Lines from the file (with trailing newlines stripped)

    Raises:
        FileNotFoundError: If file does not exist

    Example:
        >>> for line in iter_lines("trace.bin.ndjson"):
        ...     record = json.loads(line)
    """
    with open_compressed_file(filepath) as f:
        for line in f:
            yield line.rstrip("\n\r")
