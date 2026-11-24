# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for tritonparse.tools.compression module.

Covers magic number detection, transparent reading of gzip/zstd/plain files,
and the critical multi-frame zstd scenario (per-record frame concatenation).
"""

import base64
import gzip
import io
import json
import logging
import random
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd
from tritonparse.tools.compression import (
    detect_compression,
    is_gzip_file,
    is_zstd_file,
    iter_lines,
    open_compressed_file,
)


class DetectCompressionTest(unittest.TestCase):
    """Tests for detect_compression() magic number detection."""

    def test_detect_gzip(self):
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = f.name
            with gzip.open(f, "wb") as gz:
                gz.write(b"hello\n")
        try:
            self.assertEqual(detect_compression(path), "gzip")
        finally:
            Path(path).unlink()

    def test_detect_zstd(self):
        cctx = zstd.ZstdCompressor()
        data = cctx.compress(b"hello\n")
        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as f:
            path = f.name
            f.write(data)
        try:
            self.assertEqual(detect_compression(path), "zstd")
        finally:
            Path(path).unlink()

    def test_detect_plain_text(self):
        with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False, mode="w") as f:
            path = f.name
            f.write('{"key": "value"}\n')
        try:
            self.assertEqual(detect_compression(path), "none")
        finally:
            Path(path).unlink()

    def test_detect_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            detect_compression("/nonexistent/path/file.ndjson")


class IsCompressedFileTest(unittest.TestCase):
    """Tests for is_gzip_file() and is_zstd_file() helpers."""

    def test_is_gzip_file(self):
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = f.name
            with gzip.open(f, "wb") as gz:
                gz.write(b"test\n")
        try:
            self.assertTrue(is_gzip_file(path))
            self.assertFalse(is_zstd_file(path))
        finally:
            Path(path).unlink()

    def test_is_zstd_file(self):
        cctx = zstd.ZstdCompressor()
        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as f:
            path = f.name
            f.write(cctx.compress(b"test\n"))
        try:
            self.assertTrue(is_zstd_file(path))
            self.assertFalse(is_gzip_file(path))
        finally:
            Path(path).unlink()

    def test_nonexistent_returns_false(self):
        self.assertFalse(is_gzip_file("/nonexistent"))
        self.assertFalse(is_zstd_file("/nonexistent"))


class OpenCompressedFileTest(unittest.TestCase):
    """Tests for open_compressed_file() transparent reading."""

    def test_read_plain_text(self):
        with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False, mode="w") as f:
            path = f.name
            f.write('{"a": 1}\n{"a": 2}\n')
        try:
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"a": 1', lines[0])
        finally:
            Path(path).unlink()

    def test_read_gzip(self):
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = f.name
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                gz.write('{"b": 1}\n{"b": 2}\n')
        try:
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
        finally:
            Path(path).unlink()

    def test_read_gzip_multi_member(self):
        """Gzip member concatenation (same pattern as TritonTraceHandler gzip mode)."""
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            for i in range(5):
                record = f'{{"line": {i}}}\n'.encode("utf-8")
                member = gzip.compress(record)
                f.write(member)
        try:
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 5)
            self.assertIn('"line": 0', lines[0])
            self.assertIn('"line": 4', lines[4])
        finally:
            Path(path).unlink()

    @staticmethod
    def _write_members(f, records):
        """Write each record as its own gzip member, as TritonTraceHandler does."""
        for r in records:
            f.write(gzip.compress(r.encode("utf-8")))

    def test_read_gzip_truncated_trailing_member(self):
        """A writer killed mid-record leaves a partial last gzip member.

        The complete records before it must still be readable, as they are for
        plain-text and zstd traces.
        """
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            self._write_members(f, [f'{{"line": {i}}}\n' for i in range(3)])
            last = b'{"line": 3, "payload": "' + b"x" * 4096 + b'"}\n'
            partial = gzip.compress(last)
            f.write(partial[: len(partial) // 2])
        try:
            with self.assertLogs("tritonparse", level="WARNING") as logs:
                with open_compressed_file(path) as fh:
                    records = [line for line in fh if line.endswith("\n")]
            self.assertEqual(len(records), 3)
            self.assertIn('"line": 2', records[2])
            self.assertEqual(len(logs.output), 1)
            self.assertIn("truncated gzip record", logs.output[0])
            with self.assertLogs("tritonparse", level="WARNING"):
                with open_compressed_file(path) as fh:
                    text = fh.read()
            self.assertEqual(text.count('"line": 2}\n'), 1)
        finally:
            Path(path).unlink()

    def test_read_gzip_truncated_member_partial_line(self):
        """The decodable prefix of a truncated record comes through as is.

        With data that does not compress well, gzip can still decode part of
        the cut-off member. That part is returned as a last line without a
        newline, the same as a truncated plain-text trace, and the parser drops
        it as invalid JSON.
        """
        payload = base64.b64encode(random.Random(0).randbytes(8192))
        last = b'{"line": 1, "payload": "' + payload + b'"}\n'
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            self._write_members(f, ['{"line": 0}\n'])
            partial = gzip.compress(last)
            f.write(partial[: len(partial) // 2])
        try:
            with self.assertLogs("tritonparse", level="WARNING"):
                with open_compressed_file(path) as fh:
                    lines = list(fh)
            self.assertEqual(lines[0], '{"line": 0}\n')
            self.assertEqual(len(lines), 2)
            tail = lines[1]
            self.assertFalse(tail.endswith("\n"))
            self.assertGreater(len(tail), 0)
            self.assertTrue(last.decode("utf-8").startswith(tail))
            with self.assertRaises(ValueError):
                json.loads(tail)
        finally:
            Path(path).unlink()

    def test_read_gzip_corrupt_middle_member_raises(self):
        """Only a truncated tail is tolerated; a bad CRC mid-file still raises."""
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
        members = [
            gzip.compress(f'{{"line": {i}}}\n'.encode("utf-8")) for i in range(3)
        ]
        bad = bytearray(members[1])
        bad[-8] ^= 0xFF  # first byte of the CRC32 in the member trailer
        Path(path).write_bytes(members[0] + bytes(bad) + members[2])
        try:
            with self.assertRaises(gzip.BadGzipFile):
                with open_compressed_file(path) as fh:
                    fh.read()
        finally:
            Path(path).unlink()

    def test_gzip_seek_and_tell(self):
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            self._write_members(f, [f'{{"line": {i}}}\n' for i in range(3)])
        try:
            with open_compressed_file(path) as fh:
                self.assertTrue(fh.seekable())
                self.assertEqual(fh.readline(), '{"line": 0}\n')
                pos = fh.tell()
                rest = fh.read()
                self.assertEqual(rest, '{"line": 1}\n{"line": 2}\n')
                fh.seek(pos)
                self.assertEqual(fh.read(), rest)
                fh.seek(0)
                self.assertEqual(fh.readline(), '{"line": 0}\n')
        finally:
            Path(path).unlink()

    def test_gzip_seek_with_truncated_tail(self):
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            self._write_members(f, [f'{{"line": {i}}}\n' for i in range(2)])
            partial = gzip.compress(b'{"line": 2, "payload": "' + b"x" * 4096 + b'"}\n')
            f.write(partial[: len(partial) // 2])
        complete = '{"line": 0}\n{"line": 1}\n'
        try:
            with self.assertLogs("tritonparse", level="WARNING") as logs:
                with open_compressed_file(path) as fh:
                    self.assertEqual(fh.read(), complete)
                    # seek resets the truncation state, so the data reads again.
                    fh.seek(0)
                    self.assertEqual(fh.read(), complete)
                    # Seeking past the end stops where the readable data ends.
                    fh.seek(0)
                    end = fh.buffer.seek(1 << 20)
                    self.assertEqual(end, len(complete))
                    self.assertEqual(fh.buffer.read(), b"")
            self.assertEqual(len(logs.output), 1)
        finally:
            Path(path).unlink()

    def test_read_zstd_single_frame(self):
        cctx = zstd.ZstdCompressor()
        content = '{"c": 1}\n{"c": 2}\n'
        with tempfile.NamedTemporaryFile(suffix=".zst", delete=False) as f:
            path = f.name
            f.write(cctx.compress(content.encode("utf-8")))
        try:
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
        finally:
            Path(path).unlink()

    def test_read_zstd_multi_frame(self):
        """Multi-frame zstd (same pattern as TritonTraceHandler zstd mode).

        This is the critical test: each JSON record is compressed as a separate
        zstd frame. The reader must use read_across_frames=True to read all
        frames, not just the first one.
        """
        cctx = zstd.ZstdCompressor()
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            for i in range(10):
                record = f'{{"line": {i}}}\n'.encode("utf-8")
                f.write(cctx.compress(record))
        try:
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            # Must read ALL 10 frames, not just the first one
            self.assertEqual(len(lines), 10)
            for i in range(10):
                self.assertIn(f'"line": {i}', lines[i])
        finally:
            Path(path).unlink()

    def test_read_zstd_truncated_trailing_frame(self):
        """A writer killed mid-record leaves a partial last zstd frame.

        The complete records before it are read, and one warning names the
        file, as for a truncated gzip trace.
        """
        cctx = zstd.ZstdCompressor()
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            for i in range(3):
                f.write(cctx.compress(f'{{"line": {i}}}\n'.encode("utf-8")))
            last = b'{"line": 3, "payload": "' + b"x" * 4096 + b'"}\n'
            partial = cctx.compress(last)
            f.write(partial[: len(partial) // 2])
        try:
            with self.assertLogs("tritonparse", level="WARNING") as logs:
                with open_compressed_file(path) as fh:
                    lines = list(fh)
            self.assertEqual(lines, [f'{{"line": {i}}}\n' for i in range(3)])
            self.assertEqual(len(logs.output), 1)
            self.assertIn("truncated zstd record", logs.output[0])
            with self.assertLogs("tritonparse", level="WARNING"):
                with open_compressed_file(path) as fh:
                    self.assertEqual(fh.read(), "".join(lines))
        finally:
            Path(path).unlink()

    def test_read_zstd_cut_matches_stream_reader(self):
        """Cut a multi-frame trace at every offset and compare with zstandard.

        The decoded bytes must match stream_reader(read_across_frames=True),
        so the only change is the warning, which must appear exactly when the
        cut falls inside a frame. The frames include an empty frame, a
        skippable frame and a record larger than one zstd block.
        """
        cctx = zstd.ZstdCompressor()
        big = base64.b64encode(random.Random(0).randbytes(150_000))
        skippable = b"\x50\x2a\x4d\x18" + (4).to_bytes(4, "little") + b"abcd"
        frames = [
            cctx.compress(b'{"line": 0}\n'),
            cctx.compress(b""),
            skippable,
            cctx.compress(b'{"line": 1, "payload": "' + big + b'"}\n'),
            cctx.compress(b'{"line": 2}\n'),
        ]
        data = b"".join(frames)
        boundaries = set()
        offset = 0
        for frame in frames:
            offset += len(frame)
            boundaries.add(offset)
        # Every offset in the small frames, and a spread through the large one.
        cuts = sorted(
            set(range(4, len(frames[0]) + len(frames[1]) + len(frames[2]) + 40))
            | set(range(4, len(data) + 1, 997))
            | {
                b + d
                for b in boundaries
                for d in (-1, 0, 1, 3, 5)
                if b + d <= len(data)
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
        try:
            for cut in cuts:
                with self.subTest(cut=cut):
                    Path(path).write_bytes(data[:cut])
                    reader = zstd.ZstdDecompressor().stream_reader(
                        io.BytesIO(data[:cut]), read_across_frames=True
                    )
                    expected = reader.read()
                    with self.assertLogs("tritonparse", level="WARNING") as logs:
                        with open_compressed_file(path) as fh:
                            got = fh.buffer.read()
                        # assertLogs fails when nothing is logged.
                        logging.getLogger("tritonparse").warning("end")
                    self.assertEqual(got, expected)
                    warned = [m for m in logs.output if "truncated zstd" in m]
                    self.assertEqual(len(warned), 0 if cut in boundaries else 1)
        finally:
            Path(path).unlink()

    def test_read_zstd_corrupt_middle_frame_raises(self):
        """Only a truncated tail gets the warning; corrupt data still raises."""
        cctx = zstd.ZstdCompressor(write_checksum=True)
        frames = [cctx.compress(f'{{"line": {i}}}\n'.encode("utf-8")) for i in range(3)]
        bad = bytearray(frames[1])
        bad[-1] ^= 0xFF  # last byte of the content checksum
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
        Path(path).write_bytes(frames[0] + bytes(bad) + frames[2])
        try:
            with self.assertRaises(zstd.ZstdError):
                with open_compressed_file(path) as fh:
                    fh.read()
        finally:
            Path(path).unlink()

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            with open_compressed_file("/nonexistent/file"):
                pass


class IterLinesTest(unittest.TestCase):
    """Tests for iter_lines() helper."""

    def test_iter_lines_plain(self):
        with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False, mode="w") as f:
            path = f.name
            f.write("line1\nline2\nline3\n")
        try:
            lines = list(iter_lines(path))
            self.assertEqual(lines, ["line1", "line2", "line3"])
        finally:
            Path(path).unlink()

    def test_iter_lines_zstd_multi_frame(self):
        """iter_lines must handle multi-frame zstd files correctly."""
        cctx = zstd.ZstdCompressor()
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            for i in range(5):
                f.write(cctx.compress(f"record_{i}\n".encode("utf-8")))
        try:
            lines = list(iter_lines(path))
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0], "record_0")
            self.assertEqual(lines[4], "record_4")
        finally:
            Path(path).unlink()


class ZstdRoundTripTest(unittest.TestCase):
    """End-to-end round-trip test simulating TritonTraceHandler write + read."""

    def test_write_read_round_trip(self):
        """Simulate TritonTraceHandler.emit() zstd path, then read back."""
        cctx = zstd.ZstdCompressor()
        records = [
            '{"event": "compilation", "kernel": "add_kernel", "idx": 0}\n',
            '{"event": "compilation", "kernel": "matmul_kernel", "idx": 1}\n',
            '{"event": "launch", "kernel": "add_kernel", "idx": 2}\n',
        ]

        with tempfile.NamedTemporaryFile(
            suffix=".bin.ndjson", mode="ab+", delete=False
        ) as f:
            path = f.name
            # Simulate per-record compression (same as emit())
            for record in records:
                compressed = cctx.compress(record.encode("utf-8"))
                f.write(compressed)

        try:
            # Verify magic number detection identifies as zstd
            self.assertEqual(detect_compression(path), "zstd")

            # Verify all records are readable
            with open_compressed_file(path) as fh:
                read_lines = fh.readlines()

            self.assertEqual(len(read_lines), len(records))
            for original, read_back in zip(records, read_lines):
                self.assertEqual(original, read_back)
        finally:
            Path(path).unlink()

    def test_empty_file(self):
        """Empty file should be detected as plain text."""
        with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False) as f:
            path = f.name
        try:
            self.assertEqual(detect_compression(path), "none")
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            self.assertEqual(lines, [])
        finally:
            Path(path).unlink()
