# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for tritonparse.tools.compression module."""

import gzip
import tempfile
import unittest
import zstd
from pathlib import Path

from tritonparse.tools.compression import (
    detect_compression,
    is_gzip_file,
    is_zstd_file,
    iter_lines,
    open_compressed_file,
)


class DetectCompressionTest(unittest.TestCase):
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
    def test_read_plain_text(self):
        with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False, mode="w") as f:
            path = f.name
            f.write('{"a": 1}\n{"a": 2}\n')
        try:
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)
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
        cctx = zstd.ZstdCompressor()
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            for i in range(10):
                record = f'{{"line": {i}}}\n'.encode("utf-8")
                f.write(cctx.compress(record))
        try:
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 10)
            for i in range(10):
                self.assertIn(f'"line": {i}', lines[i])
        finally:
            Path(path).unlink()

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            with open_compressed_file("/nonexistent/file"):
                pass


class IterLinesTest(unittest.TestCase):
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
        cctx = zstd.ZstdCompressor()
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", delete=False) as f:
            path = f.name
            for i in range(5):
                f.write(cctx.compress(f"record_{i}\n".encode("utf-8")))
        try:
            lines = list(iter_lines(path))
            self.assertEqual(len(lines), 5)
        finally:
            Path(path).unlink()


class ZstdRoundTripTest(unittest.TestCase):
    def test_write_read_round_trip(self):
        cctx = zstd.ZstdCompressor()
        records = [
            '{"event": "compilation", "kernel": "add_kernel", "idx": 0}\n',
            '{"event": "compilation", "kernel": "matmul_kernel", "idx": 1}\n',
            '{"event": "launch", "kernel": "add_kernel", "idx": 2}\n',
        ]
        with tempfile.NamedTemporaryFile(suffix=".bin.ndjson", mode="ab+", delete=False) as f:
            path = f.name
            for record in records:
                compressed = cctx.compress(record.encode("utf-8"))
                f.write(compressed)
        try:
            self.assertEqual(detect_compression(path), "zstd")
            with open_compressed_file(path) as fh:
                read_lines = fh.readlines()
            self.assertEqual(len(read_lines), len(records))
            for original, read_back in zip(records, read_lines):
                self.assertEqual(original, read_back)
        finally:
            Path(path).unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".ndjson", delete=False) as f:
            path = f.name
        try:
            self.assertEqual(detect_compression(path), "none")
            with open_compressed_file(path) as fh:
                lines = fh.readlines()
            self.assertEqual(lines, [])
        finally:
            Path(path).unlink()