# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Reproduce + regression test for the multiprocess trace write corruption bug.

Root cause: subprocess workers (e.g. inductor compile_worker pool) are not in
the distributed group, so `_get_current_rank()` returns None and they all
share the same `dedicated_log_triton_trace_{user}_.ndjson` file. The shared
file plus a non-atomic per-record write pattern produces an interleaved file
where individual lines fail json.loads — exactly what was observed in the
production 365MB trace.

Where the non-atomic write pattern comes from in production:
  - NFS / FUSE / EdenFS: O_APPEND is not atomic across processes (the local
    Linux kernel's i_rwsem only covers local FS, not network FS).
  - Kernel short writes on very large buffers: BufferedWriter loops to
    retry, and other processes' writes can interleave between retries.
  - Compression paths that emit multiple bytes objects per record.

On a local SSD with single-syscall plain writes, O_APPEND is naturally
atomic and corruption does NOT reproduce. So this test covers TWO different
but related guarantees of the Phase 1 fix:

1. test_each_worker_writes_to_distinct_pid_file
     Direct test of the file-sharing root cause. Always reproduces locally.

2. test_corruption_under_fragmented_writes
     Faithfully reproduces the production data-corruption symptom by
     simulating non-atomic writes (each record split into many small
     write+flush pairs). This is the only way to trigger byte-level
     interleaving on a local SSD. Without the fix this test FAILS with
     many json.loads errors. With the fix it PASSES because each worker
     writes to its own file (interleaving impossible regardless of write
     atomicity).

3. test_concurrent_subprocess_write_no_corruption
     Sanity check using the natural single-syscall write path. Tends to
     pass before the fix on local SSD (false negative for the bug), which
     is itself useful information when debugging FS-specific behavior.

"""

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import re
import shutil
import tempfile
import unittest


_PID_REGEX = re.compile(r"pid_(\d+)_")


# Number of subprocess workers writing concurrently. 8 matches the typical
# TORCHINDUCTOR_COMPILE_THREADS default and is enough to reliably exercise
# the file-sharing path.
NUM_WORKERS = 8

# Records each worker emits. Larger N makes the fragmented-writes test more
# reliably trigger interleaving (the production bug is statistical at low N).
RECORDS_PER_WORKER = 15

# Each record carries a payload this large. 100KB is well above Python's
# text-mode buffer (8KB) and large enough to give the corruption check
# meaningful weight while keeping total trace size bounded for CI.
RECORD_PAYLOAD_BYTES = 100_000


def _worker_emit_records(args: tuple[str, int, int, int]) -> int:
    """
    Subprocess worker entry point.

    Initializes tritonparse from scratch in this fresh process (spawn context
    guarantees no shared module state with parent), then emits a batch of
    large records through the structured logging handler.

    Returns the number of records actually emitted.
    """
    trace_dir, worker_id, n_records, payload_bytes = args

    # Set TRITON_TRACE before importing tritonparse so the module-level
    # `triton_trace_folder` picks it up.
    os.environ["TRITON_TRACE"] = trace_dir

    # Force plain (non-compressed) ndjson — that is the buggy path.
    os.environ["TRITON_TRACE_COMPRESSION"] = "none"

    import tritonparse.structured_logging as sl

    sl.init_basic(trace_folder=trace_dir)

    # Force any internal emit() exception to propagate instead of being
    # silently swallowed by Handler.handleError.
    handler = sl.TRITON_TRACE_HANDLER
    if handler is None:
        raise RuntimeError(
            f"worker {worker_id}: TRITON_TRACE_HANDLER is None after init_basic"
        )
    if handler not in sl.triton_trace_log.handlers:
        raise RuntimeError(
            f"worker {worker_id}: handler not attached; "
            f"root_dir={handler.root_dir} "
            f"handlers={sl.triton_trace_log.handlers}"
        )

    import sys

    def _reraise(_record):
        _, exc_value, _ = sys.exc_info()
        if exc_value is not None:
            raise exc_value
        raise RuntimeError("handleError invoked without active exception")

    handler.handleError = _reraise

    # Build a payload large enough to force Python's text-mode buffered I/O
    # to split a single .write() into multiple write() syscalls. This is the
    # precondition for cross-process interleaving.
    payload = "X" * payload_bytes

    emitted = 0
    for i in range(n_records):
        # Use trace_structured_triton — the same entry point real Triton
        # compilation hooks use. It builds a TritonLogRecord with the
        # metadata/payload attributes the custom JSON formatter needs.
        sl.trace_structured_triton(
            "compilation",
            metadata_fn=lambda i=i: {
                "worker_id": worker_id,
                "record_id": i,
            },
            payload_fn=lambda: json.dumps({"file_content": {"kernel.ptx": payload}}),
        )
        emitted += 1

    handler.close()
    return emitted


def _worker_emit_records_fragmented(
    args: tuple[str, int, int, int, int],
) -> int:
    """
    Like _worker_emit_records but forces each record's write to be split
    into many small `write() + flush()` syscalls. This simulates the
    multi-syscall pattern that NFS / FUSE create in production, allowing
    cross-process interleaving to manifest even on local-SSD test runners
    where single-syscall O_APPEND would be naturally atomic.

    Implementation: after init_basic, perform one priming emit so the
    handler's stream is opened, then wrap that stream with a chunking
    proxy. Subsequent emits go through the proxy.
    """
    trace_dir, worker_id, n_records, payload_bytes, chunk_size = args

    os.environ["TRITON_TRACE"] = trace_dir
    os.environ["TRITON_TRACE_COMPRESSION"] = "none"

    import sys

    import tritonparse.structured_logging as sl

    sl.init_basic(trace_folder=trace_dir)

    handler = sl.TRITON_TRACE_HANDLER
    if handler is None:
        raise RuntimeError(
            f"worker {worker_id}: TRITON_TRACE_HANDLER is None after init_basic"
        )

    def _reraise(_record):
        _, exc_value, _ = sys.exc_info()
        if exc_value is not None:
            raise exc_value
        raise RuntimeError("handleError invoked without active exception")

    handler.handleError = _reraise

    # Priming emit so handler.stream is opened and we can wrap it.
    sl.trace_structured_triton("init", payload_fn=lambda: '{"_priming": true}')

    class _FragmentingWriter:
        """Forwards .write to inner stream in chunk_size pieces, flushing
        after each piece. Other attributes (close, flush, fileno, ...)
        delegate to the inner stream."""

        def __init__(self, inner, chunk):
            self._inner = inner
            self._chunk = chunk

        def write(self, data):
            view = data
            n = len(view)
            for i in range(0, n, self._chunk):
                self._inner.write(view[i : i + self._chunk])
                self._inner.flush()
            return n

        def __getattr__(self, name):
            return getattr(self._inner, name)

    handler.stream = _FragmentingWriter(handler.stream, chunk_size)

    # Sanity: count the number of fragmented writes to confirm fragmentation
    # actually happened (fail loudly if our wrapper somehow got bypassed).
    write_call_count = [0]
    inner_write_orig = handler.stream._inner.write

    def counting_write(data):
        write_call_count[0] += 1
        return inner_write_orig(data)

    handler.stream._inner.write = counting_write

    payload = "X" * payload_bytes
    emitted = 0
    for i in range(n_records):
        sl.trace_structured_triton(
            "compilation",
            metadata_fn=lambda i=i: {
                "worker_id": worker_id,
                "record_id": i,
            },
            payload_fn=lambda: json.dumps({"file_content": {"kernel.ptx": payload}}),
        )
        emitted += 1

    handler.close()
    # Each ~100KB record fragmented into 256-byte chunks should be ~390 calls.
    # 15 records → ~5850 inner writes. If anywhere near n_records, the
    # wrapper got bypassed.
    expected_min_writes = (payload_bytes // chunk_size) * n_records // 2
    if write_call_count[0] < expected_min_writes:
        raise RuntimeError(
            f"worker {worker_id}: fragmenting wrapper apparently bypassed; "
            f"got {write_call_count[0]} inner writes, expected ≥ {expected_min_writes}"
        )
    return emitted


def _verify_trace_dir(trace_dir: str) -> dict:
    """
    Read every trace file produced and count json.loads success/failure
    per file. Returns aggregate stats.
    """
    stats = {
        "files": [],
        "total_size_bytes": 0,
        "total_lines": 0,
        "total_ok": 0,
        "total_fail": 0,
    }
    pattern = os.path.join(trace_dir, "dedicated_log_triton_trace_*")
    for path in sorted(glob.glob(pattern)):
        size = os.path.getsize(path)
        ok = 0
        fail = 0
        with open(path, "rb") as fp:
            for raw in fp:
                if not raw.strip():
                    continue
                try:
                    json.loads(raw.decode("utf-8"))
                    ok += 1
                except (json.JSONDecodeError, UnicodeDecodeError):
                    fail += 1
        stats["files"].append({"path": path, "size": size, "ok": ok, "fail": fail})
        stats["total_size_bytes"] += size
        stats["total_lines"] += ok + fail
        stats["total_ok"] += ok
        stats["total_fail"] += fail
    return stats


class MultiprocessWriteTest(unittest.TestCase):
    """
    Reproduce subprocess write file-sharing in the no-rank trace path.

    This test spawns NUM_WORKERS fresh subprocess workers (spawn context, no
    fork, no inherited file descriptors), each of which independently
    initializes tritonparse and emits RECORDS_PER_WORKER large records.
    """

    def setUp(self):
        self.trace_dir = tempfile.mkdtemp(prefix="tritonparse_repro_")

    def tearDown(self):
        shutil.rmtree(self.trace_dir, ignore_errors=True)

    def _run_workers_and_collect_stats(self) -> dict:
        # spawn (not fork) mirrors inductor's compile_worker behavior:
        # subprocess.Popen-style fresh processes with no inherited handler.
        ctx = mp.get_context("spawn")
        worker_args = [
            (self.trace_dir, wid, RECORDS_PER_WORKER, RECORD_PAYLOAD_BYTES)
            for wid in range(NUM_WORKERS)
        ]
        with ctx.Pool(NUM_WORKERS) as pool:
            emitted_per_worker = pool.map(_worker_emit_records, worker_args)

        self.assertEqual(
            sum(emitted_per_worker),
            NUM_WORKERS * RECORDS_PER_WORKER,
            "Workers did not all emit the expected record count",
        )

        return _verify_trace_dir(self.trace_dir)

    def _format_stats(self, stats: dict) -> str:
        diagnostic = (
            f"\n  trace_dir={self.trace_dir}"
            f"\n  files={len(stats['files'])}"
            f"\n  total_size_bytes={stats['total_size_bytes']}"
            f"\n  total_lines={stats['total_lines']}"
            f"\n  total_ok={stats['total_ok']}"
            f"\n  total_fail={stats['total_fail']}"
        )
        for f in stats["files"]:
            diagnostic += (
                f"\n    {os.path.basename(f['path'])}: "
                f"size={f['size']} ok={f['ok']} fail={f['fail']}"
            )
        return diagnostic

    def test_each_worker_writes_to_distinct_pid_file(self):
        """
        Primary fix verification: with PID suffix, N workers → N distinct files.

        BEFORE fix: glob returns 1 file (all workers share no-rank file),
                    no file name contains 'pid_'.
        AFTER fix:  glob returns NUM_WORKERS files, each with a unique PID
                    in its name.
        """
        stats = self._run_workers_and_collect_stats()
        diagnostic = self._format_stats(stats)

        self.assertEqual(
            len(stats["files"]),
            NUM_WORKERS,
            f"Expected {NUM_WORKERS} distinct PID files, got "
            f"{len(stats['files'])}. Workers are sharing one no-rank file "
            f"(file-sharing root cause not fixed):{diagnostic}",
        )

        pids_seen = set()
        for f in stats["files"]:
            m = _PID_REGEX.search(os.path.basename(f["path"]))
            self.assertIsNotNone(
                m,
                f"File name lacks 'pid_N_' marker: "
                f"{os.path.basename(f['path'])} (PID suffix not applied)",
            )
            pids_seen.add(int(m.group(1)))
        self.assertEqual(
            len(pids_seen),
            NUM_WORKERS,
            f"Expected {NUM_WORKERS} distinct PIDs, got {len(pids_seen)} "
            f"({sorted(pids_seen)}):{diagnostic}",
        )

    def test_concurrent_subprocess_write_no_corruption(self):
        """
        Sanity check using the natural single-syscall write path. May pass
        even before the fix on local SSD (where O_APPEND is naturally atomic),
        guaranteed to pass after the fix. False-negative on local SSD; see
        test_corruption_under_fragmented_writes for the production-faithful
        corruption reproducer.
        """
        stats = self._run_workers_and_collect_stats()
        diagnostic = self._format_stats(stats)

        self.assertEqual(stats["total_fail"], 0, f"Found corrupt lines:{diagnostic}")
        self.assertEqual(
            stats["total_ok"],
            NUM_WORKERS * RECORDS_PER_WORKER,
            f"Record count mismatch (lost data?):{diagnostic}",
        )

    def test_corruption_under_fragmented_writes(self):
        """
        Faithfully reproduce the production data-corruption symptom on local
        SSD by simulating non-atomic per-record writes.

        Each worker's emit() is wrapped to split its single Python `.write`
        into many small `os.write` syscalls (with flush between), mirroring
        what NFS / FUSE non-atomic O_APPEND or kernel short-write retry
        cycles do in production.

        BEFORE fix: workers share one file, fragmented writes from N
                    processes interleave → many lines fail json.loads
                    (this matches the production 365MB corrupted trace).
        AFTER fix:  workers have separate PID files → no shared file → no
                    interleaving possible regardless of write fragmentation.
        """
        ctx = mp.get_context("spawn")
        chunk_size = 256  # Small enough to force many syscalls per record
        worker_args = [
            (
                self.trace_dir,
                wid,
                RECORDS_PER_WORKER,
                RECORD_PAYLOAD_BYTES,
                chunk_size,
            )
            for wid in range(NUM_WORKERS)
        ]
        with ctx.Pool(NUM_WORKERS) as pool:
            emitted_per_worker = pool.map(_worker_emit_records_fragmented, worker_args)

        self.assertEqual(
            sum(emitted_per_worker),
            NUM_WORKERS * RECORDS_PER_WORKER,
            "Workers did not all emit the expected record count",
        )

        stats = _verify_trace_dir(self.trace_dir)
        diagnostic = self._format_stats(stats)

        # Expect every line — including the "_priming" lines — to parse.
        # Total expected = NUM_WORKERS * (RECORDS_PER_WORKER + 1 priming).
        expected_records = NUM_WORKERS * (RECORDS_PER_WORKER + 1)
        self.assertEqual(
            stats["total_fail"],
            0,
            f"Found corrupt lines under fragmented writes "
            f"(this matches the production bug):{diagnostic}",
        )
        self.assertEqual(
            stats["total_ok"],
            expected_records,
            f"Record count mismatch under fragmented writes:{diagnostic}",
        )
