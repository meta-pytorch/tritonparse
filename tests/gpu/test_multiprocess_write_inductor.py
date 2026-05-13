# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Reproduce + regression test for multiprocess trace write file-sharing
through inductor's real compile_worker pool.

This is the production-realistic counterpart to
tests/cpu/test_multiprocess_write.py — instead of synthetic subprocess
workers, it triggers real Triton kernel compilation through torch.compile
with TORCHINDUCTOR_COMPILE_THREADS > 1 so that inductor spawns its actual
compile_worker subprocess pool. Each worker independently initializes
tritonparse and writes trace events.

Bug manifestation matches the synthetic test:
  Before fix: all worker subprocesses share one
              `dedicated_log_triton_trace_{user}_.ndjson` file.
  After fix:  each worker writes its own `..._pid_{PID}_.ndjson`.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import tempfile

import torch
import tritonparse.structured_logging
from tests.test_utils import GPUTestBase  # noqa: F401  (used via inheritance)


_PID_REGEX = re.compile(r"pid_(\d+)_")
_LOG_PREFIX = "dedicated_log_triton_trace_"

# Number of inductor compile worker subprocesses. > 1 forces inductor to use
# its subprocess pool rather than compiling inline in the main process.
COMPILE_THREADS = 4


def _kernel_a(x):
    return (x * 2 + 1).relu().sum()


def _kernel_b(x):
    return (x.sin() + x.cos()).pow(2).mean()


def _kernel_c(x):
    return torch.softmax(x @ x.T, dim=-1).sum()


def _kernel_d(x):
    return (x.tanh() * x.exp().clamp(max=10.0)).log1p().mean()


class MultiprocessWriteInductorTest(GPUTestBase):
    """
    Reproduce file-sharing across inductor compile_worker subprocesses.
    """

    def setUp(self):
        super().setUp()
        self.trace_dir = tempfile.mkdtemp(prefix="tritonparse_inductor_repro_")

        # Save env vars we will mutate so tearDown can restore them.
        self._saved_env = {
            k: os.environ.get(k)
            for k in (
                "TRITON_TRACE",
                "TRITON_TRACE_COMPRESSION",
                "TORCHINDUCTOR_COMPILE_THREADS",
                "TORCHINDUCTOR_WORKER_START_METHOD",
                "TRITON_ALWAYS_COMPILE",
            )
        }

        # Set BEFORE inductor spawns workers — workers inherit this env
        # and must see it during their own tritonparse import.
        os.environ["TRITON_TRACE"] = self.trace_dir
        os.environ["TRITON_TRACE_COMPRESSION"] = "none"
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(COMPILE_THREADS)
        # Force subprocess pool (not fork), matching the production scenario
        # in which the bug was originally observed.
        os.environ["TORCHINDUCTOR_WORKER_START_METHOD"] = "subprocess"
        # Skip Triton cache to guarantee actual compilation work for workers.
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"

        # Initialize tritonparse in the main process. The workers will do
        # their own init when they import tritonparse via inductor.
        tritonparse.structured_logging.init(self.trace_dir)

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.trace_dir, ignore_errors=True)
        super().tearDown()

    def _list_trace_files(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.trace_dir, f"{_LOG_PREFIX}*")))

    def _trigger_inductor_compilation(self):
        """Run several distinct torch.compile units to populate the worker pool.

        Use larger tensors and more ops per function to ensure inductor generates
        Triton kernels (small ops may be inlined or use ATen).
        """
        x = torch.randn(2048, 2048, device=self.cuda_device)
        for fn in (_kernel_a, _kernel_b, _kernel_c, _kernel_d):
            compiled = torch.compile(fn, dynamic=False, fullgraph=True)
            compiled(x)
        torch.cuda.synchronize()

    def test_inductor_main_process_uses_pid_filename(self):
        """
        Verify the writer fix's PID suffix is wired through real inductor flow.

        BEFORE fix: trace files have no `pid_` infix (e.g.,
                    `dedicated_log_triton_trace_user_.ndjson`).
        AFTER fix:  every trace file has a `pid_{PID}_` infix.

        Note on test scope: in the buck-test RE environment, inductor's compile
        worker subprocesses do not import tritonparse and so do not write trace
        events of their own — only the main process traces. This test therefore
        only validates that the main process's trace file carries the PID
        suffix when reached via inductor's torch.compile flow. The multi-worker
        cross-process file-sharing scenario is exhaustively covered by
        tests/cpu/test_multiprocess_write.py with explicit spawn workers.
        """
        self._trigger_inductor_compilation()

        files = self._list_trace_files()
        self.assertGreater(
            len(files), 0, f"No trace files produced in {self.trace_dir}"
        )

        # Surface diagnostic state on failure.
        diagnostic = f"\n  trace_dir={self.trace_dir}\n  files:"
        for path in files:
            size = os.path.getsize(path)
            diagnostic += f"\n    {os.path.basename(path)} ({size} bytes)"

        pid_tagged_files = [f for f in files if _PID_REGEX.search(os.path.basename(f))]
        non_pid_files = [f for f in files if not _PID_REGEX.search(os.path.basename(f))]

        # Primary assertion: every produced trace file must have the PID
        # suffix in its name. Before the fix this fails because inductor
        # writes `dedicated_log_triton_trace_user_.ndjson` with no PID.
        # After the fix every file has `pid_{N}_` infix.
        self.assertEqual(
            len(non_pid_files),
            0,
            f"Found trace files without `pid_` infix (PID suffix fix not "
            f"applied or wired through inductor flow):{diagnostic}\n"
            f"  non_pid_files={[os.path.basename(f) for f in non_pid_files]}",
        )
        self.assertGreaterEqual(
            len(pid_tagged_files),
            1,
            f"Expected >=1 PID-tagged trace file from the main process "
            f"(inductor compile_worker subprocesses don't import tritonparse "
            f"in the RE test env, so they don't trace; multi-worker scenario "
            f"is covered by tests/cpu/test_multiprocess_write.py):"
            f"{diagnostic}",
        )

        # Secondary check: each produced file must be cleanly parseable.
        # On local SSD this typically holds even before the fix
        # (O_APPEND atomic), but we keep the check as a guard.
        for path in files:
            with open(path, "rb") as fp:
                for raw in fp:
                    if not raw.strip():
                        continue
                    try:
                        json.loads(raw.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        self.fail(
                            f"Corrupt line in {os.path.basename(path)}: "
                            f"{type(e).__name__}: {e}{diagnostic}"
                        )
