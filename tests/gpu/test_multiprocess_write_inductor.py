# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Regression test for the PID-suffix trace filename on the main-process
path that goes through torch.compile / inductor's Triton compilation.

The multi-worker file-sharing scenario (multiple processes writing to
distinct pid-tagged files) is exhaustively covered by
tests/cpu/test_multiprocess_write.py with explicit spawn workers. This
test complements that one by exercising the inductor entry path; it
runs inductor inline in the main process (TORCHINDUCTOR_COMPILE_THREADS=1)
to validate that tritonparse's PID-suffix writer fix is wired through
torch.compile without depending on inductor's compile-worker pool.

Bug manifestation (main process):
  Before fix: `dedicated_log_triton_trace_{user}_.ndjson`
  After fix:  `dedicated_log_triton_trace_{user}_pid_{PID}_.ndjson`
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import tempfile

import torch
import torch._inductor.config as inductor_config
import tritonparse.structured_logging
from tests.test_utils import GPUTestBase  # noqa: F401  (used via inheritance)


_PID_REGEX = re.compile(r"pid_(\d+)_")
_LOG_PREFIX = "dedicated_log_triton_trace_"

# Compile inline in the main process. Setting >1 would route compilation
# through inductor's compile-worker pool, whose default path forks workers
# after a pre_fork_setup() that initializes CUDA. That collides with
# upstream NVIDIA Triton 3.7.0+ on cu130 (cuInit returns
# CUDA_ERROR_NOT_INITIALIZED in the forked child, producing
# `RuntimeError: 0 active drivers ([])`) whenever ~/.triton/cache is cold.
# The multi-worker scenario is covered by tests/cpu/test_multiprocess_write.py;
# here we only need the main-process path through inductor.
COMPILE_THREADS = 1


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
                "TRITON_ALWAYS_COMPILE",
            )
        }

        os.environ["TRITON_TRACE"] = self.trace_dir
        os.environ["TRITON_TRACE_COMPRESSION"] = "none"
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(COMPILE_THREADS)
        # Skip Triton cache to guarantee actual compilation work.
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"

        # TORCHINDUCTOR_COMPILE_THREADS is read only once, on the first
        # call into inductor, by decide_compile_threads() via the lazy
        # get_compile_threads() path (torch/_inductor/async_compile.py).
        # If any earlier test in the same process has already triggered
        # inductor compilation, config.compile_threads is cached to the
        # CPU-count default and the env var becomes a no-op. Override
        # the config attribute directly so this test always runs inline,
        # regardless of test ordering.
        self._saved_compile_threads = inductor_config.compile_threads
        inductor_config.compile_threads = COMPILE_THREADS

        # Initialize tritonparse in the main process.
        tritonparse.structured_logging.init(self.trace_dir)

    def tearDown(self):
        inductor_config.compile_threads = self._saved_compile_threads
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
        """Run several distinct torch.compile units so inductor generates and
        compiles Triton kernels inline in the main process. Larger tensors and
        more ops per function keep ATen fallback from suppressing codegen.
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

        Scope: only validates the main process's trace file — inductor runs
        inline (TORCHINDUCTOR_COMPILE_THREADS=1) so there are no worker
        subprocesses to trace. The multi-worker cross-process file-sharing
        scenario is exhaustively covered by tests/cpu/test_multiprocess_write.py
        with explicit spawn workers.
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
            f"(multi-worker scenario is covered by "
            f"tests/cpu/test_multiprocess_write.py):{diagnostic}",
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
