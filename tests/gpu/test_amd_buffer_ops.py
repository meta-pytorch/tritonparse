# Copyright (c) Meta Platforms, Inc. and affiliates.
"""GPU end-to-end test for AMD buffer-ops analysis on real hardware.

Compiles & runs a real Triton kernel, parses the trace, and asserts the
derived `amd_buffer_ops` payload end to end: on an AMD GPU the kernel
must compile to AMDGCN, and the analyzer must emit a well-formed status
for it. AMD-only via `skip_unless_amd` (no AMDGCN exists elsewhere);
auto-skips on machines without any GPU via GPUTestBase. Mirrors
tests/gpu/test_roofline_e2e.py.

The assertions stay at wiring level (field present, status in the
documented enum, `enabled` consistent with status) instead of pinning
an exact status — the compiler's choice of buffer vs global ops per
kernel is version-dependent. Pin exact values only after observing
them in CI logs.

Test Plan:
```
python -m unittest tests.gpu.test_amd_buffer_ops -v   # skips without AMD GPU
```
"""

import json
import os
import tempfile

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import tritonparse.parse.utils
import tritonparse.structured_logging
from tests.test_utils import GPUTestBase, skip_unless_amd
from tritonparse.tools.compression import open_compressed_file

_AMD_BUFFER_OPS_STATUSES = frozenset({"all_buffer", "partial", "none", "unknown"})


def _collect_statuses(parsed):
    """Scan a parsed dir for AMDGCN compilations and amd_buffer_ops payloads.

    Returns (compilations_with_amdgcn, statuses). Pure artifact scan —
    unit-testable without a GPU.
    """
    statuses = []
    compilations_with_amdgcn = 0
    for fname in os.listdir(parsed):
        if not (fname.endswith(".ndjson") or fname.endswith(".ndjson.gz")):
            continue
        with open_compressed_file(os.path.join(parsed, fname)) as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                except (json.JSONDecodeError, AttributeError):
                    continue
                if event.get("event_type") == "compilation":
                    # Compilation artifacts live in payload.file_content
                    # keyed by filename (e.g. kernel.amdgcn) — same
                    # convention as the .sass scan in
                    # test_structured_logging.py.
                    file_content = event.get("payload", {}).get("file_content", {})
                    if any(key.endswith(".amdgcn") for key in file_content):
                        compilations_with_amdgcn += 1
                if event.get("event_type") == "ir_analysis":
                    payload = event.get("ir_analysis", {})
                    if "amd_buffer_ops" in payload:
                        statuses.append(payload["amd_buffer_ops"])
    return compilations_with_amdgcn, statuses


@skip_unless_amd
class TestAmdBufferOpsE2E(GPUTestBase):
    """AMD buffer-ops derivation on live gfx9xx hardware."""

    def _run_and_collect_statuses(self, run_fn):
        """Init logging, run the kernel(s), parse, return amd_buffer_ops payloads."""
        temp_dir = tempfile.mkdtemp()
        logs = os.path.join(temp_dir, "logs")
        parsed = os.path.join(temp_dir, "parsed")
        os.makedirs(logs, exist_ok=True)
        os.makedirs(parsed, exist_ok=True)

        # Default TRITONPARSE_ANALYSIS ("all") already includes
        # amd_buffer_ops whenever TTGIR + AMDGCN stages exist.
        tritonparse.structured_logging.init(logs, enable_trace_launch=True)
        run_fn()
        torch.cuda.synchronize()

        tritonparse.parse.utils.unified_parse(source=logs, out=parsed, overwrite=True)
        return _collect_statuses(parsed)

    def test_amd_buffer_ops_present(self):
        """Elementwise add on AMD GPU yields a well-formed status payload."""

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        def run():
            n = 1024
            block_size = 256
            x = torch.randn(n, device=self.cuda_device, dtype=torch.float32)
            y = torch.randn(n, device=self.cuda_device, dtype=torch.float32)
            out = torch.empty_like(x)
            add_kernel[(triton.cdiv(n, block_size),)](x, y, out, n, block_size)

        compilations, statuses = self._run_and_collect_statuses(run)
        self.assertGreater(
            compilations,
            0,
            "Expected a compilation with an AMDGCN stage on AMD GPU",
        )
        self.assertGreater(
            len(statuses), 0, "Expected amd_buffer_ops in ir_analysis events"
        )
        for status in statuses:
            self.assertIn(status.get("status"), _AMD_BUFFER_OPS_STATUSES)
            self.assertIsInstance(status.get("enabled"), bool)
            self.assertEqual(status["enabled"], status["status"] == "all_buffer")
            self.assertTrue(status.get("reason"))
            print(f"amd_buffer_ops payload: {status}")


if __name__ == "__main__":
    import unittest

    unittest.main()
