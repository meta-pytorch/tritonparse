# Copyright (c) Meta Platforms, Inc. and affiliates.
"""GPU end-to-end test for the per-launch `roofline` event.

Compiles & runs real Triton kernels, parses the trace with roofline enabled, and
asserts the real `roofline` event — including that two launches of the SAME
compiled kernel with different input sizes get different per-launch bytes_moved
(the whole point of computing roofline per launch). Gated by GPUTestBase +
@requires_gpu so it auto-skips when no GPU is present (keeps `make test` green)
and runs under `make test-cuda`. Mirrors
tests/gpu/test_structured_logging.py::test_whole_workflow.

Test Plan:
```
python -m unittest tests.gpu.test_roofline_e2e -v
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
from tests.test_utils import GPUTestBase, requires_gpu
from tritonparse.tools.compression import open_compressed_file


@requires_gpu
class TestRooflineE2E(GPUTestBase):
    """End-to-end per-launch roofline assertions on real non-GEMM and GEMM kernels."""

    def setUp(self):
        super().setUp()
        # Ensure roofline runs during parsing (save/restore around the test).
        self._original_env = os.environ.get("TRITONPARSE_ANALYSIS")
        os.environ["TRITONPARSE_ANALYSIS"] = "roofline"

    def tearDown(self):
        if self._original_env is None:
            os.environ.pop("TRITONPARSE_ANALYSIS", None)
        else:
            os.environ["TRITONPARSE_ANALYSIS"] = self._original_env
        super().tearDown()

    def _run_and_collect_rooflines(self, run_fn):
        """Init logging, run the kernel(s), parse, and return roofline payloads."""
        temp_dir = tempfile.mkdtemp()
        logs = os.path.join(temp_dir, "logs")
        parsed = os.path.join(temp_dir, "parsed")
        os.makedirs(logs, exist_ok=True)
        os.makedirs(parsed, exist_ok=True)

        tritonparse.structured_logging.init(logs, enable_trace_launch=True)
        run_fn()
        torch.cuda.synchronize()

        tritonparse.parse.utils.unified_parse(source=logs, out=parsed, overwrite=True)

        rooflines = []
        for fname in os.listdir(parsed):
            if not (fname.endswith(".ndjson") or fname.endswith(".ndjson.gz")):
                continue
            with open_compressed_file(os.path.join(parsed, fname)) as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    if event.get("event_type") == "roofline":
                        rooflines.append(event["roofline"])
        return rooflines

    def test_non_gemm_per_launch_bytes(self):
        """Elementwise add launched at two sizes -> two per-launch entries with
        different bytes_moved (~ 3 * n * 4)."""

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        block_size = 256

        def launch(n):
            x = torch.randn(n, device=self.cuda_device, dtype=torch.float32)
            y = torch.randn(n, device=self.cuda_device, dtype=torch.float32)
            out = torch.empty_like(x)
            add_kernel[(triton.cdiv(n, block_size),)](x, y, out, n, block_size)

        def run():
            # Same compiled kernel, two different problem sizes -> two launches.
            launch(1024)
            launch(2048)

        rooflines = self._run_and_collect_rooflines(run)
        self.assertTrue(rooflines, "No roofline event found")
        rf = rooflines[0]

        self.assertIs(rf["is_gemm"], False)
        self.assertEqual(rf["estimation_method"], "ttir_static_x_launch")
        self.assertIsNotNone(rf["bytes_moved_per_cta"])

        per_launch = rf["per_launch"]
        self.assertGreaterEqual(
            len(per_launch), 2, "expected >=2 launches recorded for this kernel"
        )
        sizes = sorted({pl["bytes_moved"] for pl in per_launch})
        # The whole point: per-launch totals differ for the same compiled kernel.
        self.assertGreaterEqual(
            len(sizes), 2, f"per-launch bytes_moved did not differ: {sizes}"
        )
        # Hand-computed totals: 2 loads + 1 store of n f32 elements.
        self.assertIn(3 * 1024 * 4, sizes)
        self.assertIn(3 * 2048 * 4, sizes)

        # Each entry carries grid / num_ctas / a positive byte total.
        for pl in per_launch:
            self.assertIsNotNone(pl["grid"])
            self.assertGreater(pl["num_ctas"], 0)
            self.assertGreater(pl["bytes_moved"], 0)

    def test_gemm_per_launch_flops(self):
        """matmul with tl.dot -> is_gemm True, per-launch flops > 0 and positive AI."""

        @triton.jit
        def matmul_kernel(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(axis=0)
            pid_n = tl.program_id(axis=1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc += tl.dot(a, b)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            c = acc.to(tl.float16)
            c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
            tl.store(c_ptrs, c)

        def run():
            m = n = k = 64
            block = 32
            a = torch.randn((m, k), device=self.cuda_device, dtype=torch.float16)
            b = torch.randn((k, n), device=self.cuda_device, dtype=torch.float16)
            c = torch.empty((m, n), device=self.cuda_device, dtype=torch.float16)
            grid = (triton.cdiv(m, block), triton.cdiv(n, block))
            matmul_kernel[grid](
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                block,
                block,
                block,
            )

        rooflines = self._run_and_collect_rooflines(run)
        gemms = [rf for rf in rooflines if rf["is_gemm"]]
        self.assertTrue(gemms, "No GEMM roofline event found")
        rf = gemms[0]

        self.assertIs(rf["is_gemm"], True)
        self.assertIsNotNone(rf["flops_per_cta"])
        self.assertGreater(rf["flops_per_cta"], 0)
        self.assertTrue(rf["per_launch"], "GEMM roofline has no per-launch entries")
        for pl in rf["per_launch"]:
            self.assertGreater(pl["num_ctas"], 0)
            self.assertGreater(pl["flops"], 0)
            self.assertIsInstance(pl["arithmetic_intensity"], float)
            self.assertGreater(pl["arithmetic_intensity"], 0.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
