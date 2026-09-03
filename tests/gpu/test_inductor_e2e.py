# Copyright (c) Meta Platforms, Inc. and affiliates.
"""End-to-end tracing of inductor-generated Triton kernels.

torch.compile is how most traces in the wild are produced, but the suite only
covered it obliquely: test_multiprocess_write_inductor asserts on filenames
without parsing, and test_context_manager counts output files without looking
inside them. Nothing checked that an inductor kernel's frame attribution and
generated source actually survive into the parsed trace.

Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python -m unittest tests.gpu.test_inductor_e2e -v
```
"""

import glob
import os
import re
import shutil
import tempfile

import torch
import torch._inductor.config as inductor_config
import tritonparse.context_manager
from tests.test_utils import GPUTestBase
from tritonparse._json_compat import loads
from tritonparse.shared_vars import TEST_KEEP_OUTPUT
from tritonparse.tools.compression import open_compressed_file


# Codegen must stay inline. Above 1, inductor forks compile workers after a
# pre_fork_setup() that initializes CUDA, which on cu130 makes cuInit return
# CUDA_ERROR_NOT_INITIALIZED in the child whenever ~/.triton/cache is cold.
# See the longer note in tests/gpu/test_multiprocess_write_inductor.py.
COMPILE_THREADS = 1

# f{frame_id}_fc{frame_compile_id}_a{attempt}_cai{compiled_autograd_id}
_FRAME_FILE = re.compile(r"^f(\d+)_fc(\d+)_a(\d+)_cai.*\.ndjson(?:\.(?:gz|zst))?$")


def _pointwise(x):
    return (x * 2 + 1).relu()


def _reduction(x):
    return torch.softmax(x, dim=-1).sum()


class InductorTracingTest(GPUTestBase):
    """Parsed-trace assertions for kernels inductor generated."""

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp(prefix="tritonparse_inductor_e2e_")
        self._saved_threads = inductor_config.compile_threads
        inductor_config.compile_threads = COMPILE_THREADS

    def tearDown(self):
        inductor_config.compile_threads = self._saved_threads
        if TEST_KEEP_OUTPUT:
            print(f"Preserving output (TEST_KEEP_OUTPUT=1): {self.temp_dir}")
        else:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    def _trace(self, split_inductor_compilations):
        """Compile two functions under tracing; return the parsed output dir."""
        out_dir = os.path.join(
            self.temp_dir, f"parsed_split_{split_inductor_compilations}"
        )
        os.makedirs(out_dir, exist_ok=True)
        with tritonparse.context_manager.TritonParseManager(
            enable_trace_launch=True,
            split_inductor_compilations=split_inductor_compilations,
            out=out_dir,
        ):
            x = torch.randn(512, 512, device=self.cuda_device)
            torch.compiler.reset()
            # Without this, an FX graph cache hit would replay the kernels and
            # emit no compilation events at all.
            with inductor_config.patch(force_disable_caches=True):
                for fn in (_pointwise, _reduction):
                    torch.compile(fn, dynamic=False, fullgraph=True)(x)
            torch.cuda.synchronize()
        return out_dir

    @staticmethod
    def _read_events(path):
        with open_compressed_file(path) as fh:
            return [loads(line) for line in fh if line.strip()]

    def _compilations(self, out_dir):
        events = []
        for path in sorted(glob.glob(os.path.join(out_dir, "*.ndjson*"))):
            events.extend(self._read_events(path))
        comps = [e for e in events if e.get("event_type") == "compilation"]
        self.assertTrue(comps, f"no compilation events in {out_dir}")
        return comps

    def test_inductor_kernels_are_traced(self):
        """Both kernel shapes inductor emits reach the parsed trace."""
        comps = self._compilations(self._trace(split_inductor_compilations=False))
        names = {c["payload"]["metadata"]["name"] for c in comps}

        # Inductor names its kernels triton_{poi,per,red}_fused_*; the exact
        # suffix depends on the fusion decisions, so match the prefix only.
        self.assertTrue(
            all(n.startswith("triton_") for n in names),
            f"expected only inductor-generated kernels, got {names}",
        )
        self.assertTrue(
            any(n.startswith("triton_poi_") for n in names),
            f"expected a pointwise kernel, got {names}",
        )
        self.assertTrue(
            any(n.startswith(("triton_per_", "triton_red_")) for n in names),
            f"expected a reduction kernel, got {names}",
        )

    def test_compilations_carry_frame_attribution(self):
        """pt_info ties each kernel back to its torch.compile frame.

        This is what the split-by-frame output and tlparse cross-linking are
        built on, and it is the field that identifies a trace as inductor's.
        """
        comps = self._compilations(self._trace(split_inductor_compilations=False))
        frame_ids = set()
        for comp in comps:
            name = comp["payload"]["metadata"]["name"]
            with self.subTest(kernel=name):
                pt_info = comp["payload"].get("pt_info")
                if pt_info is None:
                    self.fail(f"{name} has no pt_info")
                self.assertIsInstance(pt_info.get("frame_id"), int)
                self.assertIsInstance(pt_info.get("frame_compile_id"), int)
                frame_ids.add(pt_info.get("frame_id"))

        # The two compiled functions are separate frames, so more than one
        # frame_id must appear -- otherwise attribution collapsed them.
        self.assertGreater(len(frame_ids), 1, f"expected >1 frame, got {frame_ids}")

    def test_python_source_is_inductor_generated_code(self):
        """The captured source is output_code.py, not the user's function."""
        comps = self._compilations(self._trace(split_inductor_compilations=False))
        for comp in comps:
            name = comp["payload"]["metadata"]["name"]
            with self.subTest(kernel=name):
                source = comp["payload"].get("python_source") or {}
                self.assertIn("torchinductor", source.get("file_path", ""))
                self.assertIn("@triton.jit", source.get("code", ""))

    def test_split_produces_one_file_per_frame(self):
        """split=True is the tlparse layout: a file per frame/compile id."""
        out_dir = self._trace(split_inductor_compilations=True)
        names = [os.path.basename(p) for p in glob.glob(os.path.join(out_dir, "*"))]
        frame_files = [n for n in names if _FRAME_FILE.match(n)]
        self.assertGreater(
            len(frame_files), 1, f"expected several per-frame files, got {names}"
        )

        # Each file must hold exactly the frame its name claims.
        for fname in frame_files:
            frame_id = int(_FRAME_FILE.match(fname).group(1))
            with self.subTest(file=fname):
                comps = [
                    e
                    for e in self._read_events(os.path.join(out_dir, fname))
                    if e.get("event_type") == "compilation"
                ]
                self.assertTrue(comps, f"{fname} has no compilations")
                for comp in comps:
                    self.assertEqual(comp["payload"]["pt_info"]["frame_id"], frame_id)

    def test_no_split_merges_every_frame_into_one_file(self):
        """split=False is what the web viewer needs: a single mapped file."""
        out_dir = self._trace(split_inductor_compilations=False)
        mapped = glob.glob(os.path.join(out_dir, "*_mapped.ndjson*"))
        self.assertEqual(
            len(mapped), 1, f"expected exactly one mapped file, got {mapped}"
        )
        frame_ids = {
            e["payload"]["pt_info"]["frame_id"]
            for e in self._read_events(mapped[0])
            if e.get("event_type") == "compilation"
        }
        self.assertGreater(
            len(frame_ids), 1, "the single file must hold every frame, not just one"
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
