# Copyright (c) Meta Platforms, Inc. and affiliates.
"""CPU unit tests for the per-launch `roofline` event (synthetic TTIR + launches).

A compiled kernel is launched many times with different inputs/grids, so the
roofline (total bytes moved, FLOPs) is computed per launch by joining the per-CTA
TTIR template with each launch's grid and tensor arg sizes. These tests prove the
core logic — per-CTA byte counting, GEMM detection, the launch join (incl.
per-launch differentiation), env-var gating, and schema validation — without a
GPU, so `make test` always covers it.
"""

import os
import unittest

from tritonparse.backend import AnalyzerContext, get_backend_registry, LAUNCH_LEVEL
from tritonparse.parse.ir_analysis import (
    _bytes_from_launch_args,
    _generate_launch_analysis,
    _grid_num_ctas,
    _roofline_count_bytes,
    _roofline_flops,
    generate_roofline,
)
from tritonparse.validation.json_validator import validate_record

# ---------------------------------------------------------------------------
# Synthetic TTIR fixtures (shapes/dtypes chosen for exact arithmetic).
# ---------------------------------------------------------------------------

# Non-GEMM: two loads + one store of tensor<256x!tt.ptr<f32>>.
# Per-CTA bytes = (2 + 1) * 256 * 4 = 3072.
TTIR_ELEMENTWISE = """
module {
  tt.func public @add_kernel() {
    %9 = tt.load %8, %6 : tensor<256x!tt.ptr<f32>> loc(#loc8)
    %12 = tt.load %11, %6 : tensor<256x!tt.ptr<f32>> loc(#loc10)
    tt.store %15, %13, %6 : tensor<256x!tt.ptr<f32>> loc(#loc13)
    tt.return loc(#loc)
  }
}
"""

# GEMM: first tt.dot is 64x32 * 32x128 -> 64x128, plus parseable I/O.
# flops_per_cta = 2*M*N*K = 2*64*128*32 = 524288.
# Per-CTA bytes = a(64*32*2) + b(32*128*2) + c(64*128*2) = 4096 + 8192 + 16384 = 28672.
TTIR_GEMM = """
module {
  tt.func public @matmul_kernel() {
    %a = tt.load %ap : tensor<64x32x!tt.ptr<f16>> loc(#loc1)
    %b = tt.load %bp : tensor<32x128x!tt.ptr<f16>> loc(#loc2)
    %71 = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<64x32xf16> * tensor<32x128xf16> -> tensor<64x128xf32> loc(#loc39)
    tt.store %cp, %c : tensor<64x128x!tt.ptr<f16>> loc(#loc40)
    tt.return loc(#loc)
  }
}
"""

# Unknown dtype on an I/O line (mxfp scale type f8e8m0) -> per-CTA bytes null.
TTIR_UNKNOWN_DTYPE = """
module {
  tt.func public @weird_kernel() {
    %9 = tt.load %8 : tensor<128x!tt.ptr<f8e8m0>> loc(#loc8)
    tt.return loc(#loc)
  }
}
"""


def _compilation(ttir: str) -> dict:
    """Minimal compilation event carrying just the TTIR (what generate_roofline reads)."""
    return {
        "payload": {
            "file_content": {"kernel.ttir": ttir},
            "file_path": {},
            "metadata": {"name": "kernel", "backend_name": "cuda"},
        }
    }


def _launch(grid, numel, occ, n_tensors=3, element_size=4) -> dict:
    """A launch event with `n_tensors` tensor args of `numel` elements each."""
    args = {
        f"in_ptr{i}": {"numel": numel, "element_size": element_size}
        for i in range(n_tensors)
    }
    return {
        "event_type": "launch",
        "occurrence_id": occ,
        "grid": list(grid),
        "extracted_args": args,
    }


class TestRooflinePerCTA(unittest.TestCase):
    """The per-CTA helpers on synthetic TTIR (the launch-invariant template)."""

    def test_non_gemm_byte_count(self):
        total, breakdown, _ = _roofline_count_bytes(TTIR_ELEMENTWISE)
        self.assertEqual(total, 3072)
        self.assertEqual(breakdown, {"load": 2048, "store": 1024})
        is_gemm, flops, _ = _roofline_flops(TTIR_ELEMENTWISE)
        self.assertFalse(is_gemm)
        self.assertIsNone(flops)

    def test_gemm_flops_and_bytes(self):
        total, _, _ = _roofline_count_bytes(TTIR_GEMM)
        self.assertEqual(total, 28672)
        is_gemm, flops, _ = _roofline_flops(TTIR_GEMM)
        self.assertTrue(is_gemm)
        self.assertEqual(flops, 2 * 64 * 128 * 32)  # 524288

    def test_unknown_dtype_is_conservative_null(self):
        total, breakdown, notes = _roofline_count_bytes(TTIR_UNKNOWN_DTYPE)
        self.assertIsNone(total)
        self.assertIsNone(breakdown)
        self.assertTrue(any("dtype" in n for n in notes))


class TestRooflineLaunchHelpers(unittest.TestCase):
    """The launch-side helpers: grid product and arg-size byte sum."""

    def test_grid_num_ctas(self):
        self.assertEqual(_grid_num_ctas([4]), 4)
        self.assertEqual(_grid_num_ctas([4, 2, 3]), 24)
        self.assertIsNone(_grid_num_ctas([]))
        self.assertIsNone(_grid_num_ctas(None))

    def test_bytes_from_args_counts_tensors_only(self):
        args = {
            "in_ptr0": {"numel": 10, "element_size": 4},
            "out_ptr0": {"numel": 10, "element_size": 4},
            "n_elements": {"type": "int", "value": 10},  # scalar, ignored
        }
        self.assertEqual(_bytes_from_launch_args(args), 80)

    def test_bytes_from_args_doubles_in_out(self):
        args = {"in_out_ptr0": {"numel": 10, "element_size": 4}}
        self.assertEqual(_bytes_from_launch_args(args), 80)  # read + write

    def test_bytes_from_args_none_when_no_tensors(self):
        self.assertIsNone(_bytes_from_launch_args({}))
        self.assertIsNone(_bytes_from_launch_args({"n": {"type": "int", "value": 5}}))


class TestGenerateRoofline(unittest.TestCase):
    """generate_roofline: the per-launch join (the heart of the redesign)."""

    def setUp(self):
        self.original_env = os.environ.get("TRITONPARSE_ANALYSIS")

    def tearDown(self):
        if self.original_env is None:
            os.environ.pop("TRITONPARSE_ANALYSIS", None)
        else:
            os.environ["TRITONPARSE_ANALYSIS"] = self.original_env

    def test_per_launch_differentiation(self):
        """Same compiled kernel, two launches with different sizes -> different bytes."""
        comp = _compilation(TTIR_ELEMENTWISE)
        launches = [
            (_launch([1], 256, occ=10), 0),  # grid 1, 256-elem tensors
            (_launch([4], 1024, occ=11), 1),  # grid 4, 1024-elem tensors
        ]
        r = generate_roofline(comp, launches)
        self.assertIsNotNone(r)
        self.assertFalse(r["is_gemm"])
        self.assertEqual(r["estimation_method"], "ttir_static_x_launch")
        self.assertEqual(r["bytes_moved_per_cta"], 3072)
        self.assertEqual(len(r["per_launch"]), 2)

        pl0, pl1 = r["per_launch"]
        self.assertEqual(pl0["launch_index"], 0)
        self.assertEqual(pl0["occurrence_id"], 10)
        self.assertEqual(pl0["num_ctas"], 1)
        self.assertEqual(pl0["bytes_from_ir_x_grid"], 3072)  # 3072 * 1
        self.assertEqual(pl0["bytes_from_args"], 3072)  # 3 * 256 * 4
        self.assertEqual(pl0["bytes_moved"], 3072)

        self.assertEqual(pl1["num_ctas"], 4)
        self.assertEqual(pl1["bytes_from_ir_x_grid"], 12288)  # 3072 * 4
        self.assertEqual(pl1["bytes_from_args"], 12288)  # 3 * 1024 * 4
        self.assertEqual(pl1["bytes_moved"], 12288)

        # The whole point: per-launch bytes differ for the same compiled kernel.
        self.assertNotEqual(pl0["bytes_moved"], pl1["bytes_moved"])

    def test_gemm_per_launch_flops_and_intensity(self):
        comp = _compilation(TTIR_GEMM)
        # No extracted_args -> bytes come from IR*grid only.
        gemm_launch = {"event_type": "launch", "occurrence_id": 20, "grid": [8]}
        r = generate_roofline(comp, [(gemm_launch, 0)])
        self.assertTrue(r["is_gemm"])
        self.assertEqual(r["flops_per_cta"], 524288)
        pl = r["per_launch"][0]
        self.assertEqual(pl["num_ctas"], 8)
        self.assertEqual(pl["bytes_from_ir_x_grid"], 28672 * 8)
        self.assertIsNone(pl["bytes_from_args"])
        self.assertEqual(pl["bytes_moved"], 28672 * 8)
        self.assertEqual(pl["flops"], 524288 * 8)
        self.assertAlmostEqual(pl["arithmetic_intensity"], 524288 / 28672)
        # GEMM lower-bound caveat is recorded.
        self.assertTrue(any("lower bound" in n for n in r.get("notes", [])))

    def test_bytes_moved_takes_min(self):
        """bytes_moved is the smaller of IR*grid and arg-size (conservative)."""
        comp = _compilation(TTIR_ELEMENTWISE)  # per-CTA 3072
        # grid 4 -> IR*grid = 12288; args undercount (1 tiny tensor) -> 4 bytes.
        launch = {
            "event_type": "launch",
            "occurrence_id": 30,
            "grid": [4],
            "extracted_args": {"in_ptr0": {"numel": 1, "element_size": 4}},
        }
        pl = generate_roofline(comp, [(launch, 0)])["per_launch"][0]
        self.assertEqual(pl["bytes_from_ir_x_grid"], 12288)
        self.assertEqual(pl["bytes_from_args"], 4)
        self.assertEqual(pl["bytes_moved"], 4)  # min

    def test_null_template_still_uses_args(self):
        """Unparseable TTIR -> per-CTA null, but arg sizes still give a per-launch estimate."""
        comp = _compilation(TTIR_UNKNOWN_DTYPE)
        launch = _launch([2], 128, occ=40, n_tensors=1, element_size=1)
        r = generate_roofline(comp, [(launch, 0)])
        self.assertIsNone(r["bytes_moved_per_cta"])
        pl = r["per_launch"][0]
        self.assertIsNone(pl["bytes_from_ir_x_grid"])
        self.assertEqual(pl["bytes_from_args"], 128)
        self.assertEqual(pl["bytes_moved"], 128)

    def test_env_var_gating(self):
        """TRITONPARSE_ANALYSIS gates roofline through the launch-level dispatch.

        Gating now lives in the registry dispatch (_generate_launch_analysis),
        not inline in generate_roofline, so the env var is exercised there.
        """
        comp = _compilation(TTIR_ELEMENTWISE)
        ctx = AnalyzerContext(launches_with_indices=[(_launch([1], 256, occ=10), 0)])

        os.environ["TRITONPARSE_ANALYSIS"] = "roofline"
        self.assertIn("roofline", _generate_launch_analysis(comp, ctx))
        os.environ["TRITONPARSE_ANALYSIS"] = "none"
        self.assertEqual(_generate_launch_analysis(comp, ctx), {})
        os.environ["TRITONPARSE_ANALYSIS"] = "loop_schedules"
        self.assertNotIn("roofline", _generate_launch_analysis(comp, ctx))
        os.environ["TRITONPARSE_ANALYSIS"] = "all"
        self.assertIn("roofline", _generate_launch_analysis(comp, ctx))
        os.environ.pop("TRITONPARSE_ANALYSIS", None)
        self.assertIn("roofline", _generate_launch_analysis(comp, ctx))

    def test_none_when_no_launches_or_no_ttir(self):
        self.assertIsNone(generate_roofline(_compilation(TTIR_ELEMENTWISE), []))
        no_ttir = {"payload": {"file_content": {}, "file_path": {}}}
        self.assertIsNone(generate_roofline(no_ttir, [(_launch([1], 8, 1), 0)]))
        self.assertIsNone(generate_roofline(None, [(_launch([1], 8, 1), 0)]))


class TestRooflineLaunchAnalyzer(unittest.TestCase):
    """Roofline is a registered launch-level analyzer (not compile-level)."""

    def test_roofline_registered_as_launch_level(self):
        registry = get_backend_registry()
        for adapter_name in ("cuda_triton", "hip_triton", "cann_triton"):
            adapter = registry.resolve(adapter_name=adapter_name)
            self.assertIn("roofline", adapter.list_analyzer_keys())
            info = adapter._analysis_registry.get_analyzer_info("roofline")
            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info.level, LAUNCH_LEVEL)

    def test_roofline_split_by_level(self):
        """Roofline appears only in launch-level executables, not compile-level."""
        registry = get_backend_registry()
        file_content = {"kernel.ttir": "ttir content"}
        for adapter_name in ("cuda_triton", "hip_triton", "cann_triton"):
            adapter = registry.resolve(adapter_name=adapter_name)
            self.assertNotIn(
                "roofline", adapter.list_executable_analyzers(file_content)
            )
            self.assertIn(
                "roofline",
                adapter.list_executable_analyzers(file_content, level=LAUNCH_LEVEL),
            )


class TestRooflineSchema(unittest.TestCase):
    """The emitted `roofline` event validates; the strict contract bites."""

    @staticmethod
    def _event(roofline):
        return {"event_type": "roofline", "hash": "abc123", "roofline": roofline}

    def test_generate_roofline_output_validates(self):
        for ttir in (TTIR_ELEMENTWISE, TTIR_GEMM, TTIR_UNKNOWN_DTYPE):
            r = generate_roofline(_compilation(ttir), [(_launch([2], 64, occ=1), 0)])
            ok, errors = validate_record(self._event(r))
            self.assertTrue(ok, f"Unexpected errors for {ttir[:30]!r}: {errors}")

    def test_valid_full_record(self):
        rec = {
            "is_gemm": True,
            "estimation_method": "ttir_static_x_launch",
            "bytes_moved_per_cta": 28672,
            "flops_per_cta": 524288,
            "bytes_breakdown_per_cta": {"load": 12288, "store": 16384},
            "per_launch": [
                {
                    "launch_index": 0,
                    "occurrence_id": 9,
                    "grid": [8],
                    "num_ctas": 8,
                    "bytes_moved": 229376,
                    "bytes_from_ir_x_grid": 229376,
                    "bytes_from_args": None,
                    "flops": 4194304,
                    "arithmetic_intensity": 18.285714285714285,
                }
            ],
            "notes": [
                "gemm flops/intensity omit the runtime reduction-loop trip count (lower bound)"
            ],
        }
        ok, errors = validate_record(self._event(rec))
        self.assertTrue(ok, f"Unexpected errors: {errors}")

    def test_missing_required_field_rejected(self):
        rec = {
            "is_gemm": False,
            "estimation_method": "ttir_static_x_launch",
            "bytes_moved_per_cta": 0,
            "flops_per_cta": None,
            # per_launch intentionally omitted
        }
        ok, _ = validate_record(self._event(rec))
        self.assertFalse(ok)

    def test_estimation_method_enum_enforced(self):
        rec = {
            "is_gemm": False,
            "estimation_method": "ttir_static",  # old value, now invalid
            "bytes_moved_per_cta": 0,
            "flops_per_cta": None,
            "per_launch": [],
        }
        ok, _ = validate_record(self._event(rec))
        self.assertFalse(ok)

    def test_per_launch_entry_missing_required_rejected(self):
        rec = {
            "is_gemm": False,
            "estimation_method": "ttir_static_x_launch",
            "bytes_moved_per_cta": 0,
            "flops_per_cta": None,
            "per_launch": [
                {"launch_index": 0, "grid": [1], "num_ctas": 1}
            ],  # no bytes_moved
        }
        ok, _ = validate_record(self._event(rec))
        self.assertFalse(ok)

    def test_unknown_extra_key_rejected(self):
        rec = {
            "is_gemm": False,
            "estimation_method": "ttir_static_x_launch",
            "bytes_moved_per_cta": 0,
            "flops_per_cta": None,
            "per_launch": [],
            "surprise": 1,  # additionalProperties: false
        }
        ok, _ = validate_record(self._event(rec))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
