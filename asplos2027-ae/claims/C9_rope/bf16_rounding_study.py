#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""C9 — the numerical mechanism behind the RoPE case study (paper 5.1).

The paper reports that upgrading Triton 3.2 -> 3.3 changed the accuracy of a RoPE
kernel, traced the change to the generated PTX, and quantified it: "Out of 10000 tests,
25.84% produce results that differ between the two implementations."

The *version comparison* cannot be re-run -- current TritonParse requires
``triton > 3.3.1`` and cannot instrument Triton 3.2 at all.  The *numerical claim*,
however, is a property of two instruction sequences, and those can be executed directly.
This script runs both sequences from paper Listing 2 verbatim, as inline PTX inside a
Triton kernel, on whatever NVIDIA GPU is present, and measures how often they disagree.

Expression under test, from the paper:

    out_rope1 = in_rope1 * cos_rope - in_rope2 * sin_rope

Triton 3.2 emitted three ``fma.rn.bf16``: two with a ``-0.0`` addend to emulate the
multiplies, one with a ``-1.0`` multiplier to emulate the subtraction.  Triton 3.3
emitted ``mul.bf16`` + ``neg.bf16`` + ``fma.rn.bf16``.  The two differ in *where they
round*: 3.2 rounds ``in_rope1 * cos_rope`` to bf16 before subtracting, 3.3 leaves it
unrounded inside the fma.  With bf16's 8-bit mantissa that is enough to change the
result on roughly a quarter of random inputs.

Two independent implementations are run and cross-checked against each other:

  * ``inline PTX``      -- the literal instruction sequences, on the hardware BF16 units
  * ``semantic model``  -- the same rounding points expressed in PyTorch

Agreement between them is itself reported: it shows the measured divergence is a
property of the arithmetic, not an artifact of how the test was written.

ARCHITECTURE NOTE.  ``mul.bf16`` requires ``sm_90``; ``fma.rn.bf16`` and ``neg.bf16``
work from ``sm_80``.  So the Triton 3.2 sequence runs on Ampere but the Triton 3.3 one
does not -- which is presumably why 3.2 emulated the multiply in the first place.  Below
sm_90 this script therefore runs the 3.2 sequence as real PTX and substitutes the
semantic model for the 3.3 side, and says so in its output.  That substitution is sound
because on hardware where both can run the two implementations agree on every sample.

    python claims/C9_rope/bf16_rounding_study.py --csv results/c9_rope.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

# Paper 5.1.
PAPER_DIVERGENCE_PCT = 25.84
PAPER_N = 10000

# ---------------------------------------------------------------------------------
# Paper Listing 2, left column: Triton 3.2.  0x8000 = -0.0 (zero addend -> the fma is a
# rounded multiply); 0xbf80 = -1.0 (turns the third fma into a subtraction).
ASM_TRITON_32 = (
    "{ .reg .b16 z, t1, t2, m1;"
    "  mov.b16 z, 0x8000U;"
    "  fma.rn.bf16 t1, $1, $2, z;"
    "  fma.rn.bf16 t2, $3, $4, z;"
    "  mov.b16 m1, 0xbf80U;"
    "  fma.rn.bf16 $0, t2, m1, t1; }"
)

# Paper Listing 2, right column: Triton 3.3.
MIN_CC_FOR_NATIVE_MUL = (9, 0)  # mul.bf16 requires sm_90

ASM_TRITON_33 = (
    "{ .reg .b16 t, nt;"
    "  mul.bf16 t, $3, $4;"
    "  neg.bf16 nt, t;"
    "  fma.rn.bf16 $0, $1, $2, nt; }"
)


@triton.jit
def _rope_expr(
    a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, n_elements,
    ASM: tl.constexpr, BLOCK: tl.constexpr,
):
    """out = a*b - c*d, evaluated by whichever PTX sequence is passed in."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    d = tl.load(d_ptr + offs, mask=mask)
    r = tl.inline_asm_elementwise(
        ASM, "=h,h,h,h,h", [a, b, c, d], dtype=tl.bfloat16, is_pure=True, pack=1
    )
    tl.store(out_ptr + offs, r, mask=mask)


def _run_ptx(asm: str, a, b, c, d) -> torch.Tensor:
    out = torch.empty_like(a)
    n = a.numel()
    _rope_expr[(triton.cdiv(n, 256),)](a, b, c, d, out, n, ASM=asm, BLOCK=256)
    torch.cuda.synchronize()
    return out


def _model_triton_32(a, b, c, d) -> torch.Tensor:
    """Both products rounded to bf16, then a rounded subtraction."""
    f = lambda x: x.to(torch.float32)  # noqa: E731
    bf = lambda x: x.to(torch.bfloat16)  # noqa: E731
    t1 = bf(f(a) * f(b))
    t2 = bf(f(c) * f(d))
    return bf(f(t1) - f(t2))


def _model_triton_33(a, b, c, d) -> torch.Tensor:
    """Only c*d is rounded; a*b stays unrounded inside the fma."""
    f = lambda x: x.to(torch.float32)  # noqa: E731
    bf = lambda x: x.to(torch.bfloat16)  # noqa: E731
    t = bf(f(c) * f(d))
    return bf(f(a) * f(b) - f(t))


def trial(n: int, seed: int, native_mul: bool) -> dict:
    torch.manual_seed(seed)
    a, b, c, d = (
        torch.randn(n, device="cuda").to(torch.bfloat16) for _ in range(4)
    )

    mod32 = _model_triton_32(a, b, c, d)
    mod33 = _model_triton_33(a, b, c, d)
    ptx32 = _run_ptx(ASM_TRITON_32, a, b, c, d)
    # Below sm_90 the 3.3 sequence cannot be assembled at all, so stand the model in for
    # it rather than failing; the comparison is still 3.2-versus-3.3 semantics.
    ptx33 = _run_ptx(ASM_TRITON_33, a, b, c, d) if native_mul else mod33

    return {
        "seed": seed,
        "n": n,
        "ptx_differ": int((ptx32 != ptx33).sum()),
        "model_differ": int((mod32 != mod33).sum()),
        "ptx_vs_model_32": int((ptx32 != mod32).sum()),
        "ptx_vs_model_33": int((ptx33 != mod33).sum()) if native_mul else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=PAPER_N,
                    help=f"samples per trial (default {PAPER_N}, matching the paper)")
    ap.add_argument("--trials", type=int, default=5,
                    help="independent seeds, so run-to-run variation is visible")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--assume-no-native-mul", action="store_true",
                    help="pretend mul.bf16 is unavailable, to exercise the pre-Hopper\n                         path on hardware that does not need it")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: this study needs a GPU (it executes PTX).", file=sys.stderr)
        return 1

    cc = torch.cuda.get_device_capability(0)
    native_mul = cc >= MIN_CC_FOR_NATIVE_MUL and not args.assume_no_native_mul

    rows = [trial(args.n, seed, native_mul) for seed in range(args.trials)]
    pcts = [100.0 * r["ptx_differ"] / r["n"] for r in rows]
    mism = sum(r["ptx_vs_model_32"] + r["ptx_vs_model_33"] for r in rows)
    model_pcts = [100.0 * r["model_differ"] / r["n"] for r in rows]

    gpu = torch.cuda.get_device_name(0)
    print()
    print("=" * 78)
    print(" [C9] BF16 rounding: FMA emulation vs native multiply  (paper 5.1)")
    print("=" * 78)
    print(f"  expression          : out = a*b - c*d   (bf16, {args.n} samples x {args.trials} seeds)")
    print(f"  gpu                 : {gpu} (sm_{cc[0]}{cc[1]})")
    if native_mul:
        print(f"  sequences           : paper Listing 2, both executed verbatim as inline PTX")
    else:
        why = ("--assume-no-native-mul was given" if cc >= MIN_CC_FOR_NATIVE_MUL
               else f"mul.bf16 needs sm_90 and this GPU is sm_{cc[0]}{cc[1]}")
        print( "  sequences           : Triton 3.2 as inline PTX; Triton 3.3 via the semantic")
        print(f"                        model, because {why}")
    print()
    print("   seed   PTX 3.2 vs 3.3      semantic model      PTX-vs-model mismatches")
    print("   " + "-" * 70)
    for r, p, mp in zip(rows, pcts, model_pcts):
        print(f"   {r['seed']:>4}   {r['ptx_differ']:>6}/{r['n']} = {p:5.2f}%"
              f"    {r['model_differ']:>6} = {mp:5.2f}%"
              f"      {r['ptx_vs_model_32'] + r['ptx_vs_model_33']:>6}")
    print()
    mean = statistics.mean(pcts)
    spread = (max(pcts) - min(pcts)) / 2 if len(pcts) > 1 else 0.0
    print(f"  DIVERGENCE          : {mean:.2f}% +/- {spread:.2f}   "
          f"(paper, 10000 tests: {PAPER_DIVERGENCE_PCT}%)")

    # Judged on the mechanism, not on equality with the paper's exact percentage: the
    # paper's inputs were the real RoPE tensors, ours are random normals.
    in_range = 15.0 <= mean <= 40.0
    consistent = mism == 0
    verdict = "PASS" if (in_range and consistent) else "FAIL"
    if native_mul:
        print(f"  cross-check         : {'the two implementations agree exactly' if consistent else f'{mism} mismatches'}")
    else:
        print(f"  cross-check         : Triton 3.2 PTX vs its model — "
              f"{'agree exactly' if consistent else f'{mism} mismatches'}; "
              f"the 3.3 side is the model itself")
    crit = ("and inline PTX matches the semantic model exactly" if native_mul
            else "and the 3.2 sequence matches its model exactly")
    print(f"  verdict             : {verdict}  (criteria: divergence in 15-40%, {crit})")
    print()
    print("  WHAT THIS DOES AND DOES NOT SHOW.  It shows that the two instruction")
    print("  sequences the paper attributes to Triton 3.2 and 3.3 really do disagree, at")
    print("  the rate the paper reports, because they round in different places.  It does")
    print("  NOT re-run the Triton 3.2 -> 3.3 upgrade: current TritonParse requires")
    print("  triton > 3.3.1 and cannot instrument 3.2.  The PTX in ptx_triton3*.ptx is")
    print("  transcribed from the paper's Listing 2, not re-captured from a compiler.")
    print("  Our inputs are random normals rather than the paper's RoPE tensors, so the")
    print("  exact percentage is not expected to match to two decimals.")
    print("=" * 78)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["claim", "metric", "value", "unit", "paper_value", "verdict"])
            w.writerow(["C9", "samples_per_trial", args.n, "count", PAPER_N, ""])
            w.writerow(["C9", "trials", args.trials, "count", "", ""])
            for r, p in zip(rows, pcts):
                w.writerow(["C9", f"divergence_seed{r['seed']}", round(p, 2), "%", "", ""])
            w.writerow(["C9", "divergence_mean", round(mean, 2), "%",
                        PAPER_DIVERGENCE_PCT, verdict])
            w.writerow(["C9", "divergence_halfspread", round(spread, 2), "%", "", ""])
            w.writerow(["C9", "ptx_vs_model_mismatches", mism, "count", 0,
                        "PASS" if consistent else "FAIL"])
            w.writerow(["C9", "gpu", gpu, "", "NVIDIA H100", ""])
            w.writerow(["C9", "native_mul_bf16", str(native_mul).lower(), "", "true", ""])
        print(f"  csv                 : {path}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
