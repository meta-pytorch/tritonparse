# C9 — RoPE accuracy change (paper §5.1)

## What the paper claims

A RoPE kernel's numerical accuracy changed when Triton was upgraded from 3.2 to 3.3.
TritonParse's cross-trace IR diff located the change in the generated PTX: Triton 3.2
emulated the multiplies with `fma.rn.bf16` against a zero addend, while Triton 3.3
emitted native `mul.bf16` / `neg.bf16` / `fma.rn.bf16`. The paper quantifies the effect:

> Out of 10000 tests, 25.84% produce results that differ between the two
> implementations.

## What can and cannot be re-run

**Cannot:** the Triton 3.2 → 3.3 comparison itself. Current TritonParse requires
`triton > 3.3.1` and cannot instrument Triton 3.2 at all, so there is no way to
re-capture the two traces. The PTX files here are **transcribed from the paper's
Listing 2**, not recovered from a compiler; each carries a provenance header saying so.

**Can:** the numerical claim. It is a property of two instruction sequences.
`bf16_rounding_study.py` runs both sequences *verbatim* as inline PTX inside a Triton
kernel and measures how often they disagree.

**Partly, below Hopper:** `mul.bf16` is an `sm_90` instruction — `ptxas` rejects it for
`sm_80` with *"requires .target sm_90 or higher"*. That restriction is not incidental to
this case study, it is the cause of it: Triton 3.2 emulated the multiply because the
hardware of the day had no such instruction, and `fma.rn.bf16` (which the 3.2 sequence
uses throughout) has been available since `sm_80`. On an Ampere or Ada GPU the script
therefore executes the 3.2 sequence as real PTX and evaluates the 3.3 side with the
semantic model below, prints which path it took, and writes `native_mul_bf16=false` to
its CSV. The divergence rate is unaffected. What is lost is one leg of the cross-check —
the model is validated against real `mul.bf16` PTX only where that instruction exists —
so on `sm_90` and above the script always uses the hardware and never the substitute.
Pass `--assume-no-native-mul` to exercise the pre-Hopper path on a newer GPU.

## Why they disagree

For `out = a*b - c*d`:

| | rounds `a*b`? | rounds `c*d`? | final rounding |
|---|---|---|---|
| Triton 3.2 (`fma` emulation) | **yes** — `fma(a, b, -0.0)` materialises a bf16 | yes | yes |
| Triton 3.3 (native `mul`) | **no** — stays unrounded inside the final `fma` | yes | yes |

The single difference is whether `a*b` is rounded to bf16 before the subtraction. With
bf16's 8-bit mantissa that is enough to change the result on roughly a quarter of random
inputs — which is exactly the effect the paper measured.

## Layout

```
ptx_triton32_fma_emulation.ptx   paper Listing 2, left column  (+ provenance header)
ptx_triton33_native_mul.ptx      paper Listing 2, right column (+ provenance header)
bf16_rounding_study.py           runs both, cross-checks, reports
```

## How it is checked

Two independent implementations are run and compared against **each other** as well as
against the paper:

- **inline PTX** — the literal instruction sequences, executing on the hardware BF16 units
- **semantic model** — the same rounding points expressed in PyTorch

Reporting their agreement matters: it shows the measured divergence is a property of the
arithmetic rather than of how the test was written. In our runs they agree on every one
of 50,000 samples — which is also what licenses the pre-Hopper substitution described
above.

## Measured result

On an H100, 10000 samples × 5 seeds:

```
   seed   PTX 3.2 vs 3.3        semantic model     PTX-vs-model mismatches
      0     2439/10000 = 24.39%    2439 = 24.39%          0
      1     2446/10000 = 24.46%    2446 = 24.46%          0
      2     2473/10000 = 24.73%    2473 = 24.73%          0
      3     2388/10000 = 23.88%    2388 = 23.88%          0
      4     2341/10000 = 23.41%    2341 = 23.41%          0

  DIVERGENCE : 24.17% +/- 0.66     (paper, 10000 tests: 25.84%)
```

The verdict criterion is divergence in 15–40 % **and** exact agreement between the two
implementations — not equality with 25.84 % to two decimals. Our inputs are random
normals; the paper's were the actual RoPE tensors, so the exact rate is not expected to
coincide.

## Running it standalone

```bash
python claims/C9_rope/bf16_rounding_study.py --csv results/c9_rope.csv
python claims/C9_rope/bf16_rounding_study.py --n 100000 --trials 10   # tighter estimate
```

It is also invoked automatically by `../../01_gpu_core.sh`.
