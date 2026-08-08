# C8 — Bug reproduction (paper §4.4)

## What the paper claims

Three failure classes — a **device-side assertion**, an **illegal memory access (IMA)**,
and a **hang** — are injected into five `gpt-oss-20b` Triton kernels, giving 15
mutations.  TritonParse generates a minimal reproducer for each and reproduces the
failure in **100 %** of cases (paper §4.4; the mutations themselves are paper
Listing 1).

## What this check does, and why it is a fair test

Reviewers cannot realistically run the paper's version of this: `gpt-oss-20b` is a
gated download of tens of GB and needs MXFP4 kernels.

The three mutations, however, are **independent of kernel size** — they are three source
edits, reproduced verbatim from Listing 1:

| Mutation | Edit | Paper Listing 1 |
|---|---|---|
| `assert` | `assert stride_ym == 192, "stride check failed"` | identical |
| `ima` | `offs = tl.full((BLOCK,), BLOCK * 1000000, dtype=tl.int32)` on an unmasked load | `offs_y_n = tl.full((N_EXPTS_ACT,), N_EXPTS_ACT * 1000000, dtype=tl.int32)` |
| `hang` | `if pid == 0: while pid == pid: tl.atomic_add(...)` | identical |

Applying them to a ~30-line kernel exercises **the same code path** as the paper:
error classification, the 30 s hang timeout, trace capture across a fatal fault, and
reproducer template filling.  Only the kernel is smaller.

The paper's own `gpt-oss-20b` runs are not shipped: the model is a gated,
multi-tens-of-GB download and the traces are correspondingly large. What is reproduced
here is the mechanism, at a size a reviewer can actually run.

## Layout

```
toy_kernel.py          clean baseline; runs correctly, records a trace
patches/assert.patch   \
patches/ima.patch       >  literal diffs, readable side-by-side with Listing 1
patches/hang.patch     /
run.sh                 inject -> record -> parse -> generate -> execute -> compare
```

`run.sh` is invoked automatically by `../../01_gpu_core.sh`; it can also be run alone.

## Pipeline, per mutation

1. **Inject** — `patch` the baseline in a scratch directory.
2. **Record** — run the mutated workload under TritonParse with a 30 s timeout (the
   paper's default for hang classification).  Two of the three mutations kill or wedge
   the process.
3. **Check the trace survived** — this is the paper's key design point: traces are
   written *before* the launch, so even a workload that never returns leaves a
   complete, parseable trace.  Measured here: 31 KB of trace from the hang case.
4. **Parse** — `tritonparseoss parse`.
5. **Generate** — `tritonparseoss reproduce ... --kernel-import copy --embed-context`.
   Both flags are required; the default import mode emits
   `from <original module> import <kernel>`, which cannot resolve elsewhere.
6. **Execute** — run the reproducer **from a clean directory**, so a reproducer that
   secretly depends on the recording tree cannot pass.
7. **Compare** — the original run and the reproducer must land in the same fault class.

## Reading the verdict

Matching is by **fault class**, not by exact CUDA error code.  This is a deliberate,
necessary choice: for the IMA mutation the same binary has been observed reporting both
`cudaErrorIllegalAddress` and `cudaErrorInvalidAddressSpace` across runs, because the
faulting address depends on where PyTorch's caching allocator places the tensors.  Both
are out-of-bounds faults.  `run.sh` prints the codes it actually saw, so the variance is
visible rather than hidden.

Related detail worth knowing if you modify the patches: the out-of-bounds offset has to
be *large*.  At 10^6 elements the access still lands inside the allocator's slab and does
not fault at all — which is why the paper multiplies by 1000000.

## Expected output

```
 BUG      ORIGINAL RUN                        REPRODUCER                          VERDICT
 assert   assert (cudaErrorAssert)            assert (cudaErrorAssert)            PASS
 ima      memfault (cudaErrorIllegalAddress)  memfault (cudaErrorIllegalAddress)  PASS
 hang     hang                                hang                                PASS
 3/3 bug classes reproduced   (paper 4.4: 15/15 mutations on gpt-oss-20b)
```

Results are also written to `results/c8_bugs.csv`.
