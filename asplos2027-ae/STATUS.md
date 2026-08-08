# Badges claimed

For **“TritonParse: Multi-IR Provenance and Reproducible Debugging for Triton Kernel
Compilation”**, ASPLOS'27 submission #461.

| Badge | Claimed | |
|---|---|---|
| **Artifacts Available** | ✅ yes | |
| **Artifacts Evaluated — Functional** | ✅ yes | |
| Results Reproduced | ❌ no | see below |

---

## Artifacts Available

TritonParse is developed in the open at
<https://github.com/meta-pytorch/tritonparse> under a BSD-3-Clause licence, and is
published on PyPI as `tritonparse`.

**During evaluation** reviewers work from the branch
[`asplos2027-ae`](https://github.com/meta-pytorch/tritonparse/tree/asplos2027-ae), so
that anything they report and we fix reaches them with a `git pull` rather than through a
new tag they have to be told about. The branch is only ever fast-forwarded, and every run
stamps the branch and commit it ran into its own output, so a reviewer's log identifies
the tree without our having to ask.

**For the badge** an archival deposit with a DOI is required — a git host alone does not
qualify — and that deposit is made *after* evaluation, so that what is archived is what
was evaluated, fixes included:

- **Reserved DOI:** `10.5281/zenodo.21853863` — reserved, not yet published; the record
  is empty until the deposit is made.
- **Contents:** the tree of the tag cut at that point — full source plus this
  `asplos2027-ae/` directory, reference trace included, self-contained.

We would rather name the timing plainly than print a DOI that does not resolve yet and
let a reviewer discover the gap by clicking it.

## Artifacts Evaluated — Functional

The four ACM criteria, and where each is satisfied:

**Documented.** [`README.md`](README.md) is the single entry point: it maps every paper
claim to the script that exercises it and to the output you should see, and it also
carries the hardware requirement, installation, the CUDA-wheel choice and the version
delta against the paper. Each check additionally prints what it is testing and why, so
the console output stands on its own.

**Consistent.** Every quantitative check prints its own measurement *next to the paper's*
and states the criterion it is judged on. Continuous metrics are measured repeatedly and
reported as mean ± half-range (C6 over 3 repetitions, C9 over 5 seeds) rather than as a
single number. Where a measurement does not match the paper — notably the
runtime-compression row of C6 — the script explains the discrepancy inline instead of
omitting it, and that row is reported without being gated.

The artifact also ships the results we recorded on an H100 under `data/reference/`, and
`01_gpu_core.sh` ends by diffing a fresh run against them with per-metric tolerances
(exact for structural counts, a relative band of 25 % or 40 % for ratios). This answers a different question
from the paper comparison: not "does this substantiate the paper?" but "did this machine
behave like the authors' machine?" — and because the honest answer on someone else's
machine is often "not exactly", a difference is reported as `DIFF` and summarised as a
`NOTE` rather than as a failure. Only quantities that a different machine has no business
changing are compared at all: the autotuner's benchmark-loop length is not one of them,
so trace-record counts are printed and not gated.

Both the console output and the log files are written on the assumption that a reviewer
may paste them to us, and that reviewers are anonymous to authors: no hostname is
recorded, paths under the artifact are rewritten to `<artifact>/…` and paths under a home
directory to `~/…`, and the environment block names the *kind* of environment rather than
its location.

**Complete.** The two required scripts cover claims C1, C2, C3, C4, C6, C8 and C9 as
live experiments, and an optional third builds the Review interface of §3.3 so the one
part of the paper that can only be judged visually is still available to look at.

Two parts of the paper are deliberately *not* covered, plus one comparison that cannot
be re-run at all. The reasons are stated in `README.md` and immediately below rather
than left implicit:

- **Recording overhead (§4.2)** is a hardware-specific measurement across 37
  TritonBench operators on an H100 *and* an MI300X. Running a handful of operators on a
  reviewer's GPU would produce a percentage that cannot substantiate the paper's, so we
  ship no such check rather than one that looks like evidence but is not.
- **§5.2 (MoE regression and LLVM bisection)** requires an NVIDIA B200, vLLM, and a
  bisection that rebuilds LLVM. It is out of reach for an artifact and is not attempted.
- **§5.1's version comparison** cannot be re-run at all: current TritonParse requires
  `triton > 3.3.1` and cannot instrument Triton 3.2. What *is* checkable — the numerical
  mechanism behind the accuracy change — is shipped as C9.

**Exercisable.** `run_all.sh` is the single entry point: it builds the environment and
runs every check in order, stopping at the first failure, in about 15 minutes.
Its steps also run standalone and each exits non-zero on failure. Both were validated from a fresh clone of
the published tag, under an environment built by `setup_env.sh` on a host with no usable
conda *and* under a plain virtualenv, so neither install path rests on the other. The basic test additionally
passes inside a stock `python:3.13-slim` container with no `git`, no `patch` and no
network at all.

**Evidence of verification and validation.** Beyond the artifact scripts, the project
carries a continuously-run test suite — 501 CPU tests, run by `00_kick_the_tires.sh`,
and 28 GPU tests, run by `01_gpu_core.sh` — which GitHub Actions also runs on every push
(`.github/workflows/test.yml`, CPU on `ubuntu-latest` and GPU on
`linux.g5.4xlarge.nvidia.gpu`).

## Why not Results Reproduced

The paper's measurements require hardware we cannot expect a reviewer to have:

- **§4.2 and §4.3** were measured across 37 TritonBench operators on **both** an
  NVIDIA H100 and an AMD Instinct MI300X.
- **§4.4** uses `gpt-oss-20b`, a gated multi-tens-of-GB download requiring MXFP4 kernels.
- **§5.2** requires an **NVIDIA B200**, plus vLLM, plus a bisection that rebuilds LLVM.
- **§5.1** requires **Triton 3.2**, which current TritonParse cannot instrument at all
  (`triton > 3.3.1` is required).

Rather than ask reviewers for that, the artifact reproduces each claim's *mechanism* at a
scale that runs on one ordinary GPU and reports its numbers against the paper's. Where
even the mechanism is out of reach — §4.2 and §5.2 — the artifact says so instead
of shipping a check that cannot support the claim.

## Licence

BSD-3-Clause, unchanged from the upstream project. See `LICENSE` at the repository root.
