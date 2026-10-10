# Artifact evaluation outcome

For **“TritonParse: Multi-IR Provenance and Reproducible Debugging for Triton Kernel
Compilation”**, ASPLOS'27 submission #461.

Artifact evaluation is complete. The final awards are:

| Badge | Outcome |
|---|---|
| **Artifacts Available** | Awarded |
| **Artifacts Evaluated — Functional** | Awarded |
| Results Reproduced | Not requested; not awarded |

The supplied workloads demonstrate core functionality at a smaller scale than the
paper's experiments. The coverage and limits below describe what the artifact checks.

---

## Artifacts Available

TritonParse is developed in the open at
<https://github.com/meta-pytorch/tritonparse> under a BSD-3-Clause licence, and is
published on PyPI as `tritonparse`.

The [`asplos2027-ae`](https://github.com/meta-pytorch/tritonparse/tree/asplos2027-ae)
branch contains the artifact. Use the full commit cited in the paper or identified in
the Zenodo record to select a fixed version.

- **Reserved archival DOI:** `10.5281/zenodo.21853863`. The DOI becomes publicly
  accessible when the deposit is published.
- **Archive contents:** the complete source tree at a fixed commit, including
  `asplos2027-ae/`, its reference trace and measurements, `tests/`, `tritonparse/`,
  and `website/`. Dependencies are installed as described in the README.
- **Version identification:** record the full Git commit SHA in the Zenodo metadata
  and preserve it in the archive filename. An unpacked ZIP/tar archive is supported by
  the default installation; Git metadata is not required.

Updates after evaluation correct the optional UI packaging and the archival
instructions. The computational checks, acceptance criteria, and reference data are
unchanged.

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

Runs record their software versions, GPU, driver, and source identifier. The logging
wrapper rewrites repository paths to `<artifact>/…` and home-directory paths to `~/…`.
Logs and measurements remain local; the scripts do not upload them.

**Complete.** The two required scripts cover claims C1, C2, C3, C4, C6, C8 and C9 as
live experiments, and an optional third builds the Review interface of §3.3 so the one
part of the paper that can only be judged visually is still available to look at.

Two parts of the paper are deliberately *not* covered, plus one comparison that cannot
be re-run at all. The reasons are stated in `README.md` and immediately below rather
than left implicit:

- **Recording overhead (§4.2)** is a hardware-specific measurement across 37
  TritonBench operators on an H100 *and* an MI300X. Running a handful of operators on a
  different GPU would produce a percentage that cannot substantiate the paper's, so we
  ship no such check rather than one that looks like evidence but is not.
- **§5.2 (MoE regression and LLVM bisection)** requires an NVIDIA B200, vLLM, and a
  bisection that rebuilds LLVM. It is out of reach for an artifact and is not attempted.
- **§5.1's version comparison** cannot be re-run at all: current TritonParse requires
  `triton > 3.3.1` and cannot instrument Triton 3.2. What *is* checkable — the numerical
  mechanism behind the accuracy change — is shipped as C9.

**Exercisable.** `run_all.sh` is the single entry point: it builds the environment and
runs every check in order, stopping at the first failure, in about 15 minutes.
Its steps also run standalone and each exits non-zero on failure. The two required
scripts were validated under an environment built by `setup_env.sh` and under a plain
virtualenv. The basic test was also validated inside a stock `python:3.13-slim`
container with no `git`, no `patch`, and no network access.

**Evidence of verification and validation.** Beyond the artifact scripts, the project
carries a continuously-run test suite — 501 CPU tests, run by `00_kick_the_tires.sh`,
and 28 GPU tests, run by `01_gpu_core.sh` — which GitHub Actions also runs on every push
(`.github/workflows/test.yml`, CPU on `ubuntu-latest` and GPU on
`linux.g5.4xlarge.nvidia.gpu`).

## Results Reproduced: scope of the submission

Only Artifacts Available and Artifacts Evaluated — Functional were requested and
awarded. Results Reproduced was not requested. The artifact checks functionality and
numerical mechanisms; it does not reproduce all of the paper's experiments, which
have the following additional requirements:

- **§4.2 and §4.3** were measured across 37 TritonBench operators on **both** an
  NVIDIA H100 and an AMD Instinct MI300X.
- **§4.4** uses `gpt-oss-20b`, a gated multi-tens-of-GB download requiring MXFP4 kernels.
- **§5.2** requires an **NVIDIA B200**, plus vLLM, plus a bisection that rebuilds LLVM.
- **§5.1** requires **Triton 3.2**, which current TritonParse cannot instrument at all
  (`triton > 3.3.1` is required).

The supplied workloads exercise the supported mechanisms on one NVIDIA GPU and report
their measurements alongside the paper's. Recording overhead (§4.2) and the MoE
regression and LLVM bisection (§5.2) are outside the supplied checks, as documented in
the README and paper appendix.

## Licence

BSD-3-Clause, unchanged from the upstream project. See `LICENSE` at the repository root.
