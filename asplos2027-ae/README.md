# TritonParse — ASPLOS'27 Artifact

Artifact for **“TritonParse: Multi-IR Provenance and Reproducible Debugging for Triton
Kernel Compilation”** (ASPLOS'27, submission #461).

Badges claimed: **Artifacts Available** and **Artifacts Evaluated — Functional**.
[`STATUS.md`](STATUS.md) maps those to the ACM criteria and says why we do not claim
Results Reproduced.

**Contents** — [Quick start](#quick-start) · [Requirements](#requirements) ·
[What this covers](#what-this-artifact-covers) ·
[What it does not](#what-this-artifact-does-not-cover) ·
[Installation](#installation) ·
[Version delta vs the paper](#version-delta-versus-the-paper) ·
[Read this before filing anything](#read-this-before-filing-anything) ·
[The Review stage](#the-review-stage-c3) · [Layout](#layout) ·
[Troubleshooting](#troubleshooting)

---

## Getting the artifact

```bash
git clone --branch asplos2027-ae https://github.com/meta-pytorch/tritonparse
cd tritonparse
```

Clone; do not download a zip. TritonParse derives its version with `setuptools-scm`,
which needs git metadata that a zip has no way to carry — see *Installing TritonParse
itself* below for what that failure looks like.

For the duration of the artifact evaluation we track the branch `asplos2027-ae` rather
than a tag, so that anything you report and we fix reaches you with a `git pull`. The
branch is only ever fast-forwarded, never rewritten, so a pull always applies. Every run
prints the branch and commit it ran, and records them in `summary.txt` — quoting that
line in a report tells us exactly what you ran.

## Quick start

```bash
./asplos2027-ae/run_all.sh
```

That is the entire evaluation: **about 15 minutes** for the required steps, plus a
minute for the optional interface build. One command, nothing to install first. It builds the environment — downloading and installing conda itself if the host
has none — then runs every check in order, stopping at the first failure. One ordinary
NVIDIA GPU (SM80+, ≥ 16 GB) is enough; no H100, MI300X or B200 is required.

It runs these four steps, each of which is also a script you can run on its own:

| # | Step | Allow up to |
|---|------|-------------|
| 1 | `setup_env.sh` — conda if needed, then the environment from `environment.yml` (~5 GB) | 5 min |
| 2 | `00_kick_the_tires.sh` — the basic test | 2 min |
| 3 | `01_gpu_core.sh` — the main check | 8 min |
| 4 | `02_review_ui.sh` — the Review interface build (optional) | 1 min |

Those are budgets, not measurements: on an H100 with warm caches the three checks take
about 25 s, 3 min and 9 s. Every step prints a PASS/FAIL table and exits non-zero on
failure, and `run_all.sh` ends with one summary table over all of them.

**Every run keeps its own output, on disk.** Each run writes to
`asplos2027-ae/logs/<timestamp>/` — one log per step, the summary, the CSVs the main
check produces, and `environment.txt` with the full package list — with `logs/latest`
pointing at the most recent. Re-running never overwrites an earlier attempt, so a failure
remains available to look at. Running one check on its own gets the same treatment:
`./01_gpu_core.sh` prints its log path on the first line and writes there too, so you
never have to copy a terminal to report something. Set `AE_NO_LOG_FILE=1` for console
only.

Each check opens with the environment it ran in — date, OS, tree, python, conda version,
tritonparse/triton/torch versions, GPU and driver — so every log identifies its own
stack. Quoting that block, or attaching `environment.txt`, is the most useful thing you
can put in a report.

**The files are written to be safe to send us.** You are anonymous to us, and a log full
of `/home/<your name>/` would not leave you that way, so what lands on disk carries no
hostname and no home directory: paths under the artifact appear as `<artifact>/…` and
anything under your home as `~/…`. The environment block reports which *kind* of
environment ran — the artifact's own conda environment, or one of yours — never where it
lives. Nothing is collected or sent anywhere; the files are yours, and this only means
you can attach one without reading it line by line first.

Useful flags: `--skip-setup` to use an environment you have already activated,
`--skip-ui` to skip the one step that downloads npm packages, `--keep-going` to run
everything rather than stop at the first failure, and `--quiet` to keep step output in
the logs only.

⚠️ **Step 1 replaces an existing `tritonparse-ae` environment rather than reusing it.**
A rebuild costs a few minutes; a silently drifted environment costs far more, because it
fails inside the checks and looks like the artifact's fault. Pass
`AE_KEEP_EXISTING_ENV=1` to keep what you have.

**`00_kick_the_tires.sh` is the “basic test” for the kick-the-tires phase.** It is the
fast one and its checks are offline, so it isolates installation problems before anything
touches the GPU. It was validated end-to-end inside a stock `python:3.13-slim` container
with no `git`, no `patch`, no compiler and no network — that run used the virtualenv path
below, since the container has no conda.

---

## Requirements

### Hardware

**One NVIDIA GPU**, compute capability 8.0 or later, with at least 16 GB of memory — an
A100, L4, L40S, RTX 30xx/40xx/50xx or H100 all work. The hardware the paper was measured
on (an H100 *and* an MI300X, plus a B200 for §5.2) is **not** required, and the badges we
claim do not depend on it.

One detail is worth knowing before you read C9's output. The PTX in the paper's Listing 2
is not uniformly portable: the Triton 3.3 sequence uses `mul.bf16`, which `ptxas` accepts
only from `sm_90`, while the Triton 3.2 sequence is all `fma.rn.bf16` and assembles from
`sm_80`. That asymmetry *is* the case study — Triton 3.2 emulated the multiply because
pre-Hopper hardware had no instruction for it. So on Ampere or Ada, C9 executes the 3.2
sequence as inline PTX and substitutes a semantic model for the 3.3 side, says so in its
output, and records `native_mul_bf16=false` in its CSV row. The check still passes and the
divergence rate is unchanged; on `sm_90` the model and the real PTX agree on every sample,
which is what makes the substitution sound rather than convenient. You can see that path
on any GPU with `--assume-no-native-mul`.

| Script | Wall time | What it validates |
|--------|-----------|-------------------|
| `00_kick_the_tires.sh` | **~2 min** | Reconstruct (C2), structured query, cross-compilation IR diff (C3), reproducer *generation* (C4), the project's 501-test CPU suite |
| `01_gpu_core.sh` | **~8 min** | Multi-IR capture (C1), reproducer *execution* (C4), log-size reduction (C6), the three bug classes of §4.4 (C8), the BF16 rounding mechanism of §5.1 (C9), the 28-test GPU suite, and a diff against our recorded results |
| `02_review_ui.sh` *(optional)* | **~1 min** + npm download | Builds the Review interface of §3.3 so you can inspect a trace visually |

These are upper bounds, chosen so that a slow machine with a cold Triton compilation
cache still finishes inside them; an H100 with warm caches takes about 25 s, 3 min and
9 s respectively. Cache warmth and GPU speed are what move them, so a run near the budget
is normal rather than a symptom. Note also that 60 s of the main check is not work but
two deliberate 30-second hang timeouts in the C8 check, which are there by design.
Building the environment is separate and took 78 s with a warm pip cache, 236 s
without.

### Software

Only two things must come from the host:

1. **A shell and `curl`.** `setup_env.sh` installs conda if the host has none, and
   `environment.yml` supplies the interpreter, so no particular system Python or
   pre-existing conda is needed. An existing conda 23+ is reused (validated on 25.11.1
   and on the Miniforge 26.3.2 the script installs).
2. **An NVIDIA kernel driver.** No CUDA *toolkit* is required: the PyTorch wheels bundle
   the entire CUDA userspace (`nvidia-cublas`, `nvidia-cudnn`, `nvidia-cusolver`,
   `nvidia-cusparse`, `nvidia-nccl`, …) as ordinary pip dependencies.

Nothing else. In particular `git`, `patch` and a compiler are all unnecessary: the basic
test was validated end-to-end inside a stock `python:3.13-slim` container (Debian 13)
that has none of them, installing from a local wheel directory with no network access.
The C8 check prefers `patch(1)` when present and otherwise falls back to a pure-Python
applier that produces byte-identical results.

Node.js comes from `environment.yml` too (v24.19.0 / npm 11.17.0 as resolved by
conda-forge), so the optional `02_review_ui.sh` needs nothing extra from the host.

If you would rather not use conda, everything except Node also installs with
`pip install -r requirements-pinned.txt` into a plain `venv` on Python ≥ 3.11 — that is
how the artifact was originally validated, and both required scripts pass that way. Only
the optional interface build then needs a Node you supply yourself.

Validated host: Linux 6.16 x86-64, NVIDIA driver **580.126.09**, NVIDIA H100.

### Disk space

| | |
|---|---|
| Unpacked artifact (whole repository) | **~5 MB** |
| conda environment | **~5.1 GB** (torch 1.1 GB, triton 691 MB, `nvidia-*` 2.7 GB, Node ~120 MB) |
| Scratch output, kept for inspection | < 500 MB |

---

## What this artifact covers

TritonParse implements a four-stage workflow — **Record, Reconstruct, Review,
Reproduce**. Each row names a paper claim, the script that exercises it, and what you
should see.

| ID | Paper claim | Where | Exercised by | Expected result |
|----|-------------|-------|--------------|-----------------|
| **C1** | A single capture yields TTIR → TTGIR → LLIR → PTX/AMDGCN plus launch metadata | §3.1 | `01_gpu_core.sh` step 1 | every compilation carries all 4 stages; per-stage line counts printed |
| **C2** | Reconstruct turns raw logs into a compact, IR-mapped archive | §3.2 | `00_kick_the_tires.sh` steps 1–2 | archive + `log_file_list.json`; kernel/launch inventory |
| **C3** | Review compares compilations across IR levels (the mechanism behind both case studies) | §3.3, §5.1, §5.2 | `00_kick_the_tires.sh` step 3 (data layer), `02_review_ui.sh` (interface) | per-stage IR diff (TTIR/TTGIR/LLIR/PTX) and per-Python-line attribution; the interface builds to one self-contained HTML file you open on the shipped trace |
| **C4** | A generated reproducer is minimal and runs standalone | §3.4 | `00_kick_the_tires.sh` step 4 (generate), `01_gpu_core.sh` step 3 (execute) | a single file of roughly 750–780 lines, no import of the original workload, exits 0 from a clean directory |
| **C6** | Two-stage log-size reduction, 57.4× on NVIDIA | §4.3, Fig. 3 | `01_gpu_core.sh` step 2 | reconstruct ratio in the same order of magnitude (we measure 74–77× on an H100) |
| **C8** | assert / IMA / hang all reproduce (15/15 mutations) | §4.4, Listing 1 | `01_gpu_core.sh` step 4 | 3/3 bug classes reproduced at miniature scale |
| **C9** | RoPE accuracy change comes from native BF16 multiply replacing FMA emulation | §5.1, Listing 2 | `01_gpu_core.sh` step 5 | measured divergence rate in the same range as the paper's 25.84 %. On `sm_90`+ both sequences run as inline PTX; below it the `mul.bf16` side falls back to a semantic model (see [Hardware](#hardware)) |
| — | Verification & validation | — | `00_kick_the_tires.sh` (CPU suite), `01_gpu_core.sh` (GPU suite) | 501 CPU tests, 28 GPU tests |
| — | Agreement with the authors' machine | — | `01_gpu_core.sh` step 7 | reported, never gated: differences are marked `DIFF` and summarised as a `NOTE`, which is not a failure |

## What this artifact does *not* cover

Stated up front rather than left for you to discover. Fuller reasoning in
[`STATUS.md`](STATUS.md).

| Paper claim | Why not |
|---|---|
| Recording overhead (§4.2) | A hardware-specific measurement over 37 TritonBench operators on an H100 *and* an MI300X. A scaled-down run on one GPU yields a number that cannot substantiate the paper's, so we ship no check rather than a misleading one. |
| MoE regression + LLVM bisection (§5.2) | Needs an NVIDIA B200, vLLM, and a bisection that rebuilds LLVM. Not shippable; not attempted. |
| The Triton 3.2 vs 3.3 comparison itself (§5.1) | Current TritonParse requires `triton > 3.3.1` and cannot instrument Triton 3.2 at all. The *numerical mechanism* behind the case study is checkable and is covered as C9. |

---

## Installation

### Environment manager

`setup_env.sh` is the supported path and assumes only a shell, `curl` and an NVIDIA
driver:

```bash
./setup_env.sh
```

It does three things. It looks for a usable conda — one it installed earlier, then one on
your `PATH`, rejecting anything older than conda 23, since a distribution-packaged
`/usr/bin/conda` is often years old and cannot solve current conda-forge repodata. If it
finds none it downloads Miniforge (pinned to 26.3.2-3, SHA-256 verified) and installs it
under `asplos2027-ae/.conda`. Then it builds the environment from `environment.yml` and
prints the activation command for whichever case applied.

It is deliberately unintrusive: no `conda init`, no edit to any shell startup file,
nothing outside the artifact directory. Delete `asplos2027-ae/.conda` and the conda it
installed is gone. Re-running it reuses what exists rather than rebuilding.

Three variables adjust it: `AE_CONDA_HOME` (where to install conda), `AE_ENV_NAME` (what
to call the environment), and `AE_CONDA_DISTRO=miniconda` to install Miniconda instead of
Miniforge.

**Why Miniforge and not Miniconda.** Both give you the same `conda`. They differ in which
channels they come configured for, and here that matters: Miniconda enables Anaconda's
`defaults` channels, and a recent conda then refuses to solve until their commercial Terms
of Service have been accepted — even though `environment.yml` asks only for conda-forge,
because `defaults` is conda's *built-in* channel list rather than something the manifest
controls. Miniforge is configured for conda-forge already, so nothing needs accepting.
Accepting a licence is not something a setup script should do quietly on your behalf.

Doing it by hand is two commands if you prefer:

```bash
conda env create -f environment.yml
conda activate tritonparse-ae
```

`environment.yml` takes only the interpreter, `pip` and Node from conda-forge and defers
the scientific pins to `requirements-pinned.txt`, which stays the single source of truth
for versions. Conda has no build of `torch 2.13.0+cu130`, so duplicating those pins in
the manifest would create a second list that could drift from the first.

**If your own conda stops with a Terms-of-Service error**, that is the gate described
above:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following
channels: https://repo.anaconda.com/pkgs/main, https://repo.anaconda.com/pkgs/r
```

Neither `nodefaults` in the manifest nor `CONDA_CHANNELS` suppresses it (we tried both).
Either let `setup_env.sh` install Miniforge into a fresh `AE_CONDA_HOME`, or accept the
two channels once:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

`AE_ACCEPT_ANACONDA_TOS=1 ./setup_env.sh` runs those two commands for you, if you would
rather state your consent that way.

**Prefer not to accept Anaconda's terms, or prefer not to use conda at all?** Then use a
plain virtualenv; it needs Python ≥ 3.11 on the host and gives you everything but Node:

```bash
python3 -m venv .venv-tritonparse-ae
source .venv-tritonparse-ae/bin/activate
pip install --upgrade pip
pip install -r requirements-pinned.txt
```

Both required scripts pass under either environment; we ran them under both. Only
`02_review_ui.sh` notices the difference, because it is the one that wants Node.

### Installing TritonParse itself

⚠️ **Do not run a bare `pip install .` from an unpacked zip.** TritonParse derives its
version with `setuptools-scm`, which needs git metadata. A Zenodo archive or a GitHub
"Download ZIP" has no `.git`, so the build fails with:

```
LookupError: setuptools-scm was unable to detect version for <dir>.
Make sure you're either building from a fully intact git repository or PyPI tarballs.
```

Use one of these instead:

```bash
# (a) from this snapshot — keeps the library in sync with the bundled tests/
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TRITONPARSE=0.5.1 pip install <repo-root>

# (b) from PyPI — simplest, no build step
pip install tritonparse==0.5.1
```

`requirements-pinned.txt` uses (b). `00_kick_the_tires.sh` also runs the repository's
`tests/cpu` suite; if you want those tests to match the installed library exactly, use
(a) against the snapshot you are reviewing.

### Choosing the right CUDA wheel

`pip install torch==2.13.0` from PyPI gives the **`+cu130`** build, which requires a
recent driver. Check yours first:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

If the driver is too old for CUDA 13, install a lower-CUDA build of the *same* torch
version from PyTorch's index:

| CUDA build | torch 2.13.0 available? | Install |
|---|---|---|
| cu130 (PyPI default) | ✅ | `pip install torch==2.13.0` |
| cu129 | ✅ | `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129` |
| cu126 | ✅ | `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126` |
| **cu128** | ❌ **no 2.13.0 build exists** | — |

Any of these works; only the driver floor differs.

---

## Version delta versus the paper

**The artifact does not pin the paper's environment, and this is deliberate.**

| | Paper (Table 1) | Artifact |
|---|---|---|
| PyTorch | 2.9.0 | **2.13.0** |
| Triton | 3.4.0 | **3.7.1** |
| CUDA | 12.8 (driver 550.90.07) | **13.0** (driver 580.126.09) |
| ROCm | 6.4.0 (driver 6.7.3) | not exercised |
| GPUs | NVIDIA H100 98 GB, AMD Instinct MI300X 192 GB | any single SM80+ NVIDIA GPU |
| OS | CentOS Stream 9 | any recent x86-64 Linux |

TritonParse has evolved substantially since submission, and the paper's stack no longer
installs cleanly against current drivers. Pinning released, currently-installable
versions makes the artifact *more* likely to run on a reviewer's machine, not less.

**Consequence:** absolute numbers produced by these scripts differ from the paper. The
scripts therefore report both their own measurement and the paper's, and judge on
*mechanism and order of magnitude* rather than exact equality. For example,
`01_gpu_core.sh` reports the reconstruct compression ratio next to the paper's 57.4×
(Fig. 3); we measure 74–77× on an H100.

---

## Read this before filing anything

**1. Absolute numbers differ from the paper by design.** See the version delta above.
Every script prints its own measurement next to the paper's and judges on mechanism and
order of magnitude.

**2. `pip install .` from an unpacked zip fails.** `setuptools-scm` needs git metadata
that a Zenodo/GitHub zip does not carry. See *Installing TritonParse itself* above.

**3. C6's middle row does not match the paper, and we say why.** The paper reports
runtime compression at 21.9 % of raw; we measure ~50 % on our microbenchmark.
TritonParse's gzip mode emits one gzip member per record and so cannot exploit redundancy
*between* records — whole-file gzip of the same log is 0.84 %. The paper's figure comes
from a mix of 37 real operators. Only the reconstruct ratio, the paper's headline claim,
is used as a pass criterion; the runtime row is printed as informational with this
explanation inline.

**4. C8 matches fault *classes*, not exact CUDA error codes.** The IMA mutation has been
observed reporting both `cudaErrorIllegalAddress` and `cudaErrorInvalidAddressSpace`
across runs, because the faulting address depends on where the allocator places tensors.
Both codes are printed so the variance is visible.

**5. Generated reproducers are not `torch`+`triton`-only.** They also import `numpy` and
`tritonparse.backend`. Both are in `requirements-pinned.txt`, but the dependency surface
is slightly wider than the paper's prose suggests.

**6. Step 7 marks differences from our machine; a `DIFF` is not a failure.** The last
step of the main check diffs your CSVs against `data/reference/`, which we recorded on an
H100. It answers "did this machine behave like the authors'?", a different question from
"does this substantiate the paper?" — so a metric outside tolerance prints as `DIFF`, is
summarised as one `NOTE`, and leaves the run passing. The per-claim checks above it are
the pass criteria, and they print `FAIL` and mean it.

Trace-record counts in particular are *not* compared. Triton's autotuner sizes its
benchmark loop from the measured latency of the kernel, so the number of launches — and
therefore the number of records C6 sees — moves with the clock, the driver and whatever
else is on the GPU. We print the count, because it explains the ratio, and gate only the
ratio.

---

## The Review stage (C3)

Paper §3.3 describes an interactive interface, so half of that claim is inherently
visual. The artifact splits it:

- **The data layer is checked automatically.** `00_kick_the_tires.sh` step 3 runs
  `tritonparseoss diff`, which produces the per-IR-stage comparison and the
  per-Python-line attribution that the interface renders — and that both case studies
  in §5.1 and §5.2 rely on.
- **The interface you build and look at.** `./asplos2027-ae/02_review_ui.sh` builds it
  from `website/` into a single self-contained `standalone.html` (~1 min once npm has a
  cache; the first run also downloads ~300 MB of npm packages), verifies the build is
  usable, and tells you which trace to load and what to look for. Node.js is the only
  dependency in the whole artifact that this needs; `environment.yml` supplies it.

The same interface is hosted at **<https://meta-pytorch.org/tritonparse/>** if you would
rather not build it. Either way, all processing happens in your browser and the trace is
read locally — nothing is uploaded.

---

## Layout

```
asplos2027-ae/
├── README.md                 you are here: quick start, requirements, claim map
├── STATUS.md                 badges claimed, mapped to the ACM criteria
├── run_all.sh                THE ENTRY POINT: environment + every check, ~15 min
├── setup_env.sh              installs conda if needed, then builds the environment
├── environment.yml           conda manifest: interpreter, Node, and the pins below
├── requirements-pinned.txt   exact versions, validated 2026-08-06
│
├── logs/<timestamp>/         one directory per run_all.sh run (gitignored)
├── 00_kick_the_tires.sh      the basic test, up to ~2 min
├── 01_gpu_core.sh            the main check, up to ~8 min
├── 02_review_ui.sh           optional: build the Review interface
│
├── claims/
│   ├── C8_bugs/              assert / IMA / hang, as literal patches (paper Listing 1)
│   └── C9_rope/              BF16 rounding study + PTX excerpts from paper Listing 2
├── scripts/
│   ├── ae_kernels.py              shared kernels and workloads
│   ├── ae_platform.py             platform string with vendor build tags removed
│   ├── check_ir_capture.py        C1
│   ├── measure_storage.py         C6
│   ├── compare_to_reference.py    diffs your run against ours
│   ├── check_docs_consistency.py  keeps the numbers in these documents honest
│   └── make_reference_trace.py    re-records the reference trace, if you want your own
├── data/reference/           the numbers WE measured, for automatic comparison
├── traces/reference/         the trace the offline checks run against (~420 KB)
└── results/                  CSVs written by the main check
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `tritonparseoss: command not found` | The environment does not exist yet — run `./run_all.sh`. Each script activates the environment itself, so you should not see this once `setup_env.sh` has succeeded. |
| `CondaToSNonInteractiveError` during `conda env create` | Anaconda's Terms of Service for the built-in `defaults` channels. Accept them once; see [Environment manager](#environment-manager). |
| `no CUDA device visible` but `nvidia-smi` shows one | Check `CUDA_VISIBLE_DEVICES`; it may point at an index that does not exist. |
| torch imports but `torch.cuda.is_available()` is `False` | The wheel's CUDA build is newer than your driver. Install a `cu126`/`cu129` build — see *Choosing the right CUDA wheel*. |
| `setuptools-scm was unable to detect version` | You ran `pip install .` from a zip. See note 2 above. |
| `ModuleNotFoundError: No module named 'parameterized'` | Test-only dependency; it is in `requirements-pinned.txt`. |
| `reference trace not found` | It ships under `traces/reference/`. If you deleted it, `scripts/make_reference_trace.py` records an equivalent one on your GPU. |
| A script fails | Every script prints the path of its scratch directory, which is kept on exit and holds the full log of each step. |

## Contact

Issues and questions: <https://github.com/meta-pytorch/tritonparse/issues>.
During artifact evaluation, please route questions through the AE chairs / HotCRP.
