---
name: tritonparse-oss-sync
description: OSS sync workflow for tritonparse. Use when syncing to GitHub, fixing GitHub CI, or managing the OSS/fbcode split.
---

# Tritonparse OSS Sync Guide

## Architecture

- **fbcode** is the source of truth
- **GitHub** (`meta-pytorch/tritonparse`) is the mirror
- `tritonparse/fb/` and `tests/fb/` are fbcode-only — they do NOT exist in OSS

## What's fbcode-only (not in OSS)

| Item | Purpose |
|------|---------|
| `tritonparse/fb/utils.py` | `fb_run()`, `append_parser()`, `usage_report_logger()`, MAST/fblearner source types |
| `tritonparse/fb/source_type.py` | FB-specific source types (MAST, fblearner) |
| `tritonparse/fb/reproducer/` | `FBCodePlaceholderReplacer` for fbcode reproducer templates |
| `tests/fb/` | `test_mast_compat` |
| `BUCK`, `PACKAGE` | Buck build files |
| fbcode-only CLI args | Added via `append_parser()` (MAST job ID, fblearner URL, etc.) |

## OSS Entry Point and Dependencies

From `pyproject.toml`:
- CLI entry: `tritonparseoss = "tritonparse.cli:main"` (fbcode uses `tritonparse`)
- Runtime dependencies: `orjson>=3.9`, `rich>=13.0`
- Optional: `triton>3.3.1` or `pytorch-triton>=3.4.0`
- Dev tools: `ufmt==2.9.0`, `usort==1.1.0`, `ruff-api==0.2.0`, `ruff>=0.4.0`, `coverage>=7.0.0`
- Python >= 3.10, version via `setuptools-scm`

## GitHub CI

Workflow: `.github/workflows/test.yml`

Three jobs:
1. **`format-check`** — `make format-check` (ufmt + ruff) on `ubuntu-latest`
2. **`test-from-source`** — Compile Triton from source (~30-50 min), run all tests
3. **`test-from-pip`** — pip install Triton, run all tests

Both test jobs run on `4-core-ubuntu-gpu-t4`, Python 3.13, CUDA 12.8.

Triggers: push to `main`/`develop`, PRs to `main` (ignores `website/`, `docs/`, `*.md`).

### CI Scripts (`.ci/`)

| Script | Purpose |
|--------|---------|
| `setup.sh` | Install system deps, CUDA 12.8, Miniconda, cuDNN, PyTorch nightly |
| `install-triton.sh` | Clone and compile Triton from source (with caching) |
| `install-project.sh` | `pip install -e ".[test]"` |
| `run-tests.sh` | Run tests by `TEST_TYPE`: `cpu`, `cuda`/`gpu`, or `all` (default) |

Test runner uses `python -m unittest discover` (not pytest), scanning `tests/` recursively.

## Common Issues

### OSS CI fails but fbcode passes
Most likely a dual-path bug: a parameter or feature works in `fb_run()` but is
missing from `oss_run()`. See the tritonparse-dev skill for the dual-path checklist.
Reference: D97557615 (`procedure_checks` missing from `oss_run`).

### Format check fails
```bash
# Local fix (OSS)
make format

# fbcode
arc lint -a
```

### Triton API compatibility
OSS CI builds Triton from `main` branch daily. If Triton changes its JIT/Launch
hook API, `structured_logging.py` may break. Guard version-specific features with
try/except or version checks.

### Import errors in OSS
Never import `tritonparse.fb.*` unconditionally. Always use:
```python
if is_fbcode():
    from tritonparse.fb.xxx import yyy
```
