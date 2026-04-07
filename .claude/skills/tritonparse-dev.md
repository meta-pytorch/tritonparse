---
name: tritonparse-dev
description: Core development guide for tritonparse. Use when modifying tritonparse code, adding features, or fixing bugs.
---

# Tritonparse Development Guide

## Module Responsibilities

| Module | Entry Point | Purpose |
|--------|-------------|---------|
| `parse/` | `unified_parse()` in `parse/utils.py` | Parse structured trace logs, generate IR source mappings |
| `reproducer/` | `reproduce()` in `reproducer/orchestrator.py` | Generate standalone kernel reproducer scripts |
| `info/` | `info_command()` in `info/cli.py` | Query kernel information from trace files |
| `diff/` | `diff_command()` in `diff/cli.py` | Compare compilation events |
| `bisect/` | `bisect_command()` in `bisect/cli.py` | Bisect Triton/LLVM regressions |
| `structured_logging.py` | `init()` | Hook into Triton JIT/Launch to capture events |
| `context_manager.py` | `TritonParseManager` | Wrap init → run → parse in a `with` block |
| `ai/` | `LLMClient`, `ClaudeCodeClient` | LLM client abstraction |
| `validation/` | `json_validator.py` | JSON schema validation for trace events |
| `fb/` | `fb_run()` in `fb/utils.py` | Meta-internal extensions (not in OSS) |

## Core Parse Data Flow

```
cli.main()
  → unified_parse()                         # parse/utils.py
    → is_fbcode() ? fb_run() : oss_run()
      → Source() + RankConfig.from_cli_args()
      → copy_local_to_tmpdir()
      → parse_logs()                         # parse/common.py
        → _build_kernel_compile_mapping()    # torch trace attribution
        → parse_single_file()               # parse/trace_processor.py (per file)
          → parse_single_trace_content()     # per compilation event
          → _generate_ir_analysis()          # parse/ir_analysis.py
      → save_logs() + print_parsed_files_summary()
```

## Dual-Path Development Checklist

The codebase runs in two modes via `is_fbcode()` in `shared_vars.py`:
- **fbcode**: `unified_parse()` → `fb_run()` (in `tritonparse/fb/utils.py`)
- **OSS**: `unified_parse()` → `oss_run()` (in `tritonparse/parse/utils.py`)

When making changes, check every item:

1. **New parameter to parse pipeline** → Add to BOTH `oss_run()` and `fb_run()` signatures.
   `unified_parse()` forwards via `**kwargs`, so a missing param = `TypeError` at runtime.
   (Reference: D97557615 — `procedure_checks` missing from `oss_run()` broke 5 OSS CI tests)

2. **New CLI argument** → Add in `_add_parse_args()` (shared) or `append_parser()` (fbcode-only).
   Check if the argument needs to reach `parse_logs()` and forward it through `oss_run`/`fb_run`.

3. **New import** → Never import `tritonparse.fb.*` in the OSS code path. Use
   `if is_fbcode(): from tritonparse.fb.xxx import yyy` pattern.

4. **Forward to `parse_logs()`** → If `oss_run()` receives a new param that affects parsing,
   pass it to `parse_logs()` in `common.py`.

5. **Schema changes** → If modifying trace event fields, update JSON schemas in
   `validation/schemas/` (compilation.schema.json, launch.schema.json).

## Context Manager Pattern

```python
with TritonParseManager(
    enable_trace_launch=True,
    log_dir="/logs",
    split_inductor_compilations=True,
    out="/output",  # forwarded to unified_parse() via **parse_kwargs
) as tp:
    model(input)
# __exit__ calls unified_parse() automatically, then clear_logging_config()
```

- Extra `**parse_kwargs` are forwarded to `unified_parse()`
- `TEST_KEEP_OUTPUT=1` env var preserves temporary directories for debugging
- `keep_logs=True` or providing `log_dir=` prevents log cleanup

## Key Files

| File | What to know |
|------|-------------|
| `parse/utils.py` | `unified_parse()`, `oss_run()`, CLI arg definitions |
| `fb/utils.py` | `fb_run()`, `append_parser()`, MAST/fblearner source types |
| `parse/common.py` | `parse_logs()`, `RankConfig`, log file discovery, gzip |
| `parse/trace_processor.py` | `parse_single_file()`, two-pass processing, fake compilations |
| `parse/ir_analysis.py` | FileCheck-based IR analysis, loop scheduling |
| `shared_vars.py` | `is_fbcode()`, `DEFAULT_TRACE_FILE_PREFIX`, `TEST_KEEP_OUTPUT` |
| `structured_logging.py` | Triton JIT/Launch hooks, NDJSON event emission |
| `context_manager.py` | `TritonParseManager`, init/parse lifecycle |
| `cli.py` | CLI entry, subcommand dispatch, usage logging |

## Code Style

- License: `# Copyright (c) Meta Platforms, Inc. and affiliates.` (BSD-3, not confidential)
- Formatter: ufmt (ruff backend) + usort, line width 88
- Python >= 3.10
- Run `arc lint` before submitting
