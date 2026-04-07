---
description: Rules for Triton JIT/Launch hook compatibility in structured logging
globs: tritonparse/structured_logging.py
---

# Structured Logging Rules

`structured_logging.py` hooks into Triton's `JITHook` and `LaunchHook` APIs to
capture compilation and launch events as NDJSON trace logs.

## Compatibility

- Must support Triton >= 3.4.0. Guard version-specific features appropriately.
- Hook registration uses `triton.knobs` API: `knobs.runtime.jit_post_compile_hook`,
  `knobs.runtime.launch_enter_hook`, and `knobs.compilation.listener`.
  These APIs may change across Triton versions.

## Log format stability

- Output format is NDJSON (one JSON object per line), consumed by all downstream
  parsing code and the web viewer.
- Field additions are safe; field removals or renames are breaking changes.
- When modifying the event schema, update the corresponding JSON schemas in
  `tritonparse/validation/schemas/` (compilation.schema.json, launch.schema.json).

## Environment variables

- `TRITON_TRACE` — controls where trace logs are written
- `TRITONPARSE_DEBUG` — enables verbose debug logging via `tp_logger.py`
