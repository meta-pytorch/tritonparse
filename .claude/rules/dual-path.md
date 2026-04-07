---
description: Ensure OSS/fbcode dual-path parameter consistency in parse utils
globs: tritonparse/parse/utils.py
---

# Dual-Path Consistency Rules

`unified_parse()` dispatches to `oss_run()` (OSS) or `fb_run()` (fbcode) and
forwards parameters via `**kwargs`.

## When adding a new parameter to the parse pipeline:

1. Add it to `unified_parse()` signature and docstring
2. Add it to `oss_run()` signature and docstring
3. Add it to `fb_run()` in `tritonparse/fb/utils.py`
4. Forward it from `oss_run()` to `parse_logs()` if applicable
5. Verify both paths work — fbcode tests won't catch OSS-only breakage

## Why this matters

A parameter present in `unified_parse()` but missing from `oss_run()` causes a
`TypeError` in OSS environments (GitHub CI). This is invisible in fbcode because
fbcode uses `fb_run()` instead. See D97557615 for a real instance of this bug.
