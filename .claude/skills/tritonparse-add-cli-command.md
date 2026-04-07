---
name: tritonparse-add-cli-command
description: Guide for adding a new CLI subcommand to tritonparse. Use when creating a new subcommand like parse, reproduce, info, diff, or bisect.
---

# Adding a New CLI Subcommand

## Module Structure

Each subcommand is an independent module directory:

```
tritonparse/<command>/
├── __init__.py    # Public API exports
├── cli.py         # _add_<cmd>_args() + <cmd>_command()
└── ...            # Core logic
```

## Step-by-Step

### 1. Create the module

```
tritonparse/<new_cmd>/
├── __init__.py
├── cli.py
└── <core_logic>.py
```

### 2. Define CLI arguments in `<new_cmd>/cli.py`

Follow the pattern from `info/cli.py` (simplest reference):

```python
#  Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from typing import Optional
from tritonparse.shared_vars import is_fbcode


def _add_<cmd>_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the <cmd> subcommand."""
    parser.add_argument("input", help="Path to input file")
    # Add more arguments...


def <cmd>_command(
    input_path: str,
    skip_logger: bool = False,
    # ... other args
) -> None:
    """Main function for the <cmd> command."""
    if not skip_logger and is_fbcode():
        from tritonparse.fb.utils import usage_report_logger
        usage_report_logger()

    # Core logic here
```

### 3. Register in `tritonparse/cli.py`

Add three things:

**a. Import:**
```python
from .<new_cmd>.cli import _add_<cmd>_args, <cmd>_command
```

**b. Register subparser** (after existing subparsers):
```python
<cmd>_parser = subparsers.add_parser(
    "<cmd>",
    help="Short description",
)
_add_<cmd>_args(<cmd>_parser)
<cmd>_parser.set_defaults(func="<cmd>")
```

**c. Add dispatch** (in the `if/elif` chain in `main()`):
```python
elif args.func == "<cmd>":
    <cmd>_command(
        input_path=args.input,
        skip_logger=True,  # cli.main() already logged
    )
```

### 4. Export public API (if needed)

Add exports to `tritonparse/<new_cmd>/__init__.py` for programmatic usage.

### 5. Add tests

Create `tests/cpu/test_<cmd>.py` and add a Buck target:

```python
# In tests/cpu/BUCK
python_unittest(
    name = "test_<cmd>",
    srcs = ["test_<cmd>.py"],
    enable_lazy_imports = True,
    deps = [
        "//pytorch/tritonparse:tritonparse_lib",
        "//pytorch/tritonparse/tests:test_utils",
    ],
)
```

### 6. fbcode extensions (if needed)

If the subcommand needs fbcode-only arguments:
- Add them in `tritonparse/fb/utils.py` via `append_parser()`
- Use `if is_fbcode():` guards in the command function

## Reference Subcommands

| Subcommand | Complexity | Good reference for |
|------------|------------|-------------------|
| `info` | Simple | Basic pattern — 2 args, single function |
| `diff` | Medium | Multiple args, mode selection, output options |
| `reproduce` | Medium | Template system, fbcode replacer |
| `bisect` | Complex | Arg validation, sub-modes, resume support |

## Key Details

- **Usage logging**: `cli.main()` logs usage once at the top. Pass `skip_logger=True`
  to command functions to avoid double-logging.
- **Program name**: `tritonparse` in fbcode, `tritonparseoss` in OSS. Use `prog_name`
  variable in help text examples.
- **Error handling**: Use `parser.error()` for mutual exclusivity or validation.
  Raise `RuntimeError` for runtime failures.
