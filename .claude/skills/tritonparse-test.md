---
name: tritonparse-test
description: Testing guide for tritonparse. Use when running tests, adding tests, or debugging CI failures.
---

# Tritonparse Testing Guide

## Running Tests (Buck / fbcode)

### CPU Tests (no GPU required)

```bash
# All CPU tests
buck test //pytorch/tritonparse/tests/cpu:

# Individual test
buck test //pytorch/tritonparse/tests/cpu:test_ir_parser
```

Available CPU test targets:
- `test_ir_parser` — IR parsing and source mapping extraction
- `test_kernel_query` — Kernel info lookup and fuzzy matching
- `test_reproducer` — Reproducer generation
- `test_info_cli` — Info subcommand CLI
- `test_cli_integration` — CLI integration
- `test_function_extractor` — Kernel function extraction from source
- `test_placeholder_replacer` — Template placeholder replacement
- `test_torch_trace_parser` — Torch trace log parsing
- `test_kernel_attribution` — Kernel compilation attribution
- `test_pipeline_integration` — End-to-end pipeline
- `test_schema_validation` — JSON schema validation
- `test_fake_compilation` — Fake compilation event synthesis
- `test_bisect_executor` — Bisect shell executor
- `test_bisect_logger` — Bisect logger
- `test_bisect_state` — Bisect state management
- `test_bisect_commit_detector` — Bisect commit detection
- `test_bisect_pair_tester` — Bisect pair testing
- `test_env_manager` — Environment manager

### Diff Tests

```bash
buck test //pytorch/tritonparse/tests/cpu/diff:
```

Available diff test targets:
- `test_diff_engine` — Core diff engine logic
- `test_cli` — Diff CLI interface
- `test_tensor_value` — Tensor value analysis
- `test_kernel_matcher` — Kernel matching logic
- `test_trace_diff` — Trace-level diffing
- `test_trace_output` — Trace output formatting

### GPU Tests (remote GPU execution)

```bash
# All GPU tests
buck test //pytorch/tritonparse/tests/gpu:

# Individual test
buck test //pytorch/tritonparse/tests/gpu:test_structured_logging
```

Available GPU test targets:
- `test_structured_logging` — NDJSON event capture and SASS extraction
- `test_context_manager` — TritonParseManager lifecycle
- `test_complex_kernels` — Multi-kernel scenarios
- `test_tensor_blob` — Tensor blob storage
- `test_autotune` — Autotuning event capture
- `test_launch_tracing` — Launch event tracing
- `test_tensor_descriptor` — TensorDescriptor reconstruction

### AI Module Tests

```bash
buck test //pytorch/tritonparse/tests/ai:test_client
buck test //pytorch/tritonparse/tests/ai:test_parsers
```

### FB-Internal Tests

```bash
buck test //pytorch/tritonparse/tests/fb:test_mast_compat
```

## Running Tests (pytest / OSS)

Used by GitHub CI and local OSS development:

```bash
pytest tests/ -v -m "not cuda"   # CPU only
pytest tests/ -v                  # All tests (requires GPU)
```

## CPU vs GPU: When to Use Which

- **CPU tests** (`python_unittest`): Test parsing logic, CLI, data structures, schema
  validation — anything that doesn't need actual Triton kernel execution.
- **GPU tests** (`python_unittest_remote_gpu`): Test end-to-end workflows that require
  compiling and launching real Triton kernels (structured logging, context manager,
  autotune, tensor capture).

## Test Infrastructure

### Shared Utilities
- `tests/test_utils.py` — Common test helpers, depends on `example_output_resources`
- `tests/example_output/` — Fixture data (parsed output samples, SASS test data)

### Reproducer Tests (co-located)
- `tritonparse/reproducer/tests/` — Unit tests for the reproducer module
- `tritonparse/reproducer/tests/artifacts/` — Sample Triton kernel files for testing

### Environment Variables
- `TEST_KEEP_OUTPUT=1` — Preserve temporary directories after tests for debugging

## Adding New Tests

1. Create the test file in the appropriate directory:
   - CPU test → `tests/cpu/test_<name>.py`
   - GPU test → `tests/gpu/test_<name>.py`

2. Add a Buck target in the corresponding `BUCK` file:

   CPU test:
   ```python
   python_unittest(
       name = "test_<name>",
       srcs = ["test_<name>.py"],
       enable_lazy_imports = True,
       deps = [
           "//pytorch/tritonparse:tritonparse_lib",
           "//pytorch/tritonparse/tests:test_utils",  # if needed
       ],
   )
   ```

   GPU test:
   ```python
   python_unittest_remote_gpu(
       name = "test_<name>",
       srcs = ["test_<name>.py"],
       compile = False,
       enable_lazy_imports = True,
       fail_if_no_gpu_used = False,
       link_group_min_binary_node_count = PYTHON_EXTREME_LINK_GROUP_THRESHOLD,
       remote_execution = re_test_utils.remote_execution(
           platform = "gpu-remote-execution",
       ),
       supports_static_listing = False,
       deps = [
           "//caffe2:torch",
           "//pytorch/tritonparse:tritonparse_lib",
           "//pytorch/tritonparse/tests:test_utils",
           "//triton:triton",
       ],
   )
   ```

3. All test targets use `oncall("triton")`.

## GitHub CI vs Buck

| Aspect | Buck (fbcode) | GitHub CI (OSS) |
|--------|--------------|-----------------|
| Runner | `python_unittest` / `python_unittest_remote_gpu` | `pytest` on `4-core-ubuntu-gpu-t4` |
| Code path | `fb_run()` (fbcode) | `oss_run()` (OSS) |
| Triton source | Buck dependency `//triton:triton` | Built from source or pip |
| Jobs | — | `test-from-source`, `test-from-pip` |

A test passing in Buck but failing in GitHub CI usually means the OSS code path
(`oss_run`) has a bug. See D97557615 for a real example.
