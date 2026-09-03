# Tritonparse Tests

This directory contains the test suite for tritonparse, including both automated unit tests and manual test examples.

## Test Structure

### Automated Tests
- `test_tritonparse.py`: Comprehensive automated test suite using unittest
- `__init__.py`: Makes the directory a Python package

### Manual Test Examples
- `test_add.py`: Manual test example for Triton kernel addition (not included in automated tests)

### Sample Data
- `example_output/`: Example output directory containing:
  - `logs/`: Sample log files
  - `parsed_output_complex/`: Sample parsed output files

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -e ".[test]"
```

### Automated Tests

#### Running All Tests
```bash
# Using unittest directly
python -m unittest tests.test_tritonparse -v

# With print statements visible (unittest shows prints by default on failure)
python -m unittest tests.test_tritonparse -v

# For coverage (requires coverage package)
coverage run -m unittest tests.test_tritonparse
coverage html
```

#### Running Specific Test Categories
```bash
# CPU-only tests
python -m unittest tests.test_tritonparse.TestTritonparseCPU -v

# CUDA tests only (requires CUDA)
python -m unittest tests.test_tritonparse.TestTritonparseCUDA -v

# Specific test function
python -m unittest tests.test_tritonparse.TestTritonparseCUDA.test_whole_workflow -v
```

### Manual Test Example

The `test_add.py` file serves as a manual test example that demonstrates:

1. **Basic Triton kernel implementation**
2. **Logging setup and usage**
3. **Manual test execution**
4. **Parsing workflow**

#### Running Manual Test
```bash
# Direct execution (not through unittest)
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TRITONPARSE_DEBUG=1 python tests/test_add.py
```

This will:
- Execute a simple tensor addition kernel
- Generate log files in `./logs`
- Parse the logs and output to `./parsed_output`

## Test Categories

### CPU Tests (No CUDA Required)
- `TestTritonparseCPU.test_convert()`: Tests data conversion functionality with various data types

### CUDA Tests (Require GPU)
- `TestTritonparseCUDA.test_extract_python_source_info()`: Tests Python source code extraction during Triton compilation
- `TestTritonparseCUDA.test_whole_workflow()`: Tests complete workflow from kernel execution to log parsing

## Test Features

### Test Setup
- `TestTritonparseCUDA.setUp()`: Manages Triton hooks and compilation settings
  - Saves and restores all triton knobs (compilation and runtime hooks)
  - Ensures clean state between tests
  - Automatically restores settings even if tests fail
- `TestTritonparseCUDA.tearDown()`: Restores original settings after each test

### CUDA Device Management
- Built-in CUDA availability checking in test setUp methods
- Automatic skipping of CUDA tests when CUDA is not available using `@unittest.skipUnless`
- Self-managed CUDA device assignment in test classes

### Kernel Isolation
Each test function defines its own Triton kernel to avoid compilation cache interference:
- `extract_test_kernel`: Simple multiplication kernel (x * 3.0)
- `test_kernel`: Simple addition kernel (x + 1.0)

## Environment Variables

The following environment variables are used during testing:

- `TORCHINDUCTOR_FX_GRAPH_CACHE=0`: Disable FX graph caching
- `TRITONPARSE_DEBUG=1`: Enable debug logging
- `CUDA_VISIBLE_DEVICES=0`: Use first CUDA device (in CI)

## Test Configuration

### Unittest Configuration
- CUDA availability checking built into test classes
- Setup and teardown methods for proper test isolation
- Class-based organization for logical test grouping

### Test Isolation
- Each test method has its own kernel definition
- setUp/tearDown methods ensure clean state between tests
- Temporary directories are automatically cleaned up

## Adding New Tests

### For Automated Tests
1. Add test methods to appropriate TestCase classes in `test_tritonparse.py`
2. Use `@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")` for CUDA tests
3. Add tests to `TestTritonparseCUDA` class for tests that modify Triton settings (automatic setUp/tearDown)
4. Define kernels inside test methods to avoid cache interference

Example:
```python
class TestTritonparseCUDA(unittest.TestCase):
    def setUp(self):
        # Automatic setup for CUDA tests
        pass

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_my_function(self):
        @triton.jit
        def my_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            # Your kernel logic here
            pass

        # Your test code here
        pass
```

### For Manual Tests
1. Create new files following the pattern of `test_add.py`
2. Include direct execution with `if __name__ == "__main__"`
3. Document the test purpose and execution method

## Continuous Integration

Tests are automatically run on GitHub Actions for:
- CUDA versions (12.8)
- Code coverage reporting
- Linting and formatting checks

## Troubleshooting

### Common Issues
1. **CUDA not available**: Tests will be skipped automatically
2. **No log files generated**: Check that Triton compilation is working
3. **Import errors**: Ensure all dependencies are installed

### Debug Mode
```bash
# Run with verbose output and print statements
python -m unittest tests.test_tritonparse -v
```

### Manual Verification
```bash
# Run manual test to verify basic functionality
python tests/test_add.py
```

### Example Output
The `example_output/` directory holds the fixtures the CPU tests read:

```
example_output/
├── logs/
│   └── dedicated_log_triton_trace_findhao_.ndjson   # raw, straight off the writer
├── parsed_output_complex/
│   ├── dedicated_log_triton_trace_findhao__mapped.ndjson.gz
│   └── log_file_list.json
└── sass_test_data/
    └── test_kernel.sass
```

These files can be used to:
- Verify parsing functionality
- Understand expected output format
- Debug parsing issues
- Test with real log data

They are generated, not hand-maintained. Regenerate them with:

```bash
make regen-examples-install   # requires a CUDA GPU
```

which also refreshes the example the website serves. A few tests pin exact
counts from these files (`test_parsed_trace_event_counts`,
`test_list_kernels_multiple`); update those alongside a regeneration.
