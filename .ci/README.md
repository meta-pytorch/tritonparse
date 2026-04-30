# CI Scripts

This directory contains scripts for setting up and running the tritonparse CI environment.

## Scripts Overview

### `setup.sh`
Sets up the conda environment, installs dependencies, configures CUDA, and installs cuDNN.

**Environment Variables:**
- `CONDA_ENV`: Conda environment name (default: "tritonparse")
- `PYTHON_VERSION`: Python version (default: "3.14")
- `CUDA_VERSION`: CUDA version (default: "12.8")
- `CUDNN_VERSION`: cuDNN version (default: "9.10.2.21")

**Usage:**
```bash
CONDA_ENV=tritonparse PYTHON_VERSION=3.11 bash .ci/setup.sh
```

> **Note:**
> `setup.sh` will automatically download and execute the official PyTorch cuDNN installation script from:
> https://github.com/pytorch/pytorch/blob/main/.ci/docker/common/install_cudnn.sh
> There is no need to maintain a local cuDNN install script.

### `install-triton.sh`
Installs Triton from source by cloning the repository and building it.

> **Note:** As of the migration to the upstream Triton team's
> Triton-Nightly wheel feed, the CI workflow no longer invokes this
> script. It is kept for local development and for cases where you
> want to test against a specific upstream Triton commit (set
> `TRITON_COMMIT=<sha>` before invoking).

**Environment Variables:**
- `CONDA_ENV`: Conda environment name (required)
- `TRITON_COMMIT`: Optional commit SHA or branch (default: "main")

**Usage:**
```bash
CONDA_ENV=tritonparse bash .ci/install-triton.sh
```

### `install-project.sh`
Installs the tritonparse project in editable mode with test dependencies.

**Environment Variables:**
- `CONDA_ENV`: Conda environment name (required)

**Usage:**
```bash
CONDA_ENV=tritonparse bash .ci/install-project.sh
```

### `run-tests.sh`
Runs the test suite with proper environment setup.

**Environment Variables:**
- `CONDA_ENV`: Conda environment name (required)
- `TEST_TYPE`: Type of tests to run (default: "all")
  - `cpu`: CPU tests only
  - `cuda`: CUDA tests only
  - `all`: All tests
- `VERBOSE`: Enable verbose output (default: "true")
- `COVERAGE`: Enable coverage reporting (default: "false")

**Usage:**
```bash
# Run all tests
CONDA_ENV=tritonparse bash .ci/run-tests.sh

# Run CPU tests only with coverage
CONDA_ENV=tritonparse TEST_TYPE=cpu COVERAGE=true bash .ci/run-tests.sh

# Run CUDA tests only
CONDA_ENV=tritonparse TEST_TYPE=cuda bash .ci/run-tests.sh
```

## Complete Workflow

For a complete setup and test run:

```bash
# 1. Setup environment (includes cuDNN installation)
CONDA_ENV=tritonparse PYTHON_VERSION=3.11 bash .ci/setup.sh

# 2. Install Triton
CONDA_ENV=tritonparse bash .ci/install-triton.sh

# 3. Install project
CONDA_ENV=tritonparse bash .ci/install-project.sh

# 4. Run tests
CONDA_ENV=tritonparse TEST_TYPE=all COVERAGE=true bash .ci/run-tests.sh
```

## Local Development

For local development, you can use these scripts to set up the same environment as CI:

```bash
# Setup local environment (includes cuDNN installation)
CONDA_ENV=tritonparse-local bash .ci/setup.sh

# Install Triton
CONDA_ENV=tritonparse-local bash .ci/install-triton.sh

# Install project
CONDA_ENV=tritonparse-local bash .ci/install-project.sh

# Run tests
CONDA_ENV=tritonparse-local bash .ci/run-tests.sh
```

## Script Features

- **Error handling**: All scripts use `set -e` to stop on errors
- **Environment validation**: Scripts check for required environment variables
- **Verbose output**: Detailed logging for debugging
- **Modular design**: Each script has a single responsibility
- **Reusable**: Scripts can be used in different contexts (CI, local development)

## Dependencies

The scripts assume:
- Linux environment
- Git available
- Internet access for downloading packages
- Sufficient disk space for conda and packages
