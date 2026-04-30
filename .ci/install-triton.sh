#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Install Triton from source
# This script clones and installs Triton from the main repository

set -e

echo "🚀 Installing Triton from source..."
START_TIME=$(date +%s)

# Function to show elapsed time
show_elapsed() {
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo "⏱️ Elapsed time: ${ELAPSED}s"
}

# Pre-flight checks
echo "🔍 Running pre-flight checks..."

# Set Triton version/commit for cache consistency
TRITON_COMMIT=${TRITON_COMMIT:-"main"}
echo "🎯 Target Triton commit/branch: $TRITON_COMMIT"
TRITON_CACHE_DIR="/tmp/triton-cache"
TRITON_SOURCE_DIR="/tmp/triton"

# Check disk space (need at least 10GB for Triton compilation)
AVAILABLE_SPACE=$(df /tmp | tail -1 | awk '{print $4}')
REQUIRED_SPACE=10485760 # 10GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "⚠️ WARNING: Low disk space. Available: $(($AVAILABLE_SPACE / 1024 / 1024))GB, Recommended: 10GB"
else
    echo "✅ Sufficient disk space available: $(($AVAILABLE_SPACE / 1024 / 1024))GB"
fi

# Ensure we're in the conda environment
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi

# Activate conda environment.
# Locate conda installation: prefer existing $CONDA_HOME, else discover
# via 'conda info --base' if conda is in PATH, else fall back to the
# legacy /opt/miniconda3 path used by .ci/setup.sh on GitHub-hosted
# runners.
if [ -z "${CONDA_HOME:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_HOME="$(conda info --base)"
    else
        CONDA_HOME="/opt/miniconda3"
    fi
fi
source "${CONDA_HOME}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Create cache directory
mkdir -p "$TRITON_CACHE_DIR"

# Check if we have cached source with correct commit
if [ -f "$TRITON_CACHE_DIR/commit" ] && [ -d "$TRITON_SOURCE_DIR" ]; then
    CACHED_COMMIT=$(cat "$TRITON_CACHE_DIR/commit")
    if [ "$CACHED_COMMIT" = "$TRITON_COMMIT" ] && [ "$TRITON_COMMIT" != "main" ]; then
        echo "Found cached Triton source with correct commit ($CACHED_COMMIT)"
        echo "Will use cached source and re-install to new conda environment"
        USE_CACHED_SOURCE=true
    elif [ "$TRITON_COMMIT" = "main" ]; then
        echo "Target is 'main' branch (API fallback), will rebuild from scratch"
        echo "Cached commit: $CACHED_COMMIT"
        USE_CACHED_SOURCE=false
    else
        echo "Cached source commit mismatch: cached=$CACHED_COMMIT, target=$TRITON_COMMIT"
        echo "Will rebuild from scratch"
        USE_CACHED_SOURCE=false
    fi
else
    echo "No cached source found or no commit info, will build from scratch"
    USE_CACHED_SOURCE=false
fi

# Update libstdc++ to match system version
# Otherwise, we get errors like:
# ImportError: <conda_env>/bin/../lib/libstdc++.so.6:
# version `GLIBCXX_3.4.30' not found (required by /tmp/triton/python/triton/_C/libtriton.so)
echo "Updating libstdc++ to match system version..."
# Use the latest version for Ubuntu 22.04 that includes GLIBCXX_3.4.32
conda install -y -c conda-forge libstdcxx-ng=15.1.0
# Check if the update was successful
echo "Checking libstdc++ version after update:"
# Use $CONDA_PREFIX (set by 'conda activate') so this works regardless of
# whether conda is at /opt/miniconda3 (legacy setup.sh path) or /opt/conda
# (container path) or anywhere else.
strings "${CONDA_PREFIX}/lib/libstdc++.so.6" | grep GLIBCXX | tail -10

# Uninstall existing pytorch-triton
echo "Uninstalling existing pytorch-triton..."
pip uninstall -y pytorch-triton || true
pip uninstall -y triton || true

# Setup Triton repository based on cache status
if [ "$USE_CACHED_SOURCE" = "true" ]; then
    echo "Using cached Triton source..."
    cd "$TRITON_SOURCE_DIR"
    ACTUAL_COMMIT=$(git rev-parse HEAD)
    echo "Using cached Triton commit: $ACTUAL_COMMIT"
else
    echo "Setting up Triton repository from scratch..."
    if [ -d "$TRITON_SOURCE_DIR" ]; then
        echo "Removing existing source directory..."
        rm -rf "$TRITON_SOURCE_DIR"
    fi

    echo "Cloning Triton repository..."
    if ! git clone https://github.com/triton-lang/triton.git "$TRITON_SOURCE_DIR"; then
        echo "❌ ERROR: Failed to clone Triton repository"
        echo "This might be due to network issues or GitHub rate limiting"
        exit 1
    fi

    cd "$TRITON_SOURCE_DIR"

    # Checkout specific commit for reproducibility
    echo "Checking out commit: $TRITON_COMMIT"
    if ! git checkout "$TRITON_COMMIT"; then
        echo "❌ ERROR: Failed to checkout commit $TRITON_COMMIT"
        echo "This might be due to an invalid commit hash or network issues"
        exit 1
    fi

    ACTUAL_COMMIT=$(git rev-parse HEAD)
    echo "✅ Using Triton commit: $ACTUAL_COMMIT"
fi

# Install build dependencies
echo "Installing build dependencies..."
pip install ninja cmake wheel pybind11

# Install Triton requirements
echo "Installing Triton requirements..."
pip install -r python/requirements.txt

# Set environment to use clang compiler for faster compilation IF available.
# In the GitHub-hosted Ubuntu CI path, .ci/setup.sh apt-installs clang-19;
# in the container CI path (pytorch/almalinux-builder:cuda13.0) only gcc
# (gcc-toolset-13) is preinstalled, so we fall back to gcc to let CMake
# auto-detect rather than failing with "Could not find compiler clang++".
# Local development on a workstation without clang also benefits from
# this fallback.
if command -v clang &>/dev/null && command -v clang++ &>/dev/null; then
    echo "Setting up clang compiler for faster compilation..."
    export CC=clang
    export CXX=clang++
else
    echo "clang/clang++ not found; using default compiler (gcc)..."
fi
echo "Using CC: ${CC:-<default>}"
echo "Using CXX: ${CXX:-<default>}"

# Install Triton in editable mode with clang
if [ "$USE_CACHED_SOURCE" = "true" ]; then
    echo "Installing cached Triton to new conda environment..."
    echo "This should be fast since build artifacts are cached"
else
    echo "Compiling and installing Triton from scratch..."
    echo "This will take 30-50 minutes for compilation"
fi
pip install -e .
show_elapsed

# Verify Triton installation
echo "Verifying Triton installation..."
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Testing basic Python functionality..."
python -c "print('Python works')" || echo "❌ Basic Python test failed"
echo "Attempting to import triton..."

set +e  # Temporarily disable exit on error
IMPORT_OUTPUT=$(python -c "import triton; print(f'Triton version: {triton.__version__}')" 2>&1)
IMPORT_EXITCODE=$?
set -e  # Re-enable exit on error

echo "Import exit code: $IMPORT_EXITCODE"
echo "Import output: $IMPORT_OUTPUT"

if [ $IMPORT_EXITCODE -eq 0 ]; then
    echo "$IMPORT_OUTPUT"
    python -c "import triton; print(f'Triton path: {triton.__file__}')"
    echo "✅ Triton installation verified successfully"

    # Only save commit info after successful verification
    echo "$ACTUAL_COMMIT" >"$TRITON_CACHE_DIR/commit"
    echo "✅ Cache information saved"

    show_elapsed
    echo "🎉 Triton installation completed successfully!"
else
    echo "❌ ERROR: Failed to import triton"
    echo "Import error details:"
    echo "$IMPORT_OUTPUT"
    echo ""
    echo "Additional diagnostic information:"
    echo "Installed packages containing 'triton':"
    pip list | grep -i triton || echo "No triton packages found"
    echo ""
    echo "Python sys.path:"
    python -c "import sys; print('\n'.join(sys.path))"
    echo ""
    echo "Checking if triton directory exists in site-packages:"
    python -c "import site; print([p for p in site.getsitepackages()])" 2>/dev/null || echo "Could not get site-packages"
    find $(python -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null) -name "*triton*" 2>/dev/null || echo "Could not find triton in site-packages"

    # Clean up cache on failure to prevent corruption
    echo "🧹 Cleaning up cache due to installation failure..."
    rm -f "$TRITON_CACHE_DIR/commit"

    exit 1
fi
