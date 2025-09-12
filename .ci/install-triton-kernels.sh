#!/bin/bash

# Install Triton kernels from triton-lang/triton/python/triton_kernels

set -e

echo "🚀 Installing Triton kernels from triton-lang/triton/python/triton_kernels..."
START_TIME=$(date +%s)

# Function to show elapsed time
show_elapsed() {
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    echo "⏱️ Elapsed time: ${ELAPSED}s"
}

# Set Triton version/commit for cache consistency
TRITON_COMMIT=${TRITON_COMMIT:-"main"}
echo "🎯 Target Triton commit/branch: $TRITON_COMMIT"
TRITON_SOURCE_DIR="/tmp/triton"

# Ensure we're in the conda environment
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Ensure TRITON_SOURCE_DIR contains Triton source; otherwise, clone it
echo "🔧 Ensuring Triton source exists at $TRITON_SOURCE_DIR..."

if [ -d "$TRITON_SOURCE_DIR/.git" ]; then
    REMOTE_URL=$(git -C "$TRITON_SOURCE_DIR" remote get-url origin 2>/dev/null || echo "")
    if [[ "$REMOTE_URL" == *"triton-lang/triton"* ]]; then
        echo "✅ Found existing Triton repository: $REMOTE_URL"
    else
        echo "⚠️ Existing directory is not triton-lang/triton (origin: $REMOTE_URL). Re-cloning..."
        rm -rf "$TRITON_SOURCE_DIR"
    fi
fi

if [ ! -d "$TRITON_SOURCE_DIR/.git" ]; then
    echo "Cloning Triton repository..."
    if ! git clone https://github.com/triton-lang/triton.git "$TRITON_SOURCE_DIR"; then
        echo "❌ ERROR: Failed to clone Triton repository"
        echo "This might be due to network issues or GitHub rate limiting"
        exit 1
    fi
fi

echo "Checking out Triton commit/branch: $TRITON_COMMIT"
if ! git -C "$TRITON_SOURCE_DIR" checkout "$TRITON_COMMIT"; then
    echo "❌ ERROR: Failed to checkout $TRITON_COMMIT"
    exit 1
fi

# Install triton_kernels in editable mode
KERNELS_DIR="$TRITON_SOURCE_DIR/python/triton_kernels"
if [ ! -d "$KERNELS_DIR" ]; then
    echo "❌ ERROR: triton_kernels directory not found at $KERNELS_DIR"
    exit 1
fi

echo "📦 Installing triton_kernels from $KERNELS_DIR (editable)..."
pip install -e "$KERNELS_DIR"
show_elapsed

# Verify installation with a simple import
echo "🔎 Verifying triton_kernels installation..."
set +e
KERNELS_IMPORT_OUTPUT=$(python -c "import triton_kernels; import os; print('triton_kernels OK'); print(getattr(triton_kernels, '__file__', 'no_file'))" 2>&1)
KERNELS_IMPORT_EXITCODE=$?
set -e

echo "Import exit code: $KERNELS_IMPORT_EXITCODE"
echo "Import output: $KERNELS_IMPORT_OUTPUT"

if [ $KERNELS_IMPORT_EXITCODE -ne 0 ]; then
    echo "❌ ERROR: Failed to import triton_kernels"
    exit 1
fi

echo "✅ triton_kernels installation verified"
show_elapsed
