#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Setup script for tritonparse CI environment
# This script sets up the conda environment, installs dependencies, and configures CUDA

set -e

# Default values
CONDA_ENV=${CONDA_ENV:-"tritonparse"}
PYTHON_VERSION=${PYTHON_VERSION:-"3.14"}
CUDA_VERSION=${CUDA_VERSION:-"12.8"}

echo "Setting up tritonparse environment..."
echo "CONDA_ENV: $CONDA_ENV"
echo "PYTHON_VERSION: $PYTHON_VERSION"
echo "CUDA_VERSION: $CUDA_VERSION"

# Install system dependencies
echo "Installing system dependencies..."

# Update package lists
echo "🔄 Updating package lists..."
sudo apt-get update

# Install clang and clangd first
echo "Installing clang and clangd..."
if command -v clang-19 &>/dev/null && command -v clangd-19 &>/dev/null; then
    echo "✅ clang-19 and clangd-19 already installed"
else
    echo "📦 Installing clang-19 and clangd-19 from Ubuntu repositories..."
    sudo apt-get install -y clang-19 clangd-19
fi

# Set clang-19 and clangd-19 as the default
echo "Setting clang-19 and clangd-19 as default..."
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100
sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100

# Install CUDA and development libraries
echo "Installing CUDA and development libraries..."

# Check for specific CUDA 12.8 version
CUDA_VERSION_REQUIRED="12.8"
HAS_CORRECT_CUDA=false
# Allow skipping CUDA installation via environment variable
INSTALL_CUDA=${INSTALL_CUDA:-true}

# Try to find nvcc in a way that is safe for `set -e`
NVCC_PATH=""
if command -v nvcc &>/dev/null; then
    NVCC_PATH=$(command -v nvcc)
    echo "Found nvcc in PATH: $NVCC_PATH"
elif [ -x "/usr/local/cuda/bin/nvcc" ]; then
    NVCC_PATH="/usr/local/cuda/bin/nvcc"
    echo "Found nvcc at $NVCC_PATH"
elif [ -x "/usr/local/cuda-12.8/bin/nvcc" ]; then
    NVCC_PATH="/usr/local/cuda-12.8/bin/nvcc"
    echo "Found nvcc at $NVCC_PATH"
fi

if [ -n "$NVCC_PATH" ]; then
    echo "Verifying CUDA version using '$NVCC_PATH -v':"
    $NVCC_PATH -v
    # Get CUDA version from nvcc
    INSTALLED_CUDA_VERSION=$($NVCC_PATH --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    if [ "$INSTALLED_CUDA_VERSION" = "$CUDA_VERSION_REQUIRED" ]; then
        echo "✅ CUDA $CUDA_VERSION_REQUIRED already installed"
        HAS_CORRECT_CUDA=true
    else
        echo "⚠️ Found CUDA $INSTALLED_CUDA_VERSION, but need $CUDA_VERSION_REQUIRED"
        HAS_CORRECT_CUDA=false
    fi
else
    echo "📦 No CUDA toolkit found in PATH or standard locations"
    HAS_CORRECT_CUDA=false
fi

echo "🔧 Installing development libraries"
sudo apt-get install -y libstdc++6 libstdc++-13-dev libffi-dev libncurses-dev zlib1g-dev libxml2-dev git build-essential cmake bc gdb curl wget

if [ "$HAS_CORRECT_CUDA" != "true" ] && [ "$INSTALL_CUDA" = "true" ]; then
    echo "📦 Installing CUDA $CUDA_VERSION_REQUIRED"
    # Install all packages including CUDA toolkit (this is the big download)
    sudo apt-get install -y cuda-toolkit-12.8
elif [ "$INSTALL_CUDA" != "true" ]; then
    echo "ℹ️ Skipping CUDA installation because INSTALL_CUDA is not 'true'."
fi

# Verify clang installation
echo "Verifying clang installation..."
clang --version
clang++ --version
clangd --version

# Install Miniconda if not already installed
if [ ! -d "/opt/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh
    chmod +x /tmp/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -u -p /opt/miniconda3
fi

# Add conda to PATH
export PATH="/opt/miniconda3/bin:$PATH"
export CONDA_HOME="/opt/miniconda3"

# Initialize conda
conda init bash || true

# Create conda environment
# NOTE: Do NOT swallow failures with `|| true` here. If env creation fails
# silently, a later `conda activate` may not switch the active Python and
# `pip` will install into the base env while `python` resolves elsewhere
# (e.g. actions/setup-python's hosted-tool-cache Python), producing a
# confusing "ModuleNotFoundError: No module named 'torch'" right after a
# successful `pip install torch`.
echo "Creating conda environment: $CONDA_ENV"
# IMPORTANT: explicitly request `pip` here. conda-forge's `python` package
# does NOT pull pip in by default, so without this the env would have
# `python` but no `pip`/`python -m pip`, causing every later
# `python -m pip install ...` to fail with "No module named pip".
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "✅ conda env '$CONDA_ENV' already exists"
else
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" pip -y -c conda-forge
fi

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Sanity check: make sure `python` and `pip` resolve to the activated env.
# If `actions/setup-python` placed another Python earlier in PATH, this
# step will surface the mismatch immediately instead of after a confusing
# `pip install` succeeds against the wrong interpreter.
#
# We deliberately validate against the runtime $CONDA_PREFIX (set by
# `conda activate`) rather than a hard-coded "/opt/miniconda3/envs/..."
# path. $CONDA_PREFIX is conda's own answer for "which env is active", so
# it stays correct if the miniconda install location or envs_dirs config
# ever changes, and it also catches the case where `conda activate`
# silently failed (in which case CONDA_PREFIX is unset or points at base).
echo "Verifying active Python interpreter..."
echo "  CONDA_PREFIX:    ${CONDA_PREFIX:-<unset>}"
echo "  which python:    $(command -v python || echo 'not found')"
echo "  which pip:       $(command -v pip || echo 'not found')"

# Guard: if `conda activate` did not put a `python` on PATH, every
# subsequent `$(python -c ...)` would emit a noisy "python: command not
# found" with no useful context. Fail fast with a clear message.
if ! command -v python >/dev/null 2>&1; then
    echo "❌ No 'python' executable found on PATH after"
    echo "   'conda activate $CONDA_ENV'. CONDA_PREFIX=${CONDA_PREFIX:-<unset>}."
    echo "   The conda activation did not produce a usable Python interpreter."
    echo "   Aborting to avoid confusing pip/python mismatch failures."
    exit 1
fi

ACTUAL_PYTHON=$(python -c 'import sys; print(sys.executable)')
echo "  sys.executable:  $ACTUAL_PYTHON"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "❌ CONDA_PREFIX is not set — 'conda activate $CONDA_ENV' did not"
    echo "   take effect. Aborting."
    exit 1
fi
if [[ "$(basename "$CONDA_PREFIX")" != "$CONDA_ENV" ]]; then
    echo "❌ Active conda env ($(basename "$CONDA_PREFIX")) does not match"
    echo "   the requested env ($CONDA_ENV). Aborting."
    exit 1
fi
if [[ "$ACTUAL_PYTHON" != "$CONDA_PREFIX"/* ]]; then
    echo "❌ Python interpreter ($ACTUAL_PYTHON) is NOT inside the activated"
    echo "   conda env (\$CONDA_PREFIX=$CONDA_PREFIX). Something earlier on"
    echo "   PATH (e.g. actions/setup-python) is shadowing the env's python."
    echo "   Aborting to avoid pip/python mismatch."
    exit 1
fi

# Ensure pip is available inside the env. A previously-cached env may have
# been created without pip (older revision of this script), so bootstrap it
# via ensurepip if `python -m pip` doesn't work yet.
if ! python -m pip --version >/dev/null 2>&1; then
    echo "⚠️ pip not found in env, bootstrapping via ensurepip..."
    python -m ensurepip --upgrade || conda install -n "$CONDA_ENV" -y -c conda-forge pip
fi
python -m pip --version

# Check NVIDIA GPU information
echo "Checking NVIDIA GPU information..."
if command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi
else
    echo "nvidia-smi not found"
fi

# Detect CUDA version
echo "Detecting CUDA version..."
if [ -d "/usr/local/cuda" ]; then
    # Use readlink to safely get the target of the symlink
    if [ -L "/usr/local/cuda" ]; then
        CUDA_TARGET=$(readlink /usr/local/cuda)
        if [[ "$CUDA_TARGET" =~ cuda-([0-9.]+) ]]; then
            DETECTED_CUDA="${BASH_REMATCH[1]}"
            CUDA_VERSION="$DETECTED_CUDA"
            echo "Found CUDA version: $CUDA_VERSION"
        fi
    else
        # If not a symlink, try to find cuda-* directories
        for cuda_dir in /usr/local/cuda-*; do
            if [ -d "$cuda_dir" ]; then
                DETECTED_CUDA=$(basename "$cuda_dir" | sed 's/cuda-//')
                CUDA_VERSION="$DETECTED_CUDA"
                echo "Found CUDA version: $CUDA_VERSION"
                break
            fi
        done
    fi
    export CUDA_HOME="/usr/local/cuda"
else
    echo "CUDA not found in /usr/local/cuda"
fi

export CUDA_VERSION="$CUDA_VERSION"
echo "Using CUDA version: $CUDA_VERSION"

# Set cuDNN version for installation
export CUDNN_VERSION=${CUDNN_VERSION:-"9.10.2.21"}
echo "Using cuDNN version: $CUDNN_VERSION"

# Install cuDNN using PyTorch's script
echo "Installing cuDNN using PyTorch's script..."
curl -s https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/docker/common/install_cuda.sh -o /tmp/install_cuda.sh
chmod +x /tmp/install_cuda.sh
# The install_cudnn function is defined in install_cuda.sh.
# We source the script and call the function with sudo to install cuDNN.
# The -E flag preserves the environment variables. The function expects
# CUDA major version (e.g., "12") and CUDNN version as arguments.
CUDA_MAJOR_VERSION="${CUDA_VERSION%%.*}"
sudo -E bash -c "source /tmp/install_cuda.sh && install_cudnn \"${CUDA_MAJOR_VERSION}\" \"${CUDNN_VERSION}\""

# Install PyTorch nightly
# Use `python -m pip` (NOT bare `pip`) so the install always targets the
# same interpreter we use for verification below. A bare `pip` on PATH may
# resolve to a different Python (e.g. base conda or actions/setup-python),
# leading to a successful install followed by ModuleNotFoundError.
echo "Installing PyTorch nightly..."
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
python -m pip install packaging  # Workaround: PyTorch nightly missing packaging dependency

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
echo "  Using python: $(command -v python)"
echo "  sys.executable: $(python -c 'import sys; print(sys.executable)')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"
fi

echo "Setup completed successfully!"
