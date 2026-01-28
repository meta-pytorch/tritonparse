#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Install tritonparse project dependencies
# This script installs the project in editable mode with test dependencies

set -e

echo "Installing tritonparse project dependencies..."

# Ensure we're in the conda environment
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install project in editable mode with test dependencies
echo "Installing tritonparse in editable mode..."
pip install -e ".[test]"

# Verify installation
echo "Verifying installation..."
python -c "import tritonparse; print(f'tritonparse installed successfully')"
python -c "import coverage; print(f'coverage version: {coverage.__version__}')"

echo "Project installation completed successfully!"
