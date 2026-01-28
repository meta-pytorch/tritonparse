#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Run tritonparse tests
# This script runs the test suite with proper environment setup
#
# Test structure:
#   tests/cpu/  - CPU-only tests (no GPU required)
#   tests/gpu/  - GPU tests (require CUDA)

set -e

# Default values
TEST_TYPE=${TEST_TYPE:-"all"}
VERBOSE=${VERBOSE:-"true"}
COVERAGE=${COVERAGE:-"false"}

echo "Running tritonparse tests..."
echo "TEST_TYPE: $TEST_TYPE"
echo "VERBOSE: $VERBOSE"
echo "COVERAGE: $COVERAGE"

# Ensure we're in the conda environment
if [ -z "$CONDA_ENV" ]; then
    echo "ERROR: CONDA_ENV is not set"
    exit 1
fi

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Set environment variables
# export TORCHINDUCTOR_FX_GRAPH_CACHE=0
# export TRITONPARSE_DEBUG=1

# Build verbose flag
if [ "$VERBOSE" = "true" ]; then
    VERBOSE_FLAG="-v"
else
    VERBOSE_FLAG=""
fi

# Build unittest command based on test type
case "$TEST_TYPE" in
"cpu")
    echo "Running CPU tests only (tests/cpu/)..."
    if [ "$COVERAGE" = "true" ]; then
        echo "Running with coverage..."
        coverage run -m unittest discover -s tests/cpu -t . $VERBOSE_FLAG
        coverage report
        coverage xml
    else
        python -m unittest discover -s tests/cpu -t . $VERBOSE_FLAG
    fi
    ;;
"cuda"|"gpu")
    echo "Running GPU tests only (tests/gpu/)..."
    export CUDA_VISIBLE_DEVICES=0
    if [ "$COVERAGE" = "true" ]; then
        echo "Running with coverage..."
        coverage run -m unittest discover -s tests/gpu -t . $VERBOSE_FLAG
        coverage report
        coverage xml
    else
        python -m unittest discover -s tests/gpu -t . $VERBOSE_FLAG
    fi
    ;;
"all")
    echo "Running all tests (tests/cpu/ + tests/gpu/)..."
    export CUDA_VISIBLE_DEVICES=0
    if [ "$COVERAGE" = "true" ]; then
        echo "Running with coverage..."
        coverage run -m unittest discover -s tests -t . $VERBOSE_FLAG
        coverage report
        coverage xml
    else
        python -m unittest discover -s tests -t . $VERBOSE_FLAG
    fi
    ;;
*)
    echo "Unknown test type: $TEST_TYPE"
    echo "Available options: cpu, cuda (or gpu), all"
    exit 1
    ;;
esac

echo "Tests completed successfully!"
