#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# build_pytorch.sh - Build and install PyTorch from source
#
# This script builds PyTorch via editable pip install. It assumes
# dependencies have already been installed by prepare_build_pytorch.sh.
#
# Environment Variables (set by the caller):
#   PYTORCH_SRC_DIR  Path to the PyTorch source checkout (required)
#   VISION_SRC_DIR   Path to the torchvision source checkout (required)
#   USE_UV           Use uv package manager instead of pip (default: 0)
#   CUDA_HOME        CUDA toolkit directory (required)

set -e

if [ -z "$PYTORCH_SRC_DIR" ]; then
  echo "ERROR: PYTORCH_SRC_DIR is not set"
  exit 1
fi

if [ -z "$VISION_SRC_DIR" ]; then
  echo "ERROR: VISION_SRC_DIR is not set"
  exit 1
fi

if [ -z "$CUDA_HOME" ]; then
  echo "ERROR: CUDA_HOME is not set"
  exit 1
fi
export CUDA_HOME

USE_UV=${USE_UV:-0}

if [[ "$USE_UV" == "1" ]]; then
  PIP="uv pip"
else
  PIP="pip"
fi

echo "Building PyTorch..."
cd "$PYTORCH_SRC_DIR"
if [[ "$USE_UV" == "1" ]]; then
  CMAKE_PREFIX_PATH="${VIRTUAL_ENV}:${CMAKE_PREFIX_PATH}" \
    $PIP install --no-build-isolation -v -e .
else
  CMAKE_PREFIX_PATH="${CONDA_PREFIX:-"$(dirname "$(which conda)")/../"}:${CMAKE_PREFIX_PATH}" \
    $PIP install --no-build-isolation -v -e .
fi

# Build torchvision
if [ ! -d "$VISION_SRC_DIR" ]; then
  echo "ERROR: torchvision not found at $VISION_SRC_DIR"
  echo "Run prepare_build_pytorch.sh first to clone it."
  exit 1
fi
echo "Building torchvision..."
cd "$VISION_SRC_DIR"
$PIP install --no-build-isolation -v -e "$VISION_SRC_DIR"

echo "PyTorch build completed successfully."
