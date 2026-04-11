#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# prepare_build_pytorch.sh - Install PyTorch build dependencies
#
# This script installs dev dependencies, MKL, MAGMA, and builds the vendored
# Triton copy. It is intended to be run once before repeated bisect builds.
#
# Environment Variables (set by the caller):
#   PYTORCH_SRC_DIR  Path to the PyTorch source checkout (required)
#   VISION_SRC_DIR   Path to the torchvision source checkout (required)
#   CUDA_HOME        CUDA toolkit directory (required)
#   USE_UV           Use uv package manager instead of pip (default: 0)
#   CUDA_VERSION     CUDA version to install (default: 12.8)

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

USE_UV=${USE_UV:-0}
CUDA_VERSION=${CUDA_VERSION:-12.8}

if [[ "$USE_UV" == "1" ]]; then
  PIP="uv pip"
else
  PIP="pip"
fi

# Resolve the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 0: Install CUDA
bash "${SCRIPT_DIR}/install_cuda.sh" "$CUDA_VERSION"

# Step 1: Install dev dependencies
echo "Installing dev dependencies..."
cd "$PYTORCH_SRC_DIR"
$PIP install --group dev

# Step 2: Install MKL
echo "Installing mkl-static and mkl-include..."
$PIP install mkl-static mkl-include

# Step 3: Install MAGMA
echo "Installing MAGMA..."
bash "${SCRIPT_DIR}/install_magma.sh" "$CUDA_VERSION"

# Step 4: Build Triton (PyTorch's vendored copy)
echo "Building vendored Triton..."
make triton

# Step 5: Install timm and huggingface from commit pins
PINS_DIR=".ci/docker/ci_commit_pins"
if [ -d "$PINS_DIR" ]; then
  if [ ! -f "$PINS_DIR/timm.txt" ]; then
    echo "ERROR: $PINS_DIR/timm.txt not found"
    exit 1
  fi
  TIMM_PIN=$(tr -d '[:space:]' < "$PINS_DIR/timm.txt")
  echo "Installing timm at commit $TIMM_PIN..."
  $PIP install "git+https://github.com/huggingface/pytorch-image-models.git@${TIMM_PIN}"

  if [ ! -f "$PINS_DIR/huggingface-requirements.txt" ]; then
    echo "ERROR: $PINS_DIR/huggingface-requirements.txt not found"
    exit 1
  fi
  echo "Installing huggingface requirements..."
  $PIP install -r "$PINS_DIR/huggingface-requirements.txt"
else
  echo "ERROR: $PINS_DIR not found in PyTorch checkout"
  exit 1
fi

echo "PyTorch build preparation completed successfully."
