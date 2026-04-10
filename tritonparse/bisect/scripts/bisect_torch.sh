#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# bisect_torch.sh - Bisect PyTorch commits to find regressions
#
# This script is designed to be used with `git bisect run` to automatically
# find the first commit that introduces a regression in PyTorch.
#
# Usage:
#   cd /path/to/pytorch
#   git bisect start
#   git bisect good <known-good-commit>
#   git bisect bad <known-bad-commit>
#   TORCH_DIR=/path/to/pytorch TEST_SCRIPT=/path/to/test.py git bisect run bash bisect_torch.sh
#
# For standalone help: bash bisect_torch.sh --help

# Help message
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  cat << 'EOF'
PyTorch Bisect Script

This script is designed to be used with `git bisect run` to automatically
find the first commit that introduces a regression in PyTorch.

Usage:
  cd /path/to/pytorch
  git bisect start
  git bisect good <known-good-commit>
  git bisect bad <known-bad-commit>
  TORCH_DIR=/path/to/pytorch TEST_SCRIPT=/path/to/test.py git bisect run bash bisect_torch.sh

Required Environment Variables:
  TORCH_DIR           Path to PyTorch repository (can be auto-detected if in repo)
  TEST_SCRIPT         Path to test script that returns 0 for pass, non-0 for fail

Optional Environment Variables (with defaults):
  CONDA_ENV           Conda environment name (default: triton_bisect)
  CONDA_DIR           Conda directory (default: $HOME/miniconda3)
  LOG_DIR             Log directory (default: ./bisect_logs)
  TEST_ARGS           Arguments for test script (default: empty)
  BUILD_COMMAND       Build command (default: bash build_pytorch.sh)
  PER_COMMIT_LOG      Write per-commit log files (default: 1, set to 0 to disable)

Exit Codes (for git bisect):
  0   - Good commit (test passed)
  1   - Bad commit (test failed)
  125 - Skip (currently unused, reserved for future use)
  128 - Abort (build failed or configuration error, stops bisect)

Example:
  # Basic usage
  cd /path/to/pytorch
  git bisect start
  git bisect good v2.6.0
  git bisect bad HEAD
  TORCH_DIR=$(pwd) TEST_SCRIPT=/path/to/test.py git bisect run bash bisect_torch.sh

  # With custom environment
  TORCH_DIR=/path/to/pytorch \
  TEST_SCRIPT=/path/to/test.py \
  CONDA_ENV=my_env \
  LOG_DIR=/path/to/logs \
  git bisect run bash bisect_torch.sh

  # Disable per-commit log files (only keep commands.log)
  PER_COMMIT_LOG=0 \
  TORCH_DIR=/path/to/pytorch \
  TEST_SCRIPT=/path/to/test.py \
  git bisect run bash bisect_torch.sh
EOF
  exit 0
fi

# Resolve the directory containing this script (for locating build_pytorch.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
TORCH_DIR=${TORCH_DIR:-""}
TEST_SCRIPT=${TEST_SCRIPT:-""}
CONDA_ENV=${CONDA_ENV:-triton_bisect}
CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
USE_UV=${USE_UV:-0}
LOG_DIR=${LOG_DIR:-./bisect_logs}
TEST_ARGS=${TEST_ARGS:-""}
BUILD_COMMAND=${BUILD_COMMAND:-"bash ${SCRIPT_DIR}/build_pytorch.sh"}
PER_COMMIT_LOG=${PER_COMMIT_LOG:-1}  # Set to 0 to disable per-commit log files

# ============ Validation ============
if [ -z "$TORCH_DIR" ]; then
  # Try to auto-detect if we're in a pytorch repo
  if [ -d ".git" ] && [ -f "setup.py" ] && [ -d "torch" ]; then
    TORCH_DIR=$(pwd)
  else
    echo "ERROR: TORCH_DIR is not set and cannot be auto-detected"
    echo "Run 'bash bisect_torch.sh --help' for usage information"
    exit 128
  fi
fi

if [ -z "$TEST_SCRIPT" ]; then
  echo "ERROR: TEST_SCRIPT is not set"
  echo "Run 'bash bisect_torch.sh --help' for usage information"
  exit 128
fi

if [ ! -d "$TORCH_DIR" ]; then
  echo "ERROR: TORCH_DIR not found: $TORCH_DIR"
  exit 128
fi

if [ ! -f "$TEST_SCRIPT" ]; then
  echo "ERROR: TEST_SCRIPT not found: $TEST_SCRIPT"
  exit 128
fi

# Convert all path variables to absolute paths to avoid issues after cd
TORCH_DIR=$(realpath "$TORCH_DIR")
TEST_SCRIPT=$(realpath "$TEST_SCRIPT")
CONDA_DIR=$(realpath "$CONDA_DIR")

# Create log directory and get absolute path
mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")

# ============ Setup ============
cd "$TORCH_DIR" || exit 128

# Get current commit info
COMMIT_HASH=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=9 HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create per-commit log file (optional, controlled by PER_COMMIT_LOG)
COMMIT_LOG=""
if [ "$PER_COMMIT_LOG" = "1" ]; then
  COMMIT_LOG="$LOG_DIR/${TIMESTAMP}_bisect_torch_${SHORT_COMMIT}.log"
fi

# Helper function for logging output
log_output() {
  if [ -n "$COMMIT_LOG" ]; then
    tee -a "$COMMIT_LOG"
  else
    cat
  fi
}

# Start logging to per-commit log file (if enabled)
{
  echo "=== PyTorch Bisect Run ==="
  echo "Timestamp: $(date)"
  echo "Commit: $COMMIT_HASH"
  echo "Short: $SHORT_COMMIT"
  echo "PyTorch Dir: $TORCH_DIR"
  echo "Test Script: $TEST_SCRIPT"
  echo "Test Args: $TEST_ARGS"
  echo "Conda Env: $CONDA_ENV"
  echo "Commit Log: $COMMIT_LOG"
  echo "========================="
  echo ""
} | log_output

# Update git submodules to match the current commit
echo "Updating git submodules..." | log_output
git submodule update --init --recursive 2>&1 | log_output
echo "" | log_output

# Activate conda or uv (if enabled)
if [ "$USE_UV" == "0" ]; then
  echo "Activating conda environment: $CONDA_ENV" | log_output
  conda activate "$CONDA_ENV" || true
else
  echo "Activating uv virtualenv: $CONDA_ENV" | log_output
  # shellcheck source=/dev/null
  if ! source "${CONDA_DIR}/bin/activate"; then
    echo "ERROR: Cannot activate conda or uv" | log_output
    exit 128
  fi
fi

echo "" | log_output

# Clean build directory to avoid stale artifacts from previous commits
echo "Cleaning build directory..." | log_output
rm -rf "$TORCH_DIR/build"
echo "" | log_output

# Uninstall any existing pytorch to avoid conflicts
echo "Uninstalling existing pytorch packages..." | log_output
if [[ "$USE_UV" == "1" ]]; then
  uv pip uninstall -y torch pytorch 2>&1 | log_output || true
else
  pip uninstall -y torch pytorch 2>&1 | log_output || true
fi
echo "" | log_output

# Build PyTorch
echo "Building PyTorch..." | log_output
BUILD_START=$(date +%s)

if [ -n "$COMMIT_LOG" ]; then
  eval "$BUILD_COMMAND" 2>&1 | tee -a "$COMMIT_LOG"
  BUILD_CODE=${PIPESTATUS[0]}
else
  eval "$BUILD_COMMAND" 2>&1
  BUILD_CODE=$?
fi

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
echo "Build completed in ${BUILD_TIME}s, exit code: $BUILD_CODE" | log_output

if [ "$BUILD_CODE" -ne 0 ]; then
  echo "Build FAILED" | log_output
  exit 128
fi

echo "" | log_output

# Run test
echo "Running test..." | log_output
TEST_START=$(date +%s)

if [ -n "$COMMIT_LOG" ]; then
  # shellcheck disable=SC2086
  python "$TEST_SCRIPT" $TEST_ARGS 2>&1 | tee -a "$COMMIT_LOG"
  TEST_CODE=${PIPESTATUS[0]}
else
  # shellcheck disable=SC2086
  python "$TEST_SCRIPT" $TEST_ARGS 2>&1
  TEST_CODE=$?
fi

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))
echo "Test completed in ${TEST_TIME}s, exit code: $TEST_CODE" | log_output

# Report result
if [ "$TEST_CODE" -eq 0 ]; then
  RESULT="GOOD"
  echo "✅ Passed" | log_output
else
  RESULT="BAD"
  echo "❌ Failed" | log_output
fi

echo "" | log_output
{
  echo "=== Summary ==="
  echo "Commit: $SHORT_COMMIT"
  echo "Build: ${BUILD_TIME}s (exit $BUILD_CODE)"
  echo "Test: ${TEST_TIME}s (exit ${TEST_CODE})"
  echo "Result: $RESULT"
  if [ -n "$COMMIT_LOG" ]; then
    echo "Log: $COMMIT_LOG"
  fi
  echo "==============="
} | log_output

# Exit with test code for git bisect
exit "$TEST_CODE"
