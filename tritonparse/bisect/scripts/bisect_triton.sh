#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# bisect_triton.sh - Bisect Triton commits to find regressions
#
# This script is designed to be used with `git bisect run` to automatically
# find the first commit that introduces a regression in Triton.
#
# Usage:
#   cd /path/to/triton
#   git bisect start
#   git bisect good <known-good-commit>
#   git bisect bad <known-bad-commit>
#   TRITON_DIR=/path/to/triton TEST_SCRIPT=/path/to/test.py git bisect run bash bisect_triton.sh
#
# For standalone help: bash bisect_triton.sh --help

# Help message
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  cat << 'EOF'
Triton Bisect Script

This script is designed to be used with `git bisect run` to automatically
find the first commit that introduces a regression in Triton.

Usage:
  cd /path/to/triton
  git bisect start
  git bisect good <known-good-commit>
  git bisect bad <known-bad-commit>
  TRITON_DIR=/path/to/triton TEST_SCRIPT=/path/to/test.py git bisect run bash bisect_triton.sh

Required Environment Variables:
  TRITON_DIR          Path to triton repository (can be auto-detected if in repo)
  TEST_SCRIPT         Path to test script that returns 0 for pass, non-0 for fail

Optional Environment Variables (with defaults):
  CONDA_ENV           Conda environment name (default: triton_bisect)
  CONDA_DIR           Conda directory (default: $HOME/miniconda3)
  LOG_DIR             Log directory (default: ./bisect_logs)
  TEST_ARGS           Arguments for test script (default: empty)
  BUILD_COMMAND       Build command (default: pip install -e .)
  PER_COMMIT_LOG      Write per-commit log files (default: 1, set to 0 to disable)

Exit Codes (for git bisect):
  0   - Good commit (test passed)
  1   - Bad commit (test failed)
  125 - Skip (currently unused, reserved for future use)
  128 - Abort (build failed or configuration error, stops bisect)

Example:
  # Basic usage
  cd /path/to/triton
  git bisect start
  git bisect good v3.4.0
  git bisect bad v3.5.0
  TRITON_DIR=$(pwd) TEST_SCRIPT=/path/to/test.py git bisect run bash bisect_triton.sh

  # With custom environment
  TRITON_DIR=/path/to/triton \
  TEST_SCRIPT=/path/to/test.py \
  CONDA_ENV=my_env \
  LOG_DIR=/path/to/logs \
  git bisect run bash bisect_triton.sh

  # Disable per-commit log files (only keep commands.log)
  PER_COMMIT_LOG=0 \
  TRITON_DIR=/path/to/triton \
  TEST_SCRIPT=/path/to/test.py \
  git bisect run bash bisect_triton.sh
EOF
  exit 0
fi

# Default values
TRITON_DIR=${TRITON_DIR:-""}
TEST_SCRIPT=${TEST_SCRIPT:-""}
CONDA_ENV=${CONDA_ENV:-triton_bisect}
CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
USE_UV=${USE_UV:-0}
LOG_DIR=${LOG_DIR:-./bisect_logs}
TEST_ARGS=${TEST_ARGS:-""}
if [[ "$USE_UV" == "1" ]]; then
  BUILD_COMMAND=${BUILD_COMMAND:-"uv pip install -e ."}
else
  BUILD_COMMAND=${BUILD_COMMAND:-"pip install -e ."}
fi
PER_COMMIT_LOG=${PER_COMMIT_LOG:-1}  # Set to 0 to disable per-commit log files

# ============ Validation ============
if [ -z "$TRITON_DIR" ]; then
  # Try to auto-detect if we're in a triton repo
  if [ -d ".git" ] && [ -f "pyproject.toml" ]; then
    TRITON_DIR=$(pwd)
  else
    echo "ERROR: TRITON_DIR is not set and cannot be auto-detected"
    echo "Run 'bash bisect_triton.sh --help' for usage information"
    exit 128
  fi
fi

if [ -z "$TEST_SCRIPT" ]; then
  echo "ERROR: TEST_SCRIPT is not set"
  echo "Run 'bash bisect_triton.sh --help' for usage information"
  exit 128
fi

if [ ! -d "$TRITON_DIR" ]; then
  echo "ERROR: TRITON_DIR not found: $TRITON_DIR"
  exit 128
fi

if [ ! -f "$TEST_SCRIPT" ]; then
  echo "ERROR: TEST_SCRIPT not found: $TEST_SCRIPT"
  exit 128
fi

# Convert all path variables to absolute paths to avoid issues after cd
TRITON_DIR=$(realpath "$TRITON_DIR")
TEST_SCRIPT=$(realpath "$TEST_SCRIPT")
CONDA_DIR=$(realpath "$CONDA_DIR")

# Create log directory and get absolute path
mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")

# ============ Setup ============
cd "$TRITON_DIR" || exit 128

# Get current commit info
COMMIT_HASH=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short=9 HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create per-commit log file (optional, controlled by PER_COMMIT_LOG)
COMMIT_LOG=""
if [ "$PER_COMMIT_LOG" = "1" ]; then
  COMMIT_LOG="$LOG_DIR/${TIMESTAMP}_bisect_triton_${SHORT_COMMIT}.log"
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
  echo "=== Triton Bisect Run ==="
  echo "Timestamp: $(date)"
  echo "Commit: $COMMIT_HASH"
  echo "Short: $SHORT_COMMIT"
  echo "Triton Dir: $TRITON_DIR"
  echo "Test Script: $TEST_SCRIPT"
  echo "Test Args: $TEST_ARGS"
  echo "Conda Env: $CONDA_ENV"
  echo "========================="
  echo ""
} | log_output

# Update git submodules to match the current commit
echo "Updating git submodules..." | log_output
git submodule update --init --recursive 2>&1 | log_output
echo "" | log_output

# Activate conda or uv (if enabled)
source ${CONDA_DIR}/bin/activate
if [ $? -ne 0 ]; then
  echo "ERROR: Cannot activate conda or uv" | log_output
  exit 128
fi

if [ "$USE_UV" == "0" ]; then
  echo "Activating conda environment: $CONDA_ENV" | log_output
  conda activate "$CONDA_ENV"
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment: $CONDA_ENV" | log_output
    exit 128
  fi
fi

echo "" | log_output

# Clean build directory to avoid stale artifacts from previous commits
echo "Cleaning build directory..." | log_output
rm -rf "$TRITON_DIR/build"
rm -rf "$TRITON_DIR/python/triton.egg-info"
echo "" | log_output

# Uninstall any existing triton to avoid conflicts with PyTorch built-in
echo "Uninstalling existing triton packages..." | log_output
if [[ "$USE_UV" == "1" ]]; then
  uv pip uninstall -y triton pytorch-triton 2>&1 | log_output || true
else
  pip uninstall -y triton pytorch-triton 2>&1 | log_output || true
fi
echo "" | log_output

# Build Triton
echo "Building Triton..." | log_output
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

if [ $BUILD_CODE -ne 0 ]; then
  echo "Build FAILED" | log_output
  exit 128
fi

echo "" | log_output

# Run test
echo "Running test..." | log_output
TEST_START=$(date +%s)

if [ -n "$COMMIT_LOG" ]; then
  TRITON_ALWAYS_COMPILE=1 python "$TEST_SCRIPT" $TEST_ARGS 2>&1 | tee -a "$COMMIT_LOG"
  TEST_CODE=${PIPESTATUS[0]}
else
  TRITON_ALWAYS_COMPILE=1 python "$TEST_SCRIPT" $TEST_ARGS 2>&1
  TEST_CODE=$?
fi

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))
echo "Test completed in ${TEST_TIME}s, exit code: $TEST_CODE" | log_output

# Report result
if [ $TEST_CODE -eq 0 ]; then
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
  echo "Test: ${TEST_TIME}s (exit $TEST_CODE)"
  echo "Result: $RESULT"
  if [ -n "$COMMIT_LOG" ]; then
    echo "Log: $COMMIT_LOG"
  fi
  echo "==============="
} | log_output

# Exit with test code for git bisect
exit $TEST_CODE
