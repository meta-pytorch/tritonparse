#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# bisect_llvm.sh - Bisect LLVM commits within a compatible Triton range
#
# IMPORTANT: Run this script from the LLVM repository directory
#
# PREREQUISITE:
#   Before running git bisect, first run `make dev-install-llvm` in the Triton
#   directory to let Triton automatically clone an LLVM repository:
#     $ cd /path/to/triton
#     $ make dev-install-llvm
#   This will create llvm-project at /path/to/triton/.llvm-project
#
# USAGE:
#   Set required environment variables and run git bisect:
#     $ cd /path/to/triton/.llvm-project  # Use the LLVM cloned by Triton
#     $ export TRITON_DIR=/path/to/triton
#     $ export TEST_SCRIPT=/path/to/test.py
#     $ git bisect start
#     $ git checkout [known good LLVM commit]
#     $ git bisect good
#     $ git checkout [known bad LLVM commit]
#     $ git bisect bad
#     $ git bisect run bash /path/to/bisect_llvm.sh
#
# For help: bash bisect_llvm.sh --help

# Help message
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  cat << 'EOF'
LLVM Bisect Script

IMPORTANT: This script must be run from the LLVM repository directory.

Usage:
  cd /path/to/llvm-project
  TRITON_DIR=/path/to/triton TEST_SCRIPT=/path/to/test.py \
    git bisect run bash /path/to/bisect_llvm.sh

Required Environment Variables:
  TRITON_DIR       Triton repository path
  TEST_SCRIPT      Test script path

Optional Environment Variables (with defaults):
  CONDA_ENV        Conda environment name (default: triton_bisect)
  TEST_ARGS        Arguments to test script (default: empty)
  LOG_DIR          Log directory (default: ./bisect_logs in Triton dir)
  COMPAT_MODE      Compatibility check mode (default: 0)
                   - 1: Test exit 0 or 1 = compatible (good), others = incompatible (bad)
                        Use this to find first incompatible LLVM commit
                   - 0: Pass through test exit code (0=good, 1=bad, 125=skip)
                        Use this to find regression within compatible range
  PER_COMMIT_LOG   Write per-commit log files (default: 1, set to 0 to disable)

Example:
  # Minimal usage
  cd llvm-project
  TRITON_DIR=/path/to/triton TEST_SCRIPT=/path/to/test.py \
    git bisect run bash bisect_llvm.sh

  # If you want to save settings, create a wrapper script:
  cat > my_llvm_bisect.sh << 'WRAPPER'
  #!/bin/bash
  export TRITON_DIR=/path/to/triton
  export TEST_SCRIPT=/path/to/test.py
  export CONDA_ENV=my_env
  cd /path/to/llvm-project
  git bisect run bash /path/to/bisect_llvm.sh
  WRAPPER

  # Disable per-commit log files (only keep commands.log)
  PER_COMMIT_LOG=0 \
  TRITON_DIR=/path/to/triton \
  TEST_SCRIPT=/path/to/test.py \
  git bisect run bash bisect_llvm.sh

Exit Codes:
  0   - Good commit (test passed)
  1   - Bad commit (test failed)
  125 - Skip commit (build failed for this specific commit)
  128 - Abort bisect (configuration or environment error)
EOF
  exit 0
fi

# Default values
CONDA_ENV=${CONDA_ENV:-triton_bisect}
TRITON_DIR=${TRITON_DIR:-""}
TEST_SCRIPT=${TEST_SCRIPT:-""}
TEST_ARGS=${TEST_ARGS:-""}
LOG_DIR=${LOG_DIR:-./bisect_logs}
CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
PER_COMMIT_LOG=${PER_COMMIT_LOG:-1}  # Set to 0 to disable per-commit log files

# Validate we're in LLVM repository
if [ ! -d .git ]; then
  echo "ERROR: Not in a git repository. This script must be run from llvm-project directory."
  exit 128
fi

# Try to detect if we're in LLVM repo
if ! git remote -v 2>/dev/null | grep -q "llvm"; then
  echo "WARNING: This doesn't appear to be an LLVM repository (no 'llvm' remote found)"
  echo "Continuing anyway, but make sure you're in the right directory..."
fi

# Validate required variables
if [ -z "$TRITON_DIR" ]; then
  echo "ERROR: TRITON_DIR is not set. Please set it via environment variable"
  echo "Run 'bash bisect_llvm.sh --help' for usage information"
  exit 128
fi

if [ -z "$TEST_SCRIPT" ]; then
  echo "ERROR: TEST_SCRIPT is not set. Please set it via environment variable"
  echo "Run 'bash bisect_llvm.sh --help' for usage information"
  exit 128
fi

if [ ! -f "$TEST_SCRIPT" ]; then
  echo "ERROR: Test script not found: $TEST_SCRIPT"
  exit 128
fi

if [ ! -d "$TRITON_DIR" ]; then
  echo "ERROR: TRITON_DIR not found: $TRITON_DIR"
  exit 128
fi

# Convert all path variables to absolute paths to avoid issues after cd
TEST_SCRIPT=$(realpath "$TEST_SCRIPT")
TRITON_DIR=$(realpath "$TRITON_DIR")
CONDA_DIR=$(realpath "$CONDA_DIR")

# Get LLVM commit information
LLVM_COMMIT=$(git rev-parse HEAD)
SHORT_LLVM=$(git rev-parse --short=9 HEAD)

# Set default log directory if not specified
if [ -z "$LOG_DIR" ]; then
  LOG_DIR="$TRITON_DIR/bisect_logs"
fi

# Create log directory and convert to absolute path
mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create per-commit log file (optional, controlled by PER_COMMIT_LOG)
LOG_FILE=""
if [ "$PER_COMMIT_LOG" = "1" ]; then
  LOG_FILE="$LOG_DIR/${TIMESTAMP}_bisect_llvm_${SHORT_LLVM}.log"
fi

# Helper function for logging output
log_output() {
  if [ -n "$LOG_FILE" ]; then
    tee -a "$LOG_FILE"
  else
    cat
  fi
}

# Start logging
{
  echo "=== LLVM Bisect Run ==="
  echo "Timestamp: $(date)"
  echo "LLVM Commit: $LLVM_COMMIT"
  echo "LLVM Short: $SHORT_LLVM"
  echo "LLVM Dir: $(pwd)"
  echo "Triton Dir: $TRITON_DIR"
  echo "Test Script: $TEST_SCRIPT"
  echo "Test Args: $TEST_ARGS"
  echo "Conda Env: $CONDA_ENV"
  echo "========================"
  echo ""
} | log_output

# Update Triton git submodules (in case Triton has submodules)
echo "Updating Triton git submodules..." | log_output
cd "$TRITON_DIR" || {
  echo "ERROR: Cannot change to TRITON_DIR: $TRITON_DIR" | log_output
  exit 128
}
git submodule update --init --recursive 2>&1 | log_output
echo "" | log_output

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV" | log_output
source ${CONDA_DIR}/bin/activate
if [ $? -ne 0 ]; then
  echo "ERROR: Cannot activate conda" | log_output
  exit 128
fi

conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate conda environment: $CONDA_ENV" | log_output
  exit 128
fi

# Change to Triton directory for building
cd "$TRITON_DIR" || {
  echo "ERROR: Cannot change to TRITON_DIR: $TRITON_DIR" | log_output
  exit 128
}

# Always export LLVM_COMMIT_HASH so the build system knows which commit to use
export LLVM_COMMIT_HASH=$LLVM_COMMIT

# Clean LLVM build directory to avoid CMake cache issues
LLVM_BUILD_DIR="$TRITON_DIR/.llvm-project/build"
if [ -d "$LLVM_BUILD_DIR" ]; then
  echo "Cleaning LLVM build directory: $LLVM_BUILD_DIR" | log_output
  rm -rf "$LLVM_BUILD_DIR"
fi

# ============================================================
# Build Phase 1: Build LLVM
# ============================================================
echo "" | log_output
echo "=== Phase 1: Building LLVM $SHORT_LLVM ===" | log_output
LLVM_BUILD_START=$(date +%s)

if [ -n "$LOG_FILE" ]; then
  LLVM_BUILD_PATH="$LLVM_BUILD_DIR" scripts/build-llvm-project.sh 2>&1 | tee -a "$LOG_FILE"
  LLVM_BUILD_CODE=${PIPESTATUS[0]}
else
  LLVM_BUILD_PATH="$LLVM_BUILD_DIR" scripts/build-llvm-project.sh 2>&1
  LLVM_BUILD_CODE=$?
fi

LLVM_BUILD_END=$(date +%s)
LLVM_BUILD_TIME=$((LLVM_BUILD_END - LLVM_BUILD_START))
echo "LLVM build completed in ${LLVM_BUILD_TIME}s, exit code: $LLVM_BUILD_CODE" | log_output

if [ $LLVM_BUILD_CODE -ne 0 ]; then
  # LLVM build failure - always skip, regardless of COMPAT_MODE
  # This handles LLVM history bugs (e.g., APFloat.cpp compile error)
  echo "LLVM build failed - skipping this commit" | log_output
  echo "This is likely a known bug in LLVM history" | log_output
  exit 125  # skip - let bisect try adjacent commits
fi

# ============================================================
# Build Phase 2: Build Triton with LLVM
# ============================================================
echo "" | log_output
echo "=== Phase 2: Building Triton with LLVM ===" | log_output
TRITON_BUILD_START=$(date +%s)

if [ -n "$LOG_FILE" ]; then
  TRITON_BUILD_WITH_CLANG_LLD=1 \
  TRITON_BUILD_WITH_CCACHE=0 \
  LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include" \
  LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib" \
  LLVM_SYSPATH="$LLVM_BUILD_DIR" \
  make dev-install 2>&1 | tee -a "$LOG_FILE"
  TRITON_BUILD_CODE=${PIPESTATUS[0]}
else
  TRITON_BUILD_WITH_CLANG_LLD=1 \
  TRITON_BUILD_WITH_CCACHE=0 \
  LLVM_INCLUDE_DIRS="$LLVM_BUILD_DIR/include" \
  LLVM_LIBRARY_DIR="$LLVM_BUILD_DIR/lib" \
  LLVM_SYSPATH="$LLVM_BUILD_DIR" \
  make dev-install 2>&1
  TRITON_BUILD_CODE=$?
fi

TRITON_BUILD_END=$(date +%s)
TRITON_BUILD_TIME=$((TRITON_BUILD_END - TRITON_BUILD_START))
echo "Triton build completed in ${TRITON_BUILD_TIME}s, exit code: $TRITON_BUILD_CODE" | log_output

# Calculate total build time
BUILD_TIME=$((LLVM_BUILD_TIME + TRITON_BUILD_TIME))
BUILD_CODE=$TRITON_BUILD_CODE

if [ $TRITON_BUILD_CODE -ne 0 ]; then
  echo "Triton build failed" | log_output
  if [ "$COMPAT_MODE" = "1" ]; then
    # COMPAT_MODE: Finding first incompatible LLVM commit
    echo "COMPAT_MODE: Triton build failed - INCOMPATIBLE" | log_output
    exit 1   # bad - this LLVM is incompatible with Triton
  else
    # Normal bisect mode: Triton should not fail to build
    # (assuming good/bad commits were validated before starting bisect)
    echo "ERROR: Triton build failed in bisect mode" | log_output
    echo "This is unexpected - Triton should build with all LLVM commits in the bisect range" | log_output
    echo "Please verify:" | log_output
    echo "  1. Your good/bad commits were validated before starting bisect" | log_output
    echo "  2. The LLVM commit range in commits.csv is correct" | log_output
    exit 128  # abort bisect - Triton build failure is unexpected
  fi
fi

# Run test with TRITON_ALWAYS_COMPILE
echo "" | log_output
echo "Running test..." | log_output
TEST_START=$(date +%s)

if [ -n "$LOG_FILE" ]; then
  TRITON_ALWAYS_COMPILE=1 python "$TEST_SCRIPT" $TEST_ARGS 2>&1 | tee -a "$LOG_FILE"
  TEST_CODE=${PIPESTATUS[0]}
else
  TRITON_ALWAYS_COMPILE=1 python "$TEST_SCRIPT" $TEST_ARGS 2>&1
  TEST_CODE=$?
fi

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))
echo "Test completed in ${TEST_TIME}s, exit code: $TEST_CODE" | log_output

# Handle compatibility mode
COMPAT_MODE=${COMPAT_MODE:-0}

if [ "$COMPAT_MODE" = "1" ]; then
  # Compatibility check mode: normal exit (0 or 1) = compatible
  if [ $TEST_CODE -eq 0 ] || [ $TEST_CODE -eq 1 ]; then
    RESULT="COMPATIBLE"
    FINAL_EXIT=0
    echo "" | log_output
    echo "COMPAT_MODE: Test exited normally (exit $TEST_CODE) - COMPATIBLE" | log_output
  else
    RESULT="INCOMPATIBLE"
    FINAL_EXIT=1
    echo "" | log_output
    echo "COMPAT_MODE: Test exited abnormally (exit $TEST_CODE) - INCOMPATIBLE" | log_output
  fi
else
  # Normal mode: pass through test exit code
  FINAL_EXIT=$TEST_CODE
  if [ $TEST_CODE -eq 0 ]; then
    RESULT="GOOD"
  elif [ $TEST_CODE -eq 1 ]; then
    RESULT="BAD"
  else
    RESULT="SKIP"
  fi
fi

# Summary
{
  echo ""
  echo "=== Summary ==="
  echo "LLVM Commit: $SHORT_LLVM"
  echo "Build: ${BUILD_TIME}s (exit $BUILD_CODE)"
  echo "Test: ${TEST_TIME}s (exit $TEST_CODE)"
  echo "Mode: $([ "$COMPAT_MODE" = "1" ] && echo 'COMPAT_MODE' || echo 'NORMAL')"
  echo "Result: $RESULT"
  if [ -n "$LOG_FILE" ]; then
    echo "Log: $LOG_FILE"
  fi
  echo "==============="
} | log_output

exit $FINAL_EXIT
