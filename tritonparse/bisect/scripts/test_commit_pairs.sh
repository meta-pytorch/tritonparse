#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# test_commit_pairs.sh - Test (Triton, LLVM) commit pairs sequentially
#
# This script tests pairs of (Triton commit, LLVM commit) from a CSV file
# to find the first failing pair.
#
# USAGE:
#   TRITON_DIR=/path/to/triton \
#   TEST_SCRIPT=/path/to/test.py \
#   COMMITS_CSV=commits.csv \
#   bash test_commit_pairs.sh
#
# For help: bash test_commit_pairs.sh --help

# Help message
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  cat << 'EOF'
Sequential (Triton, LLVM) Commit Pairs Testing Script

This script tests pairs of (Triton commit, LLVM commit) sequentially
to find the first pair that causes a test failure.

Usage:
  TRITON_DIR=/path/to/triton \
  TEST_SCRIPT=/path/to/test.py \
  COMMITS_CSV=commits.csv \
  bash test_commit_pairs.sh

Required Environment Variables:
  TRITON_DIR          Path to triton repository
  TEST_SCRIPT         Path to test script
  COMMITS_CSV         CSV file containing (triton_commit,llvm_commit) pairs

Optional Environment Variables (with defaults):
  CONDA_ENV           Conda environment name (default: triton_bisect)
  CONDA_DIR           Conda directory (default: $HOME/miniconda3)
  LOG_DIR             Log directory (default: ./bisect_logs)
  TEST_ARGS           Arguments for test script (default: empty)
  BUILD_COMMAND       Custom build command template (default: LLVM_COMMIT_HASH={LLVM_COMMIT} make dev-install-llvm)
                      Use {TRITON_COMMIT} and {LLVM_COMMIT} as placeholders

CSV File Format:
  Two columns separated by comma: triton_commit,llvm_commit
  Optional header row (auto-detected and skipped)
  Empty lines are ignored
  Comment lines starting with # are ignored

  Example with header:
    triton_commit,llvm_commit
    abc123def456,def456789abc
    xyz789abc123,uvw012def345

  Example without header:
    abc123def456,def456789abc
    xyz789abc123,uvw012def345

  Example with comments:
    # Known good commits
    triton_commit,llvm_commit
    abc123def456,def456789abc
    # This pair is suspected problematic
    xyz789abc123,uvw012def345

Behavior:
  - Tests commit pairs in the order specified in the CSV
  - Shows progress (e.g., "Testing pair 5 of 20")
  - Checks out Triton to specified commit
  - LLVM checkout is handled automatically by Triton build script
  - Builds Triton with the paired LLVM commit (via LLVM_COMMIT_HASH)
  - If build fails: STOPS immediately and exits with code 1
  - If test fails: Records as first bad pair and exits with code 0
  - If all pass: Exits with code 0

Exit Codes:
  0 - Success (found first failing pair, or all pairs passed)
  1 - Error (build failed, validation error, or cannot checkout commit)

Output Files:
  - Unified log file: bisect_logs/bisect_commit_pairs_TIMESTAMP.log
  - Result file (if bad pair found): bisect_logs/bisect_first_bad_pair_TIMESTAMP.txt

Example:
  # Create CSV file
  cat > commits.csv << 'CSV'
  triton_commit,llvm_commit
  7416ffcb9,abc123def
  92982c604,def456ghi
  CSV

  # Run test
  TRITON_DIR=/path/to/triton \
  TEST_SCRIPT=/path/to/test.py \
  COMMITS_CSV=commits.csv \
  bash test_commit_pairs.sh
EOF
  exit 0
fi

# Default values
TRITON_DIR=${TRITON_DIR:-""}
TEST_SCRIPT=${TEST_SCRIPT:-""}
COMMITS_CSV=${COMMITS_CSV:-""}
CONDA_ENV=${CONDA_ENV:-triton_bisect}
CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
LOG_DIR=${LOG_DIR:-./bisect_logs}
TEST_ARGS=${TEST_ARGS:-""}
BUILD_COMMAND=${BUILD_COMMAND:-""}
# LLVM range filter (optional, passed from Python)
FILTER_GOOD_LLVM=${FILTER_GOOD_LLVM:-""}
FILTER_BAD_LLVM=${FILTER_BAD_LLVM:-""}

# ============ Validation ============
echo "========================================"
echo "Sequential Commit Pairs Testing"
echo "========================================"
echo ""

if [ -z "$TRITON_DIR" ]; then
  echo "ERROR: TRITON_DIR is not set. Please set it via environment variable"
  echo "Run 'bash test_commit_pairs.sh --help' for usage information"
  exit 1
fi

if [ -z "$TEST_SCRIPT" ]; then
  echo "ERROR: TEST_SCRIPT is not set. Please set it via environment variable"
  echo "Run 'bash test_commit_pairs.sh --help' for usage information"
  exit 1
fi

if [ -z "$COMMITS_CSV" ]; then
  echo "ERROR: COMMITS_CSV is not set. Please set it via environment variable"
  echo "Run 'bash test_commit_pairs.sh --help' for usage information"
  exit 1
fi

if [ ! -f "$COMMITS_CSV" ]; then
  echo "ERROR: CSV file not found: $COMMITS_CSV"
  exit 1
fi

if [ ! -d "$TRITON_DIR" ]; then
  echo "ERROR: TRITON_DIR not found: $TRITON_DIR"
  exit 1
fi

if [ ! -f "$TEST_SCRIPT" ]; then
  echo "ERROR: TEST_SCRIPT not found: $TEST_SCRIPT"
  exit 1
fi

# Validate that TRITON_DIR is a git repository
cd "$TRITON_DIR" || exit 1
if [ ! -d .git ]; then
  echo "ERROR: TRITON_DIR is not a git repository: $TRITON_DIR"
  exit 1
fi

# ============ Path Conversion ============
TRITON_DIR=$(realpath "$TRITON_DIR")
TEST_SCRIPT=$(realpath "$TEST_SCRIPT")
COMMITS_CSV=$(realpath "$COMMITS_CSV")
CONDA_DIR=$(realpath "$CONDA_DIR")

# Create log directory and convert to absolute path
mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")

# ============ Setup ============
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Use log file path from Python if provided, otherwise generate our own
if [ -n "$PAIR_TEST_LOG_FILE" ]; then
  LOG_FILE="$PAIR_TEST_LOG_FILE"
else
  LOG_FILE="$LOG_DIR/${TIMESTAMP}_bisect.log"
fi
RESULT_FILE="$LOG_DIR/${TIMESTAMP}_pair_test_result.txt"

# Count total commit pairs (skip empty lines and header)
TOTAL_PAIRS=0
while IFS=, read -r triton llvm; do
  [ -z "$triton" ] && continue
  triton=$(echo "$triton" | xargs | sed 's/"//g')
  [ -z "$triton" ] && continue
  # Skip comment lines (starting with #)
  [[ "$triton" == \#* ]] && continue
  # Skip header lines
  [ "$triton" = "triton" ] && continue
  [ "$triton" = "triton_commit" ] && continue
  TOTAL_PAIRS=$((TOTAL_PAIRS + 1))
done < "$COMMITS_CSV"

if [ "$TOTAL_PAIRS" -eq 0 ]; then
  echo "ERROR: No commit pairs found in CSV file: $COMMITS_CSV"
  exit 1
fi

CURRENT=0
FOUND_BAD=false

# ============ Start Logging ============
{
  echo "========================================"
  echo "Sequential Commit Pairs Testing"
  echo "========================================"
  echo "Start Time: $(date)"
  echo "Triton Dir: $TRITON_DIR"
  echo "LLVM: Managed by Triton (.llvm-project/)"
  echo "Test Script: $TEST_SCRIPT"
  echo "Test Args: $TEST_ARGS"
  echo "CSV File: $COMMITS_CSV"
  echo "Total Pairs: $TOTAL_PAIRS"
  echo "Conda Env: $CONDA_ENV"
  echo "Log File: $LOG_FILE"
  echo "========================================"
  echo ""
} | tee "$LOG_FILE"

# ============ Activate Conda ============
echo "Activating conda environment: $CONDA_ENV" | tee -a "$LOG_FILE"
source ${CONDA_DIR}/bin/activate
if [ $? -ne 0 ]; then
  echo "ERROR: Cannot activate conda" | tee -a "$LOG_FILE"
  exit 1
fi

conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate conda environment: $CONDA_ENV" | tee -a "$LOG_FILE"
  exit 1
fi

echo "✅ Conda environment activated" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============ Main Loop ============
# Track if we're within the filter range
IN_RANGE=false
PASSED_END=false

# If no filter specified, we're always in range
if [ -z "$FILTER_GOOD_LLVM" ] && [ -z "$FILTER_BAD_LLVM" ]; then
  IN_RANGE=true
fi

while IFS=, read -r triton_commit llvm_commit || [ -n "$triton_commit" ]; do
  # Skip empty lines
  [ -z "$triton_commit" ] && continue

  # Remove quotes and trim whitespace
  triton_commit=$(echo "$triton_commit" | xargs | sed 's/"//g')
  llvm_commit=$(echo "$llvm_commit" | xargs | sed 's/"//g')

  # Skip if either is empty after processing
  [ -z "$triton_commit" ] && continue
  [ -z "$llvm_commit" ] && continue

  # Skip comment lines (starting with #)
  [[ "$triton_commit" == \#* ]] && continue

  # Skip header lines
  if [ "$triton_commit" = "triton" ] || [ "$triton_commit" = "triton_commit" ]; then
    continue
  fi

  CURRENT=$((CURRENT + 1))
  SHORT_TRITON=$(echo "$triton_commit" | cut -c1-7)
  SHORT_LLVM=$(echo "$llvm_commit" | cut -c1-7)

  # ========== LLVM Range Filter Logic ==========
  # Check if we've reached the start of the range
  if [ "$IN_RANGE" = "false" ] && [ -n "$FILTER_GOOD_LLVM" ]; then
    if [[ "$llvm_commit" == *"$FILTER_GOOD_LLVM"* ]] || [[ "$FILTER_GOOD_LLVM" == *"$llvm_commit"* ]]; then
      IN_RANGE=true
      echo "→ Entering filter range at pair $CURRENT (LLVM: $SHORT_LLVM)" | tee -a "$LOG_FILE"
    fi
  fi

  # Skip pairs outside the filter range
  if [ "$IN_RANGE" = "false" ]; then
    echo "⏭️  Skipping pair $CURRENT (outside filter range, LLVM: $SHORT_LLVM)" | tee -a "$LOG_FILE"
    continue
  fi

  # Check if we've passed the end of the range
  if [ "$PASSED_END" = "true" ]; then
    echo "⏭️  Skipping pair $CURRENT (after filter range, LLVM: $SHORT_LLVM)" | tee -a "$LOG_FILE"
    continue
  fi

  {
    echo ""
    echo "========================================"
    echo "Testing pair $CURRENT of $TOTAL_PAIRS"
    echo "Triton Commit: $triton_commit"
    echo "LLVM Commit: $llvm_commit"
    echo "========================================"
  } | tee -a "$LOG_FILE"

  # ========== 1. Checkout Triton ==========
  echo "" | tee -a "$LOG_FILE"
  echo "Checking out Triton commit: $triton_commit" | tee -a "$LOG_FILE"
  echo "(LLVM $llvm_commit will be checked out automatically by build script)" | tee -a "$LOG_FILE"
  cd "$TRITON_DIR" || {
    echo "❌ ERROR: Cannot cd to $TRITON_DIR" | tee -a "$LOG_FILE"
    exit 1
  }

  git checkout "$triton_commit" 2>&1 | tee -a "$LOG_FILE"
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    {
      echo "❌ ERROR: Failed to checkout Triton $triton_commit"
      echo "Stopping test run."
      echo ""
      echo "=== Final Summary ==="
      echo "Status: Checkout Failed (Triton)"
      echo "Failed at: Pair $CURRENT of $TOTAL_PAIRS"
      echo "Triton Commit: $triton_commit"
      echo "LLVM Commit: $llvm_commit"
      echo "Log file: $LOG_FILE"
      echo "===================="
    } | tee -a "$LOG_FILE"
    exit 1
  fi

  echo "✅ Triton checkout successful" | tee -a "$LOG_FILE"

  # Update git submodules to match the current commit
  echo "" | tee -a "$LOG_FILE"
  echo "Updating git submodules..." | tee -a "$LOG_FILE"
  git submodule update --init --recursive 2>&1 | tee -a "$LOG_FILE"
  echo "✅ Submodules updated" | tee -a "$LOG_FILE"

  # ========== 2. Clean build cache ==========
  cd "$TRITON_DIR" || exit 1

  # Clean build artifacts (can be disabled with CLEAN_BUILD=false)
  CLEAN_BUILD=${CLEAN_BUILD:-true}

  if [ "$CLEAN_BUILD" = "true" ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Cleaning build cache..." | tee -a "$LOG_FILE"

    # Clean Triton build artifacts
    echo "  - Removing Triton build directories..." | tee -a "$LOG_FILE"
    rm -rf build/ dist/ *.egg-info python/build python/dist python/*.egg-info 2>&1 | tee -a "$LOG_FILE" || true

    # Clean LLVM build directory
    echo "  - Removing LLVM build directory..." | tee -a "$LOG_FILE"
    rm -rf .llvm-project/build 2>&1 | tee -a "$LOG_FILE" || true

    # Uninstall old triton from pip
    echo "  - Uninstalling old triton package..." | tee -a "$LOG_FILE"
    pip uninstall -y triton 2>&1 | tee -a "$LOG_FILE" || true

    # Clean Python cache files
    echo "  - Cleaning Python cache files..." | tee -a "$LOG_FILE"
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    echo "✅ Build cache cleaned" | tee -a "$LOG_FILE"
  else
    echo "" | tee -a "$LOG_FILE"
    echo "⚠️  Skipping build cache cleanup (CLEAN_BUILD=false)" | tee -a "$LOG_FILE"
  fi

  # ========== 3. Build Triton with this LLVM ==========
  echo "" | tee -a "$LOG_FILE"
  echo "Building Triton $SHORT_TRITON with LLVM $SHORT_LLVM..." | tee -a "$LOG_FILE"
  echo "(Build script will automatically checkout LLVM)" | tee -a "$LOG_FILE"

  # Set build command
  if [ -z "$BUILD_COMMAND" ]; then
    BUILD_CMD="LLVM_COMMIT_HASH=$llvm_commit make dev-install-llvm"
  else
    # Replace placeholders
    BUILD_CMD="${BUILD_COMMAND//\{TRITON_COMMIT\}/$triton_commit}"
    BUILD_CMD="${BUILD_CMD//\{LLVM_COMMIT\}/$llvm_commit}"
  fi

  echo "Command: $BUILD_CMD" | tee -a "$LOG_FILE"
  BUILD_START=$(date +%s)

  eval "$BUILD_CMD" 2>&1 | tee -a "$LOG_FILE"
  BUILD_CODE=${PIPESTATUS[0]}

  BUILD_END=$(date +%s)
  BUILD_TIME=$((BUILD_END - BUILD_START))
  echo "Build completed in ${BUILD_TIME}s, exit code: $BUILD_CODE" | tee -a "$LOG_FILE"

  if [ $BUILD_CODE -ne 0 ]; then
    {
      echo ""
      echo "❌ BUILD FAILED"
      echo "Triton: $triton_commit"
      echo "LLVM: $llvm_commit"
      echo "Stopping test run."
      echo ""
      echo "=== Final Summary ==="
      echo "Status: Build Failed"
      echo "Failed at: Pair $CURRENT of $TOTAL_PAIRS"
      echo "Triton Commit: $triton_commit"
      echo "LLVM Commit: $llvm_commit"
      echo "Build exit code: $BUILD_CODE"
      echo "Log file: $LOG_FILE"
      echo "===================="
    } | tee -a "$LOG_FILE"

    {
      echo "$triton_commit,$llvm_commit"
      echo "REASON: Build failed (exit code: $BUILD_CODE)"
      echo "Position: Pair $CURRENT of $TOTAL_PAIRS"
    } > "$RESULT_FILE"

    exit 1
  fi

  echo "✅ Build successful" | tee -a "$LOG_FILE"

  # ========== 4. Run test ==========
  echo "" | tee -a "$LOG_FILE"
  echo "Running test..." | tee -a "$LOG_FILE"
  TEST_START=$(date +%s)

  TRITON_ALWAYS_COMPILE=1 python "$TEST_SCRIPT" $TEST_ARGS 2>&1 | tee -a "$LOG_FILE"
  TEST_CODE=${PIPESTATUS[0]}

  TEST_END=$(date +%s)
  TEST_TIME=$((TEST_END - TEST_START))
  echo "Test completed in ${TEST_TIME}s, exit code: $TEST_CODE" | tee -a "$LOG_FILE"

  if [ $TEST_CODE -ne 0 ]; then
    {
      echo ""
      echo "❌ TEST FAILED"
      echo "Triton: $triton_commit"
      echo "LLVM: $llvm_commit"
      echo "This is the FIRST FAILING pair!"
      echo ""
      echo "=== Final Summary ==="
      echo "Status: Test Failed (First Bad Pair Found)"
      echo "Position: Pair $CURRENT of $TOTAL_PAIRS"
      echo "Triton Commit: $triton_commit"
      echo "LLVM Commit: $llvm_commit"
      echo "Build time: ${BUILD_TIME}s"
      echo "Test exit code: $TEST_CODE"
      echo "Result file: $RESULT_FILE"
      echo "Log file: $LOG_FILE"
      echo "===================="
    } | tee -a "$LOG_FILE"

    {
      echo "$triton_commit,$llvm_commit"
      echo "REASON: Test failed (exit code: $TEST_CODE)"
      echo "Position: Pair $CURRENT of $TOTAL_PAIRS"
      echo "Build time: ${BUILD_TIME}s"
      echo "Test time: ${TEST_TIME}s"
    } > "$RESULT_FILE"

    FOUND_BAD=true
    exit 0
  fi

  echo "✅ Test PASSED for this pair" | tee -a "$LOG_FILE"

  # Check if we've reached the end of the filter range
  if [ -n "$FILTER_BAD_LLVM" ]; then
    if [[ "$llvm_commit" == *"$FILTER_BAD_LLVM"* ]] || [[ "$FILTER_BAD_LLVM" == *"$llvm_commit"* ]]; then
      PASSED_END=true
      echo "→ Reached end of filter range at pair $CURRENT (LLVM: $SHORT_LLVM)" | tee -a "$LOG_FILE"
    fi
  fi

done < "$COMMITS_CSV"

# ============ All Pairs Passed ============
{
  echo ""
  echo "========================================"
  echo "✅ All commit pairs tested successfully!"
  echo "========================================"
  echo ""
  echo "=== Final Summary ==="
  echo "Status: All Passed"
  echo "Total pairs tested: $TOTAL_PAIRS"
  echo "No failing pair found"
  echo "Log file: $LOG_FILE"
  echo "End time: $(date)"
  echo "===================="
} | tee -a "$LOG_FILE"

exit 0
