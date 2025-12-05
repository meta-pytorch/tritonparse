#!/bin/bash
# bisect_triton.sh - Bisect Triton commits to find regressions
#
# USAGE:
#   Set required environment variables and run git bisect:
#     $ git bisect start
#     $ git checkout [known good commit]
#     $ git bisect good
#     $ git checkout [known bad commit]
#     $ git bisect bad
#     $ TEST_SCRIPT=../repro1/triton_only_repro.py TRITON_DIR=./ BUILD_COMMAND="pip install -e ." CONDA_ENV=bisect3 git bisect run bash ../bisect_triton.sh
#
# For help: bash bisect_triton.sh --help

# Help message
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
  cat << 'EOF'
Triton Bisect Script

Usage:
  TEST_SCRIPT=/path/to/test.py git bisect run bash bisect_triton.sh

Required Environment Variables:
  TEST_SCRIPT      Test script path

Optional Environment Variables (with defaults):
  CONDA_ENV        Conda environment name (default: bisect_triton)
  TRITON_DIR       Triton repository path (default: current directory)
  TEST_ARGS        Arguments to test script (default: empty)
  LOG_DIR          Log directory (default: ./bisect_logs)
  BUILD_COMMAND    Build command (default: pip install -e .)

Example:
  # Minimal usage
  TEST_SCRIPT=./test.py git bisect run bash bisect_triton.sh

  # With optional settings
  CONDA_ENV=my_env TRITON_DIR=/path/to/triton TEST_SCRIPT=./test.py \
    git bisect run bash bisect_triton.sh

  # If you want to save settings, create a wrapper script:
  cat > my_bisect.sh << 'WRAPPER'
  #!/bin/bash
  export TRITON_DIR=/path/to/triton
  export TEST_SCRIPT=/path/to/test.py
  export CONDA_ENV=my_env
  git bisect run bash bisect_triton.sh
  WRAPPER

Exit Codes:
  0   - Good commit (test passed)
  1   - Bad commit (test failed)
  125 - Skip commit (build failed for this specific commit)
  128 - Abort bisect (configuration or environment error)
EOF
  exit 0
fi

# Default values
CONDA_ENV=${CONDA_ENV:-bisect_triton}
TRITON_DIR=${TRITON_DIR:-$(pwd)}
TEST_SCRIPT=${TEST_SCRIPT:-""}
TEST_ARGS=${TEST_ARGS:-""}
LOG_DIR=${LOG_DIR:-./bisect_logs}
CONDA_DIR=${CONDA_DIR:-$HOME/miniconda3}
BUILD_COMMAND=${BUILD_COMMAND:-"pip install -e ."}

# Validate required variables
if [ -z "$TEST_SCRIPT" ]; then
  echo "ERROR: TEST_SCRIPT is not set. Please set it via environment variable"
  echo "Run 'bash bisect_triton.sh --help' for usage information"
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

# Create log directory and convert to absolute path
mkdir -p "$LOG_DIR"
LOG_DIR=$(realpath "$LOG_DIR")

# Get current commit information
CURRENT_COMMIT=$(git rev-parse HEAD)
SHORT_COMMIT=$(git rev-parse --short HEAD)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${TIMESTAMP}_${SHORT_COMMIT}.log"

# Start logging
{
  echo "=== Triton Bisect Run ==="
  echo "Timestamp: $(date)"
  echo "Commit: $CURRENT_COMMIT"
  echo "Short: $SHORT_COMMIT"
  echo "Triton Dir: $TRITON_DIR"
  echo "Test Script: $TEST_SCRIPT"
  echo "Test Args: $TEST_ARGS"
  echo "Conda Env: $CONDA_ENV"
  echo "========================="
  echo ""
} | tee "$LOG_FILE"

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV" | tee -a "$LOG_FILE"
source ${CONDA_DIR:-$HOME/miniconda3}/bin/activate
if [ $? -ne 0 ]; then
  echo "ERROR: Cannot activate conda" | tee -a "$LOG_FILE"
  exit 128
fi

conda activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to activate conda environment: $CONDA_ENV" | tee -a "$LOG_FILE"
  exit 128
fi

# Change to Triton directory
cd "$TRITON_DIR" || {
  echo "ERROR: Cannot change to TRITON_DIR: $TRITON_DIR" | tee -a "$LOG_FILE"
  exit 128
}

# Build Triton
echo "" | tee -a "$LOG_FILE"
echo "Building Triton..." | tee -a "$LOG_FILE"
BUILD_START=$(date +%s)

eval "$BUILD_COMMAND" 2>&1 | tee -a "$LOG_FILE"
BUILD_CODE=${PIPESTATUS[0]}  # Get exit code of the build command, not tee

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))
echo "Build completed in ${BUILD_TIME}s, exit code: $BUILD_CODE" | tee -a "$LOG_FILE"

if [ $BUILD_CODE -ne 0 ]; then
  echo "Build failed, skipping this commit" | tee -a "$LOG_FILE"
  exit 125
fi

# Run test
echo "" | tee -a "$LOG_FILE"
echo "Running test..." | tee -a "$LOG_FILE"
TEST_START=$(date +%s)

python "$TEST_SCRIPT" $TEST_ARGS 2>&1 | tee -a "$LOG_FILE"
TEST_CODE=${PIPESTATUS[0]}  # Get exit code of python, not tee

TEST_END=$(date +%s)
TEST_TIME=$((TEST_END - TEST_START))
echo "Test completed in ${TEST_TIME}s, exit code: $TEST_CODE" | tee -a "$LOG_FILE"

# Summary
{
  echo ""
  echo "=== Summary ==="
  echo "Commit: $SHORT_COMMIT"
  echo "Build: ${BUILD_TIME}s (exit $BUILD_CODE)"
  echo "Test: ${TEST_TIME}s (exit $TEST_CODE)"
  echo "Result: $([ $TEST_CODE -eq 0 ] && echo 'GOOD' || echo 'BAD')"
  echo "Log: $LOG_FILE"
  echo "==============="
} | tee -a "$LOG_FILE"

exit $TEST_CODE
