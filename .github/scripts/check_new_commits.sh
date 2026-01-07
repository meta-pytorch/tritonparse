#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Check if there are new commits since the last nightly PyPI release.
# This script is used by nightly-pypi.yml to implement on-demand nightly publishing.
#
# Exit code:
#   Always exits with 0 (success). The publish decision is communicated via GITHUB_OUTPUT.
#
# Output:
#   Sets GITHUB_OUTPUT variable: should_publish=true/false
#     - true: New commits found or check skipped due to errors (fail-open behavior)
#     - false: No new commits since last nightly release

set -o pipefail
# Note: set -e is intentionally not used to allow explicit error handling.
# The script implements fail-open behavior where errors should not block publishing.

# Configuration
MAX_RETRIES=3
RETRY_DELAY=5
CURL_TIMEOUT=10
PACKAGE_NAME="${PACKAGE_NAME:-tritonparse}"
PACKAGE_PATH="${PACKAGE_PATH:-}"  # Optional: subdirectory to check (e.g., "python/")

# Dependencies: This script requires 'jq' and 'curl' to be installed.
# These are pre-installed on ubuntu-latest GitHub Actions runners.

# Function: Fetch latest nightly version from PyPI with retry
fetch_latest_nightly() {
    local retries=0

    while [ $retries -lt $MAX_RETRIES ]; do
        local response
        response=$(curl -s --max-time $CURL_TIMEOUT \
            "https://pypi.org/pypi/${PACKAGE_NAME}/json" 2>/dev/null)
        local curl_exit=$?

        if [ $curl_exit -eq 0 ] && [ -n "$response" ]; then
            # Try to parse and extract latest dev version
            local latest
            latest=$(echo "$response" | \
                jq -r '.releases | keys[] | select(contains(".dev"))' 2>/dev/null | \
                sort -V | tail -1)
            local jq_exit=$?

            if [ $jq_exit -eq 0 ] && [ -n "$latest" ]; then
                echo "$latest"
                return 0
            fi
        fi

        retries=$((retries + 1))
        echo "::warning::PyPI API request failed (attempt $retries/$MAX_RETRIES)"

        if [ $retries -lt $MAX_RETRIES ]; then
            echo "Retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        fi
    done

    echo "::warning::All $MAX_RETRIES attempts failed"
    return 1
}

main() {
    echo "Checking for new commits since last nightly release..."

    # Step 1: Fetch latest nightly version (with retry)
    LATEST_NIGHTLY=$(fetch_latest_nightly)
    FETCH_STATUS=$?

    # Step 2: Network request failed -> skip check, proceed with publish
    if [ $FETCH_STATUS -ne 0 ]; then
        echo "::warning::Failed to fetch PyPI data after $MAX_RETRIES attempts"
        echo "::warning::Skipping commit check, proceeding with publish"
        echo "should_publish=true" >> "$GITHUB_OUTPUT"
        exit 0
    fi

    # Step 3: No nightly version exists -> first nightly release
    if [ -z "$LATEST_NIGHTLY" ]; then
        echo "No existing nightly version found, proceeding with publish"
        echo "should_publish=true" >> "$GITHUB_OUTPUT"
        exit 0
    fi

    echo "Latest nightly on PyPI: $LATEST_NIGHTLY"

    # Step 4: Extract timestamp from version (format: X.Y.Z.devYYYYMMDDHHMMSS)
    TIMESTAMP=$(echo "$LATEST_NIGHTLY" | sed -n 's/.*\.dev\([0-9]\{14\}\).*/\1/p')

    if [ -z "$TIMESTAMP" ]; then
        echo "::warning::Cannot parse timestamp from version, proceeding with publish"
        echo "should_publish=true" >> "$GITHUB_OUTPUT"
        exit 0
    fi

    # Step 5: Convert to date format for git log
    SINCE_DATE="${TIMESTAMP:0:4}-${TIMESTAMP:4:2}-${TIMESTAMP:6:2} ${TIMESTAMP:8:2}:${TIMESTAMP:10:2}:${TIMESTAMP:12:2} UTC"
    echo "Last nightly published at: $SINCE_DATE"

    # Step 6: Check for new commits since last nightly
    # If PACKAGE_PATH is set, only check commits in that subdirectory
    if [ -n "$PACKAGE_PATH" ]; then
        echo "Checking commits in path: $PACKAGE_PATH"
        COMMITS_SINCE=$(git log --since="$SINCE_DATE" --oneline -- "$PACKAGE_PATH" | wc -l)
    else
        COMMITS_SINCE=$(git log --since="$SINCE_DATE" --oneline | wc -l)
    fi

    if [ "$COMMITS_SINCE" -eq 0 ]; then
        echo "No new commits since last nightly, skipping publish"
        echo "should_publish=false" >> "$GITHUB_OUTPUT"
        exit 0
    else
        echo "Found $COMMITS_SINCE new commit(s) since last nightly"
        echo "should_publish=true" >> "$GITHUB_OUTPUT"
        exit 0
    fi
}

main
