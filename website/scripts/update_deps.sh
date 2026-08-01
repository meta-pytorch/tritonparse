#!/usr/bin/env bash

# update-deps.sh
# Usage:
#   ./update-deps.sh        # Updates within semver ranges (minor/patch only)
#   ./update-deps.sh --major # Updates to latest versions (including major)
#
# Iterates through every package.json in the project (excluding node_modules).
# For major updates, uses npm-check-updates to update package.json to latest versions.
# `--peer` makes npm-check-updates honor peer dependency ranges, so it won't propose
# a version that `npm install` would then reject; it prints what it held back and why.

set -euo pipefail

MAJOR_UPDATE=false
if [[ "${1:-}" == "--major" ]]; then
  MAJOR_UPDATE=true
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ "$MAJOR_UPDATE" == true ]]; then
  echo "🚀 Starting MAJOR version update in: $ROOT_DIR"
else
  echo "🚀 Starting dependency update (minor/patch only) in: $ROOT_DIR"
fi

# Note: avoid `mapfile -d ''` so this also works on bash 3.2 (macOS default).
PACKAGE_FILES=()
while IFS= read -r -d '' PACKAGE_FILE; do
  PACKAGE_FILES+=("$PACKAGE_FILE")
done < <(find "$ROOT_DIR" -name package.json -not -path '*/node_modules/*' -print0)

if [[ ${#PACKAGE_FILES[@]} -eq 0 ]]; then
  echo "No package.json files found. Nothing to update."
  exit 0
fi

for PACKAGE_FILE in "${PACKAGE_FILES[@]}"; do
  PACKAGE_DIR=$(dirname "$PACKAGE_FILE")
  echo "----------------------------------------------"

  if [[ "$MAJOR_UPDATE" == true ]]; then
    echo "📦 Updating ALL versions (including major) in: $PACKAGE_DIR"
    pushd "$PACKAGE_DIR" > /dev/null
      npx npm-check-updates -u --peer
      npm install
    popd > /dev/null
  else
    echo "📦 Updating dependencies (minor/patch only) in: $PACKAGE_DIR"
    pushd "$PACKAGE_DIR" > /dev/null
      npm update
    popd > /dev/null
  fi

  echo "✅ Finished: $PACKAGE_DIR"
done

echo "🎉 All dependencies have been updated."
