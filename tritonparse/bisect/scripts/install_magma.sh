#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# install_magma.sh - Install MAGMA into CUDA_HOME
#
# Based on pytorch/.ci/docker/common/install_magma.sh but installs into
# CUDA_HOME instead of /usr/local/cuda-*.
#
# Environment Variables:
#   CUDA_HOME   Installation directory (required)

set -eou pipefail

if [ -z "$CUDA_HOME" ]; then
  echo "ERROR: CUDA_HOME is not set"
  exit 1
fi

# Skip installation if MAGMA is already installed
if [ -d "$CUDA_HOME/magma/lib" ] && [ -d "$CUDA_HOME/magma/include" ]; then
  echo "MAGMA already installed at $CUDA_HOME/magma, skipping"
  exit 0
fi

export cuda_version=$1
cuda_version_nodot=${1/./}

MAGMA_VERSION="2.6.1"
magma_archive="magma-cuda${cuda_version_nodot}-${MAGMA_VERSION}-1.tar.bz2"

(
    set -x
    tmp_dir=$(mktemp -d)
    pushd "${tmp_dir}"
    curl -OLs "https://ossci-linux.s3.us-east-1.amazonaws.com/${magma_archive}"
    tar -xvf "${magma_archive}"
    mkdir -p "${CUDA_HOME}/magma"
    mv include "${CUDA_HOME}/magma/include"
    mv lib "${CUDA_HOME}/magma/lib"
    popd
    rm -rf "${tmp_dir}"
)

echo "MAGMA ${MAGMA_VERSION} installed to ${CUDA_HOME}/magma"
