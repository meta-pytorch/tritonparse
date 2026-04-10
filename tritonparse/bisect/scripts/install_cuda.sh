#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# install_cuda.sh - Install CUDA toolkit into a custom directory
#
# Based on pytorch/.ci/docker/common/install_cuda.sh but installs into
# CUDA_HOME instead of /usr/local/cuda.
#
# Environment Variables:
#   CUDA_HOME   Installation directory (required)

set -ex

if [ -z "$CUDA_HOME" ]; then
  echo "ERROR: CUDA_HOME is not set"
  exit 1
fi

# Skip installation if CUDA is already fully installed
if [ -f "$CUDA_HOME/lib64/libcudart.so" ] && \
   [ -f "$CUDA_HOME/include/cudnn.h" ] && \
   [ -f "$CUDA_HOME/lib64/libcudnn.so" ] && \
   [ -f "$CUDA_HOME/lib64/libcusparseLt.so" ]; then
  echo "CUDA already installed at $CUDA_HOME, skipping"
  exit 0
fi

arch_path=''
targetarch=${TARGETARCH:-$(uname -m)}
if [ "${targetarch}" = 'amd64' ] || [ "${targetarch}" = 'x86_64' ]; then
  arch_path='x86_64'
else
  arch_path='sbsa'
fi

NVSHMEM_VERSION=3.4.5

function install_cuda {
  version=$1
  runfile=$2
  # major_minor is exported for use by downstream scripts
  export major_minor=${version%.*}
  rm -rf "${CUDA_HOME}"
  if [[ ${arch_path} == 'sbsa' ]]; then
      runfile="${runfile}_sbsa"
  fi
  runfile="${runfile}.run"
  wget -q "https://developer.download.nvidia.com/compute/cuda/${version}/local_installers/${runfile}" -O "${runfile}"
  chmod +x "${runfile}"
  ./"${runfile}" --toolkit --silent --toolkitpath="${CUDA_HOME}"
  rm -f "${runfile}"
}

function install_cudnn {
  cuda_major_version=$1
  cudnn_version=$2
  mkdir tmp_cudnn && cd tmp_cudnn
  # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
  filepath="cudnn-linux-${arch_path}-${cudnn_version}_cuda${cuda_major_version}-archive"
  wget -q "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-${arch_path}/${filepath}.tar.xz"
  tar xf "${filepath}.tar.xz"
  cp -a "${filepath}/include/"* "${CUDA_HOME}/include/"
  cp -a "${filepath}/lib/"* "${CUDA_HOME}/lib64/"
  cd ..
  rm -rf tmp_cudnn
}

function install_nvshmem {
  cuda_major_version=$1      # e.g. "12"
  nvshmem_version=$2         # e.g. "3.3.9"

  case "${arch_path}" in
    sbsa)
      export dl_arch="aarch64"
      ;;
    x86_64)
      export dl_arch="x64"
      ;;
    *)
      export dl_arch="${arch_path}"
      ;;
  esac

  tmpdir="tmp_nvshmem"
  mkdir -p "${tmpdir}" && cd "${tmpdir}"

  # nvSHMEM license: https://docs.nvidia.com/nvshmem/api/sla.html
  filename="libnvshmem-linux-${arch_path}-${nvshmem_version}_cuda${cuda_major_version}-archive"
  suffix=".tar.xz"
  url="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-${arch_path}/${filename}${suffix}"

  # download, unpack, install
  wget -q "${url}"
  tar xf "${filename}${suffix}"
  cp -a "${filename}/include/"* "${CUDA_HOME}/include/"
  cp -a "${filename}/lib/"*     "${CUDA_HOME}/lib64/"

  # cleanup
  cd ..
  rm -rf "${tmpdir}"

  echo "nvSHMEM ${nvshmem_version} for CUDA ${cuda_major_version} (${arch_path}) installed."
}

function install_124 {
  CUDNN_VERSION=9.1.0.70
  echo "Installing CUDA 12.4.1 and cuDNN ${CUDNN_VERSION}"
  install_cuda 12.4.1 cuda_12.4.1_550.54.15_linux

  install_cudnn 12 $CUDNN_VERSION

  ldconfig
}

function install_126 {
  CUDNN_VERSION=9.10.2.21
  echo "Installing CUDA 12.6.3 and cuDNN ${CUDNN_VERSION} and NVSHMEM"
  install_cuda 12.6.3 cuda_12.6.3_560.35.05_linux

  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  ldconfig
}

function install_129 {
  CUDNN_VERSION=9.20.0.48
  echo "Installing CUDA 12.9.1 and cuDNN ${CUDNN_VERSION} and NVSHMEM"
  install_cuda 12.9.1 cuda_12.9.1_575.57.08_linux

  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  ldconfig
}

function install_128 {
  CUDNN_VERSION=9.20.0.48
  echo "Installing CUDA 12.8.1 and cuDNN ${CUDNN_VERSION} and NVSHMEM"
  install_cuda 12.8.1 cuda_12.8.1_570.124.06_linux

  install_cudnn 12 $CUDNN_VERSION

  install_nvshmem 12 $NVSHMEM_VERSION

  ldconfig
}

function install_130 {
  CUDNN_VERSION=9.20.0.48
  echo "Installing CUDA 13.0 and cuDNN ${CUDNN_VERSION} and NVSHMEM"
  install_cuda 13.0.2 cuda_13.0.2_580.95.05_linux

  install_cudnn 13 $CUDNN_VERSION

  install_nvshmem 13 $NVSHMEM_VERSION

  ldconfig
}

function install_132 {
  CUDNN_VERSION=9.20.0.48
  echo "Installing CUDA 13.2 and cuDNN ${CUDNN_VERSION} and NVSHMEM"
  install_cuda 13.2.0 cuda_13.2.0_595.45.04_linux

  install_cudnn 13 $CUDNN_VERSION

  install_nvshmem 13 $NVSHMEM_VERSION

  ldconfig
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    12.4) install_124;
        ;;
    12.6|12.6.*) install_126;
        ;;
    12.8|12.8.*) install_128;
        ;;
    12.9|12.9.*) install_129;
        ;;
    13.0|13.0.*) install_130;
        ;;
    13.2|13.2.*) install_132;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
