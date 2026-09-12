#!/usr/bin/env bash
set -euo pipefail

if [[ $(id -u) != 0 ]]; then
  echo 'Run with sudo.' >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo NVIDIA_DRIVER_ALREADY_READY
  exit 0
fi

# Do not upgrade the kernel during bootstrap: build for the running kernel.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ca-certificates curl python3 "linux-headers-$(uname -r)"
install -d -m 0755 /opt/google/cuda-installer
curl --fail --show-error --silent --location --retry 3 \
  https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
  -o /opt/google/cuda-installer/cuda_installer.pyz
python3 /opt/google/cuda-installer/cuda_installer.pyz install_driver \
  --installation-mode=binary --installation-branch=lts
echo NVIDIA_DRIVER_INSTALL_FINISHED
