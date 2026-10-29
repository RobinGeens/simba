#!/usr/bin/env bash
set -euo pipefail

# Builds mamba_ssm from the ./mamba submodule against the venv's torch (cu130).
# NOTE this installs from submodule ./mamba; it does NOT reset to the submodule commit
# git submodule update --init --recursive
#
# Requirements handled here:
#   * nvcc must match torch's CUDA major version (torch is +cu130 -> CUDA 13).
#   * The system python3.12 has no dev headers, so Python.h is taken from a module.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# CUDA toolkit matching torch (+cu130). Override with CUDA_MODULE=... if needed.
CUDA_MODULE="${CUDA_MODULE:-CUDA/13.0.0}"
module load "$CUDA_MODULE"
export CUDA_HOME="${CUDA_HOME:-$EBROOTCUDA}"

# Target GPU archs: 8.0 = A100, 9.0 = H100 (main.sh runs on both). CUDA 13 dropped sm_<8.0.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

# Always compile this source; otherwise setup.py may download a stock upstream wheel.
export MAMBA_FORCE_BUILD=TRUE

# Limit parallel nvcc jobs so the build stays light enough for a login node.
export MAX_JOBS="${MAX_JOBS:-4}"

# The venv uses /usr/bin/python3.12, which ships no dev headers. Borrow Python.h from a
# module (C-API is stable across 3.12.x). Override PYTHON_INCLUDE=... if the path changes.
PYTHON_INCLUDE="${PYTHON_INCLUDE:-/apps/leuven/rocky9/skylake/2024a/software/Python/3.12.3-GCCcore-13.3.0/include/python3.12}"
export CPATH="${PYTHON_INCLUDE}:${CPATH:-}"

# Clean build, so no object file from an older kernel version is reused.
rm -rf "$ROOT/mamba/build"

echo "CUDA_HOME=$CUDA_HOME  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"
"$ROOT/env/bin/pip" install "$ROOT/mamba" --no-build-isolation --no-deps --force-reinstall --no-index
