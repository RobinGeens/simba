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

# Target GPU arch. 9.0 = H100 (wice/gpu_h100). CUDA 13 dropped sm_<8.0, so keep >=8.0.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# Limit parallel nvcc jobs so the build stays light enough for a login node.
export MAX_JOBS="${MAX_JOBS:-4}"

# The venv uses /usr/bin/python3.12, which ships no dev headers. Borrow Python.h from a
# module (C-API is stable across 3.12.x). Override PYTHON_INCLUDE=... if the path changes.
PYTHON_INCLUDE="${PYTHON_INCLUDE:-/apps/leuven/rocky9/skylake/2024a/software/Python/3.12.3-GCCcore-13.3.0/include/python3.12}"
export CPATH="${PYTHON_INCLUDE}:${CPATH:-}"

echo "CUDA_HOME=$CUDA_HOME  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"
"$ROOT/env/bin/pip" install "$ROOT/mamba" --no-build-isolation
