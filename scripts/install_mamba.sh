#!/usr/bin/env bash
set -euo pipefail

# NOTE this installs from submodule ./mamba; it does NOT reset to the submodule commit —
# git submodule update --init --recursive
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate venv (sets LD_LIBRARY_PATH for the custom Python + loads CUDA module).
# shellcheck disable=SC1091
source "$ROOT/env/bin/activate"

pip install "$ROOT/mamba" --no-build-isolation
