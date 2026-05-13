#!/usr/bin/env bash
set -euo pipefail

# NOTE this installs from submodule ./mamba; it does NOT reset to the submodule commit —
# git submodule update --init --recursive
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/env/bin/pip" install "$ROOT/mamba" --no-build-isolation
