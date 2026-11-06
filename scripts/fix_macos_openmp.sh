#!/usr/bin/env bash
# macOS only. PyTorch bundles its own libomp while XGBoost/LightGBM use Homebrew's; two OpenMP
# runtimes in one process segfault or deadlock. Point PyTorch at the Homebrew copy.
# Linux / Google Colab do not need this.
set -euo pipefail
[[ "$(uname)" == "Darwin" ]] || { echo "Not macOS; nothing to do."; exit 0; }
VENV="${1:-.venv}"
BREW_OMP="$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib"
[[ -f "$BREW_OMP" ]] || { echo "Install it first: brew install libomp"; exit 1; }
TORCH_OMP="$("$VENV/bin/python" -c 'import os, importlib.util as u; print(os.path.join(os.path.dirname(u.find_spec("torch").origin), "lib", "libomp.dylib"))')"
if [[ -L "$TORCH_OMP" ]]; then echo "Already fixed: $TORCH_OMP -> $(readlink "$TORCH_OMP")"; exit 0; fi
mv "$TORCH_OMP" "$TORCH_OMP.bundled"
ln -s "$BREW_OMP" "$TORCH_OMP"
echo "PyTorch now uses $BREW_OMP"
