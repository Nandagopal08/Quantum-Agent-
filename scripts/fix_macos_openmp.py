"""macOS only: make PyTorch use Homebrew's libomp so PyTorch and XGBoost/LightGBM share one OpenMP runtime.

Two OpenMP runtimes in one process segfault (exit code 139) or deadlock. Linux and Google Colab do not need this.

    brew install libomp
    python scripts/fix_macos_openmp.py [--venv .venv]
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venv", default=None, help="virtualenv whose torch to patch (default: the running Python)")
    a = ap.parse_args()
    if platform.system() != "Darwin":
        print("Not macOS; nothing to do.")
        return 0
    try:
        prefix = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        print("Homebrew libomp not found. Install it first: brew install libomp")
        return 1
    brew_omp = Path(prefix) / "lib" / "libomp.dylib"
    if not brew_omp.exists():
        print("Install it first: brew install libomp")
        return 1
    python = str(Path(a.venv) / "bin" / "python") if a.venv else sys.executable
    torch_dir = subprocess.run([python, "-c", "import os, importlib.util as u; "
                                "print(os.path.dirname(u.find_spec('torch').origin))"],
                               capture_output=True, text=True, check=True).stdout.strip()
    torch_omp = Path(torch_dir) / "lib" / "libomp.dylib"
    if torch_omp.is_symlink():
        print(f"Already fixed: {torch_omp} -> {torch_omp.resolve()}")
        return 0
    if torch_omp.exists():
        torch_omp.rename(torch_omp.with_suffix(".dylib.bundled"))
    torch_omp.symlink_to(brew_omp)
    print(f"PyTorch now uses {brew_omp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
