"""Build dist/qadt_code.zip: everything Colab needs (backend, scripts, built dashboard, docs, requirements).

    python scripts/make_colab_bundle.py [--skip-frontend-build] [--output dist/qadt_code.zip]

Upload the ZIP to Google Drive (e.g. MyDrive/qadt_code.zip), then in Colab (see HOW_TO_RUN.md):

    from google.colab import drive; drive.mount('/content/drive')
    !unzip -oq /content/drive/MyDrive/qadt_code.zip -d /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
    %cd /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
    !python scripts/setup_environment.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["backend", "scripts", "documentation", "frontend/dist", "frontend/src", "frontend/package.json",
           "frontend/package-lock.json", "frontend/vite.config.js", "frontend/index.html", "requirements.txt",
           "requirements-lock.txt", "README.md", "HOW_TO_RUN.md", ".env.example"]
SKIP_PARTS = {"__pycache__", "node_modules", ".pytest_cache", ".DS_Store"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-frontend-build", action="store_true")
    ap.add_argument("--output", default=str(ROOT / "dist" / "qadt_code.zip"))
    a = ap.parse_args()
    if not a.skip_frontend_build:
        npm = shutil.which("npm")
        if not npm:
            print("npm not found: install Node.js 18+ or pass --skip-frontend-build", file=sys.stderr)
            return 1
        subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=ROOT / "frontend", check=True,
                       stdout=subprocess.DEVNULL)
        subprocess.run(["npx", "vite", "build"], cwd=ROOT / "frontend", check=True, stdout=subprocess.DEVNULL)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            if not p.exists():
                continue
            for f in [p] if p.is_file() else sorted(p.rglob("*")):
                if f.is_file() and not (SKIP_PARTS & set(f.relative_to(ROOT).parts)) and f.suffix not in (".pyc", ".ipynb"):
                    z.write(f, f.relative_to(ROOT).as_posix())
                    n += 1
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB, {n} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
