"""Run the backend (pytest) and frontend (vitest) test suites. The backend tests use synthetic fixtures in a
temporary development-mode project, so they never touch Google Drive or the real data.

    python scripts/run_tests.py [--backend-only] [-- -k qseq]
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--backend-only", action="store_true")
ap.add_argument("pytest_args", nargs="*")
a = ap.parse_args()
codes = [subprocess.run([sys.executable, "-m", "pytest", "backend/tests", "-q", *a.pytest_args], cwd=ROOT).returncode]
if not a.backend_only:
    if (ROOT / "frontend" / "node_modules").is_dir():
        codes.append(subprocess.run(["npx", "vitest", "run"], cwd=ROOT / "frontend").returncode)
    else:
        print("frontend tests skipped: run `npm install` in frontend/ first")
print("\nbackend:", "passed" if codes[0] == 0 else "FAILED", *(["· frontend:", "passed" if codes[1] == 0 else "FAILED"] if len(codes) > 1 else []))
sys.exit(max(codes))
