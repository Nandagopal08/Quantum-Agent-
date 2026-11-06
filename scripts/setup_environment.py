"""Set up the runtime: Python dependencies, system tools, Google Drive, project folders, and a stack check.

Local:
    python scripts/setup_environment.py --storage-mode development --drive-root ./.dev_storage
    python scripts/setup_environment.py --storage-mode desktop --drive-mount ~/Library/CloudStorage/GoogleDrive-<account>

Google Colab (the notebook is used only as a terminal):
    %run scripts/setup_environment.py          # inside the kernel, so it can mount Google Drive itself
    !python scripts/setup_environment.py       # works too once Drive is mounted

Options: --skip-install (no pip/apt), --check-only (report only), --requirements FILE.
This script imports nothing from ``backend`` until the dependencies are installed.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("qadt.setup")
CHECK = ["fastapi", "uvicorn", "pydantic", "sqlalchemy", "numpy", "pandas", "sklearn", "xgboost", "lightgbm",
         "catboost", "torch", "shap", "pennylane", "qiskit", "langchain_core", "langgraph", "py7zr", "pyarrow"]


def in_colab() -> bool:
    try:
        importlib.import_module("google.colab")
        return True
    except ImportError:
        return False


def in_ipython_kernel() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None and hasattr(get_ipython(), "kernel")
    except ImportError:
        return False


def sh(cmd: list[str], check: bool = True) -> int:
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, check=check).returncode


def mount_drive_in_colab() -> None:
    mount = Path(os.environ.get("QADT_DRIVE_MOUNT", "/content/drive"))
    if (mount / "MyDrive").is_dir():
        log.info("Google Drive already mounted at %s", mount)
        return
    if in_ipython_kernel():
        from google.colab import drive

        drive.mount(str(mount))
        return
    raise SystemExit(
        "Google Drive is not mounted. Colab can only mount Drive from the notebook kernel, so either run\n"
        "    %run scripts/setup_environment.py\n"
        "or first run the one-line Colab command\n"
        "    from google.colab import drive; drive.mount('/content/drive')\n"
        "and then re-run this script.")


def install(requirements: Path, colab: bool) -> None:
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])
    if colab or platform.system() == "Linux":
        if not shutil.which("unrar") and shutil.which("apt-get"):
            sh(["apt-get", "-qq", "install", "-y", "unrar"], check=False)     # IMS archive contains .rar files
    elif platform.system() == "Darwin":
        if not shutil.which("unrar"):
            log.warning("unrar not found; IMS extraction will try py7zr/bsdtar. Optional: brew install rar")
        sh([sys.executable, str(ROOT / "scripts" / "fix_macos_openmp.py")], check=False)   # PyTorch/XGBoost OpenMP clash


def check_stack() -> dict:
    report = {}
    for mod in CHECK:
        try:
            m = importlib.import_module(mod)
            report[mod] = getattr(m, "__version__", "installed")
        except Exception as exc:          # noqa: BLE001 - report every failure
            report[mod] = f"MISSING ({exc.__class__.__name__})"
    for name, value in report.items():
        (log.info if not str(value).startswith("MISSING") else log.error)("  %-15s %s", name, value)
    node = shutil.which("node")
    log.info("  %-15s %s", "node", subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
             if node else "not found (only needed to rebuild the dashboard)")
    dist = ROOT / "frontend" / "dist" / "index.html"
    log.info("  %-15s %s", "dashboard", "built" if dist.exists() else "not built: cd frontend && npm install && npx vite build")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storage-mode", choices=("colab", "desktop", "development"), default=None)
    ap.add_argument("--drive-root", default=None)
    ap.add_argument("--drive-mount", default=None)
    ap.add_argument("--requirements", default=str(ROOT / "requirements.txt"))
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a, _ = ap.parse_known_args()          # tolerate extra args from %run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", force=True)

    colab = in_colab()
    mode = a.storage_mode or os.environ.get("QADT_STORAGE_MODE") or ("colab" if colab else None)
    if mode is None:
        log.error("Not in Colab: choose --storage-mode desktop (Google Drive for desktop) or development "
                  "(local folder, flagged in the UI).")
        return 2
    os.environ["QADT_STORAGE_MODE"] = mode
    for flag, env in (("drive_root", "QADT_DRIVE_ROOT"), ("drive_mount", "QADT_DRIVE_MOUNT")):
        if getattr(a, flag):
            os.environ[env] = str(Path(getattr(a, flag)).expanduser().resolve())
    log.info("Python %s on %s · Colab=%s · storage mode %s", platform.python_version(), platform.system(), colab, mode)

    if colab and mode == "colab" and not a.check_only:
        mount_drive_in_colab()
    if not (a.skip_install or a.check_only):
        install(Path(a.requirements), colab)
    report = check_stack()
    missing = [k for k, v in report.items() if str(v).startswith("MISSING")]
    if a.check_only:
        return 1 if missing else 0
    if missing:
        log.error("missing packages: %s", ", ".join(missing))
        return 1

    sys.path.insert(0, str(ROOT))
    from backend import config
    from backend.storage import DriveNotMountedError
    from backend.workflows import setup_storage_step

    config.reset_config()
    try:
        info = setup_storage_step(mount=mode == "desktop")
    except DriveNotMountedError as exc:
        log.error("%s", exc)
        return 2
    log.info("ready: project storage at %s (Google Drive: %s)", info["root"], info["google_drive"])
    log.info("next: python scripts/run_pipeline.py   (or the individual scripts in scripts/)")
    return 0


if __name__ == "__main__":
    code = main()
    if code and not in_ipython_kernel():
        sys.exit(code)
