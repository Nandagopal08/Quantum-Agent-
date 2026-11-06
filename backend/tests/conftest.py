"""Session fixture: a throw-away development-mode project with small synthetic datasets written in the
exact C-MAPSS and IMS file formats. They exercise the code paths only; reported research results come
from the real NASA data, never from these fixtures."""

from __future__ import annotations

import io
import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("QADT_LLM_PROVIDER", "none")   # deterministic agents in tests


def _cmapss_zip(rng: np.random.Generator) -> bytes:
    def engine_rows(unit: int, life: int, stop: int | None = None):
        rows = []
        for c in range(1, (stop or life) + 1):
            frac = c / life
            sensors = []
            for s in range(1, 22):
                if s in (1, 5, 10, 16, 18, 19):          # constant sensors, as in FD001
                    sensors.append(100.0 + s)
                else:
                    trend = (1 if s % 2 else -1) * 3.0 * np.exp(4 * (frac - 1))
                    sensors.append(500 + 10 * s + trend + rng.normal(0, 0.15))
            rows.append(f"{unit} {c} 0.0 0.0 100.0 " + " ".join(f"{v:.4f}" for v in sensors))
        return rows

    train, test, rul = [], [], []
    for u in range(1, 31):
        train += engine_rows(u, int(rng.integers(130, 210)))
    for u in range(1, 11):
        life = int(rng.integers(130, 210))
        stop = int(rng.integers(40, life - 5))
        test += engine_rows(u, life, stop)
        rul.append(life - stop)
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("train_FD001.txt", "\n".join(train))
        z.writestr("test_FD001.txt", "\n".join(test))
        z.writestr("RUL_FD001.txt", "\n".join(map(str, rul)))
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("Turbofan/CMAPSSData.zip", inner.getvalue())
    return outer.getvalue()


def _ims_zip(rng: np.random.Generator, n_files: int = 80, n: int = 2048) -> bytes:
    out = io.BytesIO()
    t0 = datetime(2004, 2, 12, 10, 32, 39)
    impulse = np.zeros(n)
    impulse[(np.arange(0, n / 20000, 1 / 236.4) * 20000).astype(int)] = 1
    ring = np.convolve(impulse, np.exp(-np.arange(100) / 10) * np.sin(2 * np.pi * 4000 * np.arange(100) / 20000))[:n]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(n_files):
            severity = max(0.0, (i / n_files - 0.6) / 0.4) ** 2          # bearing 1 degrades in the last 40 %
            cols = [rng.normal(0, 0.05, n) + 0.6 * severity * ring] + [rng.normal(0, 0.05, n) for _ in range(3)]
            name = (t0 + timedelta(minutes=10 * i)).strftime("%Y.%m.%d.%H.%M.%S")
            z.writestr(f"IMS/2nd_test/{name}", "\n".join("\t".join(f"{c[k]:.4f}" for c in cols) for k in range(n)))
    return out.getvalue()


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    from backend import config
    from backend.storage import prepare_datasets
    from backend.storage.drive import ensure_project_structure

    root = tmp_path_factory.mktemp("qadt_project")
    config.configure(project_root=root, storage_mode="development", local_scratch=root / "_scratch")
    ensure_project_structure()
    rng = np.random.default_rng(0)
    for name, payload in (("cmapss", _cmapss_zip(rng)), ("ims_bearings", _ims_zip(rng))):
        target = root / "datasets" / "raw" / name / f"{name}.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)       # adopted as a valid local copy → no network access
    reports = prepare_datasets()
    assert all(r["ready"] for r in reports.values()), reports
    yield root
    config.reset_config()
