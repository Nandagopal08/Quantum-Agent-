"""Validate Google Drive persistence for real (run it in Colab, or with Google Drive for desktop).

Two phases, so that a *real* runtime restart can be tested:

    # Colab, session 1
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
    !python scripts/validate_colab_drive.py --phase write
    #  → Runtime ▸ Disconnect and delete runtime, reconnect (a brand-new VM; /content is wiped) …
    # Colab, session 2
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Quantum_Agentic_Digital_Twin_code
    !python scripts/validate_colab_drive.py --phase verify

``--phase all`` runs both in one go; the "restart" is then simulated with fresh Python subprocesses.

Checks: storage mode is Google Drive (not development), Drive mounted, project folders created, JSON / binary
round-trips with SHA-256, atomic overwrite, report history on overwrite, training-log append, checkpoint
save → resume, model save → load → identical outputs in a new process, datasets present without re-download
(download dates unchanged), and — in phase verify — every artefact from phase write still present and intact.
Writes ``reports/drive_validation_<phase>_<time>.json`` in the project folder and exits non-zero on failure.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from _cli import ROOT, log, parser, run

ap = parser(__doc__)
ap.add_argument("--phase", choices=("write", "verify", "all"), default="all")
ap.add_argument("--allow-development", action="store_true",
                help="permit a development-mode run (checks the code path only; NOT a Drive validation)")
ap.add_argument("--skip-datasets", action="store_true", help="skip the dataset re-download check")
a = ap.parse_args()

MARKER_DIR = "reports/_drive_validation"
CHILD_LOAD = """
import json, sys, torch, numpy as np
sys.path.insert(0, {root!r})
from backend.storage.artifacts import load_model
from backend.quantum_models.qseq import model_from_state
b = load_model("quantum_ai", "drive_validation_probe")
m = model_from_state(b["config"], b["state"])
x = torch.as_tensor(np.asarray(b["probe_input"]))
with torch.no_grad():
    print(json.dumps(m(x).numpy().tolist()))
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from backend.config import get_config
    from backend.storage import paths
    from backend.storage.drive import is_drive_mounted

    cfg = get_config()
    results: list[dict] = []

    def check(name: str, fn):
        t0 = time.time()
        try:
            detail = fn()
            results.append({"check": name, "ok": True, "detail": detail, "seconds": round(time.time() - t0, 2)})
            log.info("PASS  %-42s %s", name, detail if isinstance(detail, str) else "")
        except Exception as exc:          # record every failure and continue
            results.append({"check": name, "ok": False, "error": f"{exc.__class__.__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=3)})
            log.error("FAIL  %-42s %s", name, exc)

    def is_drive():
        if not cfg.is_google_drive:
            if a.allow_development:
                return "development mode — code path only, this is NOT a Google Drive validation"
            raise RuntimeError("storage mode is 'development'; run in Colab or with --storage-mode desktop")
        if not is_drive_mounted(cfg):
            raise RuntimeError(f"no Google Drive mounted at {cfg.drive_mount}")
        return f"{cfg.storage_mode} at {cfg.drive_mount}"

    check("storage is Google Drive and mounted", is_drive)
    marker = paths.root / MARKER_DIR
    state_file = marker / "phase_write.json"

    if a.phase in ("write", "all"):
        from backend.config import PROJECT_DIRS
        from backend.storage import CheckpointManager, ExperimentLogger, save_model, save_result
        from backend.storage.fileio import read_json, write_json

        written: dict[str, str] = {}

        def folders():
            missing = [d for d in PROJECT_DIRS if not (paths.root / d).is_dir()]
            if missing:
                raise RuntimeError(f"missing folders: {missing}")
            return f"{len(PROJECT_DIRS)} project folders present"

        def json_roundtrip():
            p = marker / "probe.json"
            data = {"time": time.time(), "values": list(range(1000))}
            write_json(p, data)
            if read_json(p) != data:
                raise RuntimeError("JSON read-back differs")
            written[str(p)] = sha(p)
            return "ok"

        def binary_roundtrip():
            import os

            p = marker / "probe.bin"
            blob = os.urandom(5 << 20)
            from backend.storage.fileio import atomic_write

            atomic_write(p, lambda tmp: Path(tmp).write_bytes(blob))
            if p.read_bytes() != blob:
                raise RuntimeError("5 MB binary read-back differs")
            atomic_write(p, lambda tmp: Path(tmp).write_bytes(blob[::-1]))      # atomic overwrite
            if p.read_bytes() != blob[::-1]:
                raise RuntimeError("atomic overwrite failed")
            written[str(p)] = sha(p)
            return "5 MB written, verified, atomically overwritten"

        def history():
            save_result("comparison_reports", "drive_validation_probe", {"v": 1})
            save_result("comparison_reports", "drive_validation_probe", {"v": 2})
            h = paths.area("comparison_reports") / "_history" / "drive_validation_probe"
            if not any(h.glob("*.json")):
                raise RuntimeError("previous version was not archived")
            return "previous version archived"

        def logs():
            lg = ExperimentLogger("drive_validation_probe")
            n0 = len(lg.records())
            lg.log(step=1)
            lg.log(step=2)
            if len(lg.records()) != n0 + 2:
                raise RuntimeError("log append not persisted")
            written[str(lg.path)] = sha(lg.path)
            return f"{n0 + 2} records"

        def checkpoint():
            cm = CheckpointManager("drive_validation_probe", category="quantum_ai", keep_last=2)
            for step in (1, 2, 3):
                cm.save({"step": step, "w": [step] * 10}, step)
            state, entry = CheckpointManager("drive_validation_probe", category="quantum_ai").load_latest()
            if entry is None or state["step"] != 3:
                raise RuntimeError(f"resume returned {entry}")
            return f"resumed from step {entry['step']}; kept {len(cm.list())}"

        def model_restart():
            import numpy as np
            import torch

            from backend.quantum_models.qseq import SeqModelConfig, build_model, state_to_numpy

            m = build_model(SeqModelConfig(name="q_lstm", n_features=3, window=6), seed=0)
            m.eval()
            x = np.random.default_rng(0).normal(size=(2, 6, 3))
            with torch.no_grad():
                ref = m(torch.as_tensor(x)).numpy().tolist()
            save_model({"config": m.cfg.to_dict(), "state": state_to_numpy(m), "probe_input": x, "probe_output": ref},
                        "quantum_ai", "drive_validation_probe")
            out = subprocess.run([sys.executable, "-c", CHILD_LOAD.format(root=str(ROOT))], capture_output=True,
                                 text=True, timeout=600)
            if out.returncode != 0:
                raise RuntimeError(out.stderr[-800:])
            got = json.loads(out.stdout.strip().splitlines()[-1])
            if not np.allclose(got, ref, atol=1e-12):
                raise RuntimeError("reloaded model gives different outputs")
            return "quantum model reloaded in a new process; identical outputs"

        def datasets():
            if a.skip_datasets:
                return "skipped"
            from backend.storage import load_metadata, prepare_datasets

            before = {k: v.get("download_date") for k, v in load_metadata().items()}
            if not before:
                raise RuntimeError("no datasets recorded yet; run scripts/download_datasets.py first")
            t0 = time.time()
            prepare_datasets()
            after = {k: v.get("download_date") for k, v in load_metadata().items()}
            if before != after:
                raise RuntimeError(f"datasets were downloaded again: {before} → {after}")
            return f"no re-download ({time.time() - t0:.0f} s verification)"

        for name, fn in (("project folders", folders), ("JSON round-trip", json_roundtrip),
                         ("binary round-trip + atomic overwrite", binary_roundtrip),
                         ("report history on overwrite", history), ("training-log persistence", logs),
                         ("checkpoint save → resume", checkpoint), ("model reload in a new process", model_restart),
                         ("datasets persist without re-download", datasets)):
            check(name, fn)
        write_json(state_file, {"written": written, "time": time.time(), "results": results})

    if a.phase in ("verify", "all"):
        from backend.storage.fileio import read_json

        def survived():
            st = read_json(state_file)
            if not st:
                raise RuntimeError(f"{state_file} missing: run --phase write first (before the restart)")
            bad = [p for p, h in st["written"].items() if not Path(p).is_file() or sha(Path(p)) != h]
            if bad:
                raise RuntimeError(f"changed or missing after restart: {bad}")
            age = (time.time() - st["time"]) / 60
            return f"{len(st['written'])} files intact, written {age:.0f} min ago"

        def model_after_restart():
            import numpy as np

            out = subprocess.run([sys.executable, "-c", CHILD_LOAD.format(root=str(ROOT))], capture_output=True,
                                 text=True, timeout=600)
            if out.returncode != 0:
                raise RuntimeError(out.stderr[-800:])
            from backend.storage.artifacts import load_model

            ref = load_model("quantum_ai", "drive_validation_probe")["probe_output"]
            if not np.allclose(json.loads(out.stdout.strip().splitlines()[-1]), ref, atol=1e-12):
                raise RuntimeError("model outputs differ after restart")
            return "identical outputs"

        def checkpoint_after_restart():
            from backend.storage import CheckpointManager

            state, entry = CheckpointManager("drive_validation_probe", category="quantum_ai").load_latest()
            if entry is None or state["step"] != 3:
                raise RuntimeError("checkpoint not resumable")
            return f"resumable from step {entry['step']}"

        check("artefacts survive restart", survived)
        check("model loads after restart", model_after_restart)
        check("checkpoint resumes after restart", checkpoint_after_restart)

    ok = all(r["ok"] for r in results)
    from backend.storage.fileio import write_json

    report = paths.root / "reports" / f"drive_validation_{a.phase}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_json(report, {"phase": a.phase, "storage_mode": cfg.storage_mode, "project_root": str(cfg.project_root),
                        "passed": ok, "results": results})
    log.info("%s — %d/%d checks passed · report %s", "PASSED" if ok else "FAILED",
             sum(r["ok"] for r in results), len(results), report)
    if not ok:
        raise RuntimeError("Drive validation failed")
    return {"passed": ok, "report": str(report)}


sys.exit(run("validate_colab_drive", a, main))
