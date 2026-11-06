"""Reusable workflow steps behind the command-line scripts in ``scripts/``.

Each function performs one major stage end to end, writes its outputs to the project storage (Google
Drive in ``colab``/``desktop`` mode) and returns a JSON-serialisable summary. The scripts parse arguments
and call these; ``scripts/run_pipeline.py`` chains them with a resumable step log in
``configuration/pipeline_state.json``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger("qadt.workflow")
ROOT = Path(__file__).resolve().parents[1]
Progress = Callable[[float, str], None]


def _progress(prefix: str, every: float = 15.0) -> Progress:
    last = [0.0]

    def report(fraction: float, message: str) -> None:
        if fraction >= 1.0 or time.time() - last[0] >= every or "epoch 1/" in message:
            last[0] = time.time()
            log.info("%s %5.1f%%  %s", prefix, 100 * fraction, message)
    return report


# ---------------------------------------------------------------- environment


KEY_PACKAGES = ("torch", "pennylane", "qiskit", "fastapi", "uvicorn", "pydantic", "langchain-core", "langgraph",
                "numpy", "pandas", "scipy", "scikit-learn", "shap", "xgboost", "lightgbm", "catboost", "pyarrow")


def check_environment() -> dict:
    """Package versions, dependency consistency, storage mode and the Python-only rule (no notebooks)."""
    from importlib.metadata import PackageNotFoundError, version

    from .config import get_config

    versions, missing = {}, []
    for pkg in KEY_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            missing.append(pkg)
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    notebooks = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.ipynb")
                 if not {".venv", "node_modules", "backups"} & set(p.relative_to(ROOT).parts)]
    cfg = get_config()
    out = {"python": sys.version.split()[0], "versions": versions, "missing": missing,
           "pip_check": pip.stdout.strip() or pip.stderr.strip(), "pip_ok": pip.returncode == 0,
           "storage_mode": cfg.storage_mode, "google_drive": cfg.is_google_drive, "notebooks": notebooks}
    log.info("Python %s · %d/%d key packages · pip check %s · storage %s · notebooks: %d", out["python"],
             len(versions), len(KEY_PACKAGES), "ok" if out["pip_ok"] else "FAILED", cfg.storage_mode, len(notebooks))
    if not cfg.is_google_drive:
        log.warning("storage mode is 'development': results are NOT persisted to Google Drive")
    if missing or not out["pip_ok"] or notebooks:
        raise RuntimeError(f"environment check failed: missing={missing} pip_ok={out['pip_ok']} notebooks={notebooks}")
    return out


def verify_results(subset: str = "FD001") -> dict:
    """Run scripts/verify_results.py (independent recomputation of the benchmark numbers)."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_results.py"), "--subset", subset], cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"result verification failed (exit {r.returncode}); see reports/results_verification_{subset}.json")
    return {"verified": True}


# ---------------------------------------------------------------- storage & data


def setup_storage_step(mount: bool = True) -> dict:
    from .config import get_config
    from .storage import paths
    from .storage.drive import ensure_project_structure, in_colab, mount_drive

    cfg = get_config()
    if mount and cfg.storage_mode in ("colab", "desktop"):
        mount_drive(config=cfg)
    root = ensure_project_structure()
    log.info("project storage ready at %s (mode=%s, colab=%s)", root, cfg.storage_mode, in_colab())
    return {"root": str(paths.root), "storage_mode": cfg.storage_mode, "google_drive": cfg.is_google_drive}


def download_datasets(names: Iterable[str] | None = None, force: bool = False) -> dict:
    from .storage import prepare_datasets

    reports = prepare_datasets(list(names) if names else None, force=force)
    for name, r in reports.items():
        log.info("%-13s ready=%s files=%s %s", name, r["ready"], r.get("file_count"), "; ".join(r.get("problems", [])))
    bad = [n for n, r in reports.items() if not r["ready"]]
    if bad:
        raise RuntimeError(f"datasets not ready: {bad}")
    return {n: {"ready": r["ready"], "file_count": r.get("file_count")} for n, r in reports.items()}


def preprocess_datasets(subsets: Iterable[str] = ("FD001",), ims_tests: Iterable[str] | None = None) -> dict:
    from .datasets import cmapss, ims

    out: dict[str, Any] = {"cmapss": {}, "ims": {}}
    for s in subsets:
        d = cmapss.prepare(cmapss.CmapssConfig(subset=s))
        cmapss.eda(s)
        out["cmapss"][s] = {"train_units": int(d.train["unit"].nunique()), "val_units": int(d.val["unit"].nunique()),
                            "test_units": int(d.test["unit"].nunique()), "features": len(d.feature_cols),
                            "dropped_sensors": d.dropped_sensors}
        log.info("C-MAPSS %s: %s", s, out["cmapss"][s])
    for t in (list(ims_tests) if ims_tests is not None else ims.available_tests()):
        t0 = time.time()
        ims.extract_features(t)        # cached in Drive; re-runs are cheap
        ims.eda(t)
        out["ims"][t] = {"seconds": round(time.time() - t0, 1)}
        log.info("IMS %s features ready (%.0f s)", t, time.time() - t0)
    return out


# ---------------------------------------------------------------- classical models


def train_rul_suite(subset: str = "FD001", models: Iterable[str] | None = None, epochs: int = 40, seed: int = 42) -> dict:
    from .classical_models.rul import ALL_MODELS, train_rul_model
    from .storage import save_result

    rows = []
    for m in models or ALL_MODELS:
        r = train_rul_model(m, subset, epochs=epochs, seed=seed, progress=_progress(f"[rul {m}]"))
        t, c = r["test"], r["failure_classification"]
        rows.append({"model": m, **{k: t[k] for k in ("rmse", "mae", "r2", "nasa_score")}, "pof_roc_auc": c["roc_auc"],
                     "pof_f1": c["f1"], "interval_coverage_90": r["interval_coverage_90"],
                     "train_seconds": r["train_seconds"]})
        log.info("RUL %s: RMSE %.2f MAE %.2f", m, t["rmse"], t["mae"])
    save_result("comparison_reports", f"rul_{subset}_comparison",
                {"subset": subset, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows})
    return {"subset": subset, "rows": rows}


def train_anomaly_suite(tests: Iterable[str] = ("2nd_test", "1st_test", "3rd_test"), epochs: int = 60) -> dict:
    from .classical_models.anomaly import DETECTORS, train_detector, train_supervised_cross_test
    from .datasets import ims
    from .storage import save_result

    tests = list(tests)
    keys = ("roc_auc", "precision", "recall", "f1", "accuracy", "false_alarm_rate")
    rows = []
    for t in tests:
        ims.extract_features(t)
        ims.eda(t)
        for d in DETECTORS:
            r = train_detector(d, t, epochs=epochs, progress=_progress(f"[{d} {t}]"))
            m = r["metrics"]
            rows.append({"test": t, "model": d, **{k: m[k] for k in keys}, "train_seconds": r["train_seconds"],
                         "lead_time_hours": {b: v["lead_time_hours"] for b, v in r["detection"].items()}})
            log.info("IMS %s %s: ROC-AUC %.3f F1 %.3f", t, d, m["roc_auc"] or float("nan"), m["f1"])
    if len(tests) >= 3:
        for ev in tests:
            r = train_supervised_cross_test([t for t in tests if t != ev], ev)
            m = r["metrics"]
            rows.append({"test": ev, "model": "random_forest_supervised", **{k: m[k] for k in keys},
                         "lead_time_hours": {b: v["lead_time_hours"] for b, v in r["detection"].items()},
                         "train_tests": r["train_tests"]})
    save_result("comparison_reports", "ims_anomaly_comparison", {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows,
        "labelling": "pseudo-labels: first 25% healthy, last 10% of failed bearings anomalous (see backend/datasets/ims.py)"})
    return {"rows": rows}


# ---------------------------------------------------------------- quantum


def quantum_qml_suite(subset: str = "FD001", models: Iterable[str] = ("quantum_kernel", "vqc", "qnn_regressor"),
                      qubits: Iterable[int] = (4, 6), layers: int = 2, n_train: int = 200, epochs: int = 20) -> dict:
    from .quantum_models.qml import QuantumConfig, run_experiment

    rows = []
    for model in models:
        for q in qubits:
            r = run_experiment(QuantumConfig(model=model, subset=subset, n_qubits=q, layers=layers, n_train=n_train,
                                             epochs=epochs), progress=_progress(f"[{model} {q}q]"))
            c = r["comparison"]
            rows.append({"run": r["run"], "metric": c["metric"], "quantum": c["quantum"],
                         "best_classical_model": c["best_classical_model"], "best_classical": c["best_classical"],
                         "circuit_depth": r["quantum"]["circuit_depth"]})
            log.info("%s %dq: %s quantum=%.3f best classical %s=%.3f", model, q, c["metric"], c["quantum"],
                     c["best_classical_model"], c["best_classical"])
    return {"rows": rows}


def quantum_sequence_suite(subset: str = "FD001", seeds: Iterable[int] = (42,), models: Iterable[str] | None = None,
                           epochs: int = 10, imputation_epochs: int = 8, n_train: int = 1200, n_qubits: int = 4,
                           layers: int = 2, missing_rates: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
                           verify_qiskit: bool = True, save: bool = True) -> dict:
    from .quantum_models.qseq import SEQ_MODELS
    from .quantum_models.seq_experiments import SeqExperimentConfig, run_benchmark

    cfg = SeqExperimentConfig(subset=subset, models=tuple(models or SEQ_MODELS), epochs=epochs,
                              imputation_epochs=imputation_epochs, n_train=n_train, n_qubits=n_qubits, layers=layers,
                              missing_rates=tuple(missing_rates))
    report = run_benchmark(cfg, seeds=tuple(seeds), progress=_progress("[qseq]", every=30), save=save)
    out = {"report": report["name"], "total_seconds": report["total_seconds"],
           "rul": [{k: r[k] for k in ("model", "rmse", "rmse_std", "val_rmse")} for r in report["rul"]["summary"]]}
    if verify_qiskit and save:
        out["qiskit_verification"] = verify_with_qiskit(subset)
    return out


def verify_with_qiskit(subset: str = "FD001", n_windows: int = 20) -> dict:
    """Independent check of the quantum implementation with Qiskit (exact statevector):

    1. circuit level — forward values, gradients w.r.t. inputs and weights (parameter shift vs autograd), parameter
       mapping, edge cases;
    2. model level — every saved quantum sequence model (RUL and imputation) run with all circuits on Qiskit, on
       several input batches: real test windows, a single window, random inputs, all-zero and large-magnitude inputs.
    """
    import numpy as np

    from .quantum_models.qiskit_backend import compare_backends, verify_circuit_equivalence
    from .quantum_models.qseq import QUANTUM_SEQ_MODELS
    from .quantum_models.seq_experiments import SeqExperimentConfig, load_report, load_sequences, load_seq_model
    from .storage import save_result

    rep = load_report(subset)
    cfg = SeqExperimentConfig(subset=subset, window=rep["data"]["window"] if rep else SeqExperimentConfig().window)
    d = load_sequences(cfg)
    rng = np.random.default_rng(0)
    circuit = verify_circuit_equivalence(cfg.n_qubits, cfg.layers)
    log.info("Qiskit circuit check: forward max %.1e, grad max %.1e / %.1e, mapping max %.1e",
             circuit["forward"]["max_abs"], circuit["grad_inputs"]["max_abs"], circuit["grad_weights"]["max_abs"],
             circuit["parameter_mapping"]["max_abs"])
    rows = []
    for task in ("rul", "impute"):
        for name in QUANTUM_SEQ_MODELS:
            try:
                b = load_seq_model(task, name, subset)
            except FileNotFoundError:
                continue
            X = d["X_test"][:n_windows]
            if task == "impute":            # imputer input = [interpolated values, mask]
                X = np.concatenate([X, (rng.random(X.shape) > 0.3).astype(float)], axis=-1)
            batches = {"test windows": X, "single window": X[:1], "random N(0,1)": rng.normal(size=X[:8].shape),
                       "zeros": np.zeros_like(X[:4]), "large ±10": rng.uniform(-10, 10, X[:4].shape)}
            per = {}
            for label, batch in batches.items():
                r = compare_backends(b["model"], batch)
                per[label] = {"max_abs": r["max_abs_diff"], "mean_abs": r["mean_abs_diff"],
                              "qiskit_circuit_evaluations": r["qiskit_circuit_evaluations"]}
            mx = max(v["max_abs"] for v in per.values())
            row = {"task": task, "model": name, "max_abs_output_diff": mx,
                   "mean_abs_output_diff": float(np.mean([v["mean_abs"] for v in per.values()])),
                   "qiskit_circuit_evaluations": int(sum(v["qiskit_circuit_evaluations"] for v in per.values())),
                   "windows": int(sum(len(v) for v in batches.values())), "batches": per}
            if task == "rul":
                row["max_rul_diff_cycles"] = mx * b["cap"]
            rows.append(row)
            log.info("Qiskit model check %s/%s: max |Δ| = %.2e, mean |Δ| = %.2e", task, name, mx,
                     row["mean_abs_output_diff"])
    out = {"subset": subset, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "circuit": circuit, "rows": rows,
           "note": "Simulator-to-simulator equivalence (PennyLane default.qubit vs Qiskit Statevector). "
                   "No real quantum hardware was used."}
    save_result("comparison_reports", f"qseq_qiskit_verification_{subset}", out)
    return out


# ---------------------------------------------------------------- twin, optimisation, explainability, agents


def run_digital_twin(asset_ids: Iterable[str] | None = None, subset: str = "FD001", n_assets: int = 3,
                     horizon: int = 60, load: float = 1.0, maintenance_at: int | None = None) -> dict:
    from .digital_twin.twin import SimulationParams
    from .services import fleet

    ids = list(asset_ids or fleet.default_assets(subset, n_assets))
    out = {}
    for aid in ids:
        sim = fleet.simulate(aid, SimulationParams(horizon=horizon, load=load, maintenance_at=maintenance_at))
        last = sim["series"][-1]
        out[aid] = {"final_cumulative_pof": last.get("cumulative_pof"), "final_rul_median": last.get("rul_median"),
                    "steps": len(sim["series"]), "source": "simulated"}
        log.info("twin %s: horizon %d, final cumulative PoF %s (simulated)", aid, horizon, last.get("cumulative_pof"))
    return out


def run_optimization(asset_ids: Iterable[str] | None = None, subset: str = "FD001", n_assets: int = 5,
                     qaoa_layers: int | None = None) -> dict:
    from .langchain_tools.tools import optimize_maintenance_schedule
    from .services import fleet
    from .storage import save_result

    ids = list(asset_ids or fleet.default_assets(subset, n_assets))
    res = optimize_maintenance_schedule(ids, qaoa_layers)
    name = f"schedule_{time.strftime('%Y%m%d_%H%M%S')}"
    save_result("qaoa_results", name, res)
    for r in res.get("results", []):
        log.info("%-20s cost=%12s gap=%6.2f%%", r["solver"], f"{r.get('total_expected_cost', 0):,.0f}",
                 r.get("gap_percent", 0) or 0)
    return {"saved_as": name, "assets": ids,
            "results": [{k: r.get(k) for k in ("solver", "total_expected_cost", "gap_percent")} for r in res.get("results", [])]}


def run_explainability(subset: str = "FD001", models: Iterable[str] = ("xgboost", "catboost", "lstm")) -> dict:
    from .explainability.explain import explain_rul_model
    from .services import fleet

    out = {}
    for m in models:
        try:
            explain_rul_model(fleet.rul_model(m, subset), fleet.cmapss_data(subset))
            out[m] = "saved"
        except FileNotFoundError as exc:
            out[m] = f"skipped: {exc}"
        log.info("explainability %s: %s", m, out[m])
    return out


def run_maintenance_agents(asset_ids: Iterable[str] | None = None, subset: str = "FD001", n_assets: int = 5,
                           horizon: float = 60, decision: str | None = None, reviewer: str = "cli",
                           use_llm: bool = True, timeout: float = 900) -> dict:
    """Seven-agent maintenance workflow. ``decision`` = approve | reject answers the human-approval gate."""
    from . import db
    from .agents import runner
    from .services import fleet

    ids = list(asset_ids or fleet.default_assets(subset, n_assets))
    row = runner.run_sync({"asset_ids": ids, "subset": subset, "n_assets": len(ids), "horizon": horizon,
                           "optimize": True, "use_llm": use_llm, "force_full_analysis": False, "model": None},
                          timeout=timeout)
    if row["status"] == "awaiting_approval" and decision in ("approve", "reject"):
        runner.submit_approval(row["id"], decision == "approve", reviewer, "decided from the command line")
        t0 = time.time()
        while time.time() - t0 < timeout:
            row = db.get_row(db.agent_runs, row["id"])
            if row["status"] not in ("resuming", "awaiting_approval"):
                break
            time.sleep(0.5)
    for t in (row.get("state") or {}).get("trace", []):
        log.info("  %-26s %-9s %s", t["agent"], t["status"], t.get("summary") or "")
    return {"run_id": row["id"], "status": row["status"], "assets": ids}


def run_missing_data_agents(asset_id: str = "FD001-test-unit34", missing_rate: float = 0.3, pattern: str = "mcar",
                            seed: int = 0, imputer: str | None = None, use_llm: bool = True,
                            max_attempts: int = 3) -> dict:
    from . import db
    from .agents.missing_data.runner import execute

    rec = execute(db.new_id(), {"asset_id": asset_id, "missing_rate": missing_rate, "pattern": pattern, "seed": seed,
                                "imputer": imputer, "use_llm": use_llm, "max_attempts": max_attempts,
                                "subset": asset_id.split("-")[0]})
    s = rec.get("state") or {}
    for i, t in enumerate(s.get("steps") or s.get("trace", []), 1):
        log.info("  %2d. %-13s %-9s %s", i, t.get("step", t.get("agent")), t["status"], t.get("summary") or "")
    rep = s.get("report", {})
    return {"run_id": rec["run_id"], "status": rec["status"], "summary": rep.get("summary"),
            "evaluation": rep.get("evaluation"), "error": rec.get("error")}


# ---------------------------------------------------------------- evaluation & tests


def evaluate_all(publish: bool = False) -> dict:
    """Generate RESULTS.md and API.md from the saved reports: into <project>/reports/docs/ by default, or into the
    repository's documentation/ with ``publish`` (so a test or quick run never overwrites the published results)."""
    from .storage import paths

    cmd = [sys.executable, str(ROOT / "scripts" / "generate_docs.py")] + (["--publish"] if publish else [])
    subprocess.run(cmd, check=True, cwd=ROOT)
    reports = sorted(p.name for p in paths.area("comparison_reports").glob("*.json"))
    log.info("%d comparison report(s): %s", len(reports), ", ".join(reports))
    target = ROOT / "documentation" if publish else paths.reports / "docs"
    return {"reports": reports, "results_md": str(target / "RESULTS.md")}


def run_tests(frontend: bool = True, pytest_args: Iterable[str] = ()) -> dict:
    out = {}
    r = subprocess.run([sys.executable, "-m", "pytest", "backend/tests", "-q", *pytest_args], cwd=ROOT)
    out["backend"] = "passed" if r.returncode == 0 else f"failed (exit {r.returncode})"
    if frontend:
        if (ROOT / "frontend" / "node_modules").is_dir():
            r = subprocess.run(["npx", "vitest", "run"], cwd=ROOT / "frontend")
            out["frontend"] = "passed" if r.returncode == 0 else f"failed (exit {r.returncode})"
        else:
            out["frontend"] = "skipped (run npm install in frontend/)"
    return out
