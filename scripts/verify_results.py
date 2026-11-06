"""Independently recompute the quantum sequence benchmark numbers from raw data and raw predictions.

    python scripts/verify_results.py [--subset FD001] [--tolerance 1e-9]

Checks, using only the saved reports and the original NASA files:
  * the official test set — engines 1…N of RUL_<subset>.txt, truth = min(RUL file, cap), uncapped = RUL file;
  * every seed of every model is present, seeds are distinct;
  * RMSE, MAE, R² and NASA score of every (model, seed) recomputed from the per-engine predictions;
  * summary means and sample SDs recomputed from the per-seed values;
  * imputation RMSE of every (pattern, rate, method, seed) recomputed from per-window squared errors;
  * downstream rows cover every seed;
  * the paired RUL statistics use exactly the official engines;
  * with --recompute (default when per-seed models are saved): every seed's saved models are re-loaded and every
    per-engine RUL prediction, every per-window imputation error and every downstream RUL value is recomputed from
    scratch with the same seeded masks and compared with the saved values.
Writes reports/results_verification_<subset>.json; exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from _cli import log, parser, run

ap = parser(__doc__)
ap.add_argument("--subset", default="FD001")
ap.add_argument("--tolerance", type=float, default=1e-9)
ap.add_argument("--no-recompute", action="store_true", help="skip re-running the saved models")
a = ap.parse_args()


def main():
    from backend.classical_models.metrics import regression_metrics
    from backend.storage import find_files, paths
    from backend.storage.fileio import read_json, write_json

    rep_dir = paths.area("comparison_reports")
    report = read_json(rep_dir / f"qseq_benchmark_{a.subset}.json")
    raw = read_json(rep_dir / f"qseq_raw_{a.subset}.json")
    if report is None or raw is None:
        raise FileNotFoundError("qseq_benchmark / qseq_raw report missing: run scripts/run_quantum_seq.py first")
    checks: list[dict] = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        (log.info if ok else log.error)("%s  %-60s %s", "PASS" if ok else "FAIL", name, detail)

    tol = a.tolerance
    seeds = [str(s) for s in report["seeds"]]
    check("seeds are distinct", len(set(seeds)) == len(seeds), f"{seeds}")
    if raw.get("run_tag") != report.get("run_tag"):
        check("raw file belongs to this report", False, f"{raw.get('run_tag')} vs {report.get('run_tag')}")

    # ---- official test labels
    rul_files = [p for p in find_files("cmapss", f"RUL_{a.subset}.txt")]
    truth_file = np.loadtxt(rul_files[0])
    cap = report["data"].get("rul_cap", 125)
    preds = raw["rul_predictions"]
    any_model = next(iter(preds))
    p0 = sorted(preds[any_model][seeds[0]], key=lambda x: x["unit"])
    units = [x["unit"] for x in p0]
    check("test set = official engines", units == list(range(1, len(truth_file) + 1)), f"{len(units)} engines")
    check("capped truth = min(RUL file, cap)", np.allclose([x["rul_true"] for x in p0], np.minimum(truth_file, cap)))
    check("uncapped truth = RUL file", np.allclose([x["rul_true_uncapped"] for x in p0], truth_file))

    # ---- per (model, seed) metrics
    runs = {(r["model"], str(r["seed"])): r for r in report["rul"]["runs"]}
    worst = 0.0
    missing = []
    for model, by_seed in preds.items():
        for s in seeds:
            if s not in by_seed or (model, s) not in runs:
                missing.append(f"{model}/{s}")
                continue
            p = sorted(by_seed[s], key=lambda x: x["unit"])
            m = regression_metrics([x["rul_true"] for x in p], [x["rul_pred"] for x in p])
            mu = regression_metrics([x["rul_true_uncapped"] for x in p], [x["rul_pred"] for x in p])
            saved, saved_u = runs[(model, s)]["test"], runs[(model, s)]["test_uncapped"]
            for k in ("rmse", "mae", "r2", "nasa_score"):
                worst = max(worst, abs(m[k] - saved[k]) / max(1.0, abs(saved[k])),
                            abs(mu[k] - saved_u[k]) / max(1.0, abs(saved_u[k])))
    check("every model has every seed", not missing, ", ".join(missing) or f"{len(preds)} models × {len(seeds)} seeds")
    check("RUL metrics recomputed from predictions", worst <= tol, f"max relative difference {worst:.2e}")

    # ---- summary mean / SD
    worst = 0.0
    for srow in report["rul"]["summary"]:
        vals = np.array([runs[(srow["model"], s)]["test"]["rmse"] for s in seeds])
        sd = vals.std(ddof=1) if len(vals) > 1 else None
        worst = max(worst, abs(vals.mean() - srow["rmse"]), abs((sd or 0) - (srow["rmse_std"] or 0)))
    check("summary mean ± SD recomputed", worst <= 1e-9, f"max abs difference {worst:.2e}")

    # ---- imputation from per-window errors
    errs = raw["imputation_errors"]
    worst, n = 0.0, 0
    for row in report["imputation"]["evaluation"]:
        key = f"{row['pattern']}|{row['rate']}|{row['method']}"
        vals = []
        for s in seeds:
            e = errs[key][s]
            vals.append(np.sqrt(np.sum(e["se"]) / max(1, np.sum(e["n"]))))
        worst = max(worst, abs(np.mean(vals) - row["rmse"]))
        n += 1
    # per-window errors are stored rounded to 1e-8, so allow that rounding
    check("imputation RMSE recomputed from per-window errors", worst <= 1e-6, f"{n} conditions, max abs diff {worst:.2e}")

    down = report.get("downstream_rul", [])
    short = [r for r in down if r["n_seeds"] != len(seeds)]
    check("downstream rows cover every seed", not short and down, f"{len(down)} rows")
    st = (report.get("statistics") or {}).get("rul", [])
    check("paired statistics use the official engines", all(c["clusters"] == len(truth_file) and c["seeds"] == len(seeds)
                                                            for c in st) and st, f"{len(st)} comparisons")

    # ---- accuracy table: library function vs a direct scikit-learn computation
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    from backend.quantum_models.seq_experiments import rul_accuracy_table

    worst = 0.0
    for row in rul_accuracy_table(preds):
        acc, f1s, aucs = [], [], []
        for s in seeds:
            p = preds[row["model"]][s]
            t = np.array([x["rul_true_uncapped"] for x in p]) <= 30
            y = np.array([x["rul_pred"] for x in p])
            acc.append(accuracy_score(t, y <= 30))
            f1s.append(f1_score(t, y <= 30))
            aucs.append(roc_auc_score(t, -y))
        worst = max(worst, abs(np.mean(acc) - row["accuracy"]), abs(np.mean(f1s) - row["f1"]),
                    abs(np.mean(aucs) - row["roc_auc"]))
    check("accuracy table recomputed with scikit-learn", worst <= 1e-12, f"max abs diff {worst:.2e}")

    # ---- downstream means from per-seed rows
    per_seed = raw.get("downstream")
    if per_seed:
        groups = {}
        for r in per_seed:
            groups.setdefault((r["pattern"], r["rate"], r["method"], r["rul_model"]), []).append(r["rul_rmse"])
        worst = max(abs(np.mean(groups[(r["pattern"], r["rate"], r["method"], r["rul_model"])]) - r["rul_rmse"])
                    for r in down)
        check("downstream means recomputed from per-seed rows", worst <= 1e-9, f"max abs diff {worst:.2e}")
    else:
        check("downstream per-seed rows saved", False, "this report predates per-seed downstream provenance")

    if not a.no_recompute:
        _recompute(report, raw, check, tol)

    ok = all(c["ok"] for c in checks)
    out = paths.root / "reports" / f"results_verification_{a.subset}.json"
    write_json(out, {"subset": a.subset, "run_tag": report.get("run_tag"), "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "passed": ok, "checks": checks})
    log.info("%s — %d/%d checks · %s", "VERIFIED" if ok else "FAILED", sum(c["ok"] for c in checks), len(checks), out)
    if not ok:
        raise RuntimeError("result verification failed")
    return {"passed": ok}


def _recompute(report, raw, check, tol):
    """Re-run every seed's saved models and compare with the saved raw values."""
    from backend.classical_models.metrics import regression_metrics
    from backend.missing_data import BASELINE_IMPUTERS
    from backend.quantum_models import seq_experiments as sx
    from backend.storage.artifacts import load_model
    from backend.quantum_models.qseq import model_from_state

    c = report["config"]
    cfg = sx.SeqExperimentConfig(**{k: c[k] for k in ("subset", "window", "n_qubits", "layers", "hidden", "n_train",
                                                       "n_val", "n_eval", "seed") if k in c})
    d = sx.load_sequences(cfg)
    check("evaluation data rebuilt identically", d["test_units"].tolist() == raw["test_units"]
          and d["eval_units"].tolist() == raw["eval_units"], f"{len(d['X_test'])} test, {len(d['X_eval'])} eval windows")
    models = [m for m in c["models"]]
    worst_pred = worst_imp = worst_down = 0.0
    n_pred = n_imp = n_down = 0
    saved_down = {(r["seed"], r["pattern"], r["rate"], r["method"], r["rul_model"]): r["rul_rmse"]
                  for r in raw.get("downstream") or []}
    for seed in report["seeds"]:
        try:
            rul = {m: model_from_state(**_bundle(load_model("quantum_ai", f"qseq_rul_{m}_{cfg.subset}_s{seed}")))
                   for m in models}
            imp = {m: model_from_state(**_bundle(load_model("quantum_ai", f"qseq_impute_{m}_{cfg.subset}_s{seed}")))
                   for m in models}
        except FileNotFoundError as exc:
            check(f"seed {seed}: saved models available", False, str(exc)[:120])
            continue
        for m, model in rul.items():
            pred = sx.predict_rul_windows(model, d["X_test"], d["cap"])
            saved = np.array([x["rul_pred"] for x in raw["rul_predictions"][m][str(seed)]])
            worst_pred = max(worst_pred, float(np.max(np.abs(pred - saved))))
            n_pred += len(pred)
        for pattern in c["patterns"]:
            for rate in c["missing_rates"]:
                mask, mask_test = sx.evaluation_masks(seed, rate, pattern, d["X_eval"].shape, d["X_test"].shape)
                for method in list(BASELINE_IMPUTERS) + models:
                    x_imp = sx.impute_any(method, d["X_eval"], mask, d["means"], imp)
                    se, _ = sx.window_errors(d["X_eval"], x_imp, mask)
                    ref = np.asarray(raw["imputation_errors"][f"{pattern}|{rate}|{method}"][str(seed)]["se"])
                    worst_imp = max(worst_imp, float(np.max(np.abs(se - ref))))
                    n_imp += len(se)
                    if saved_down:
                        x_t = sx.impute_any(method, d["X_test"], mask_test, d["means"], imp)
                        for rm, rmodel in rul.items():
                            v = regression_metrics(d["y_test"], sx.predict_rul_windows(rmodel, x_t, d["cap"]))["rmse"]
                            worst_down = max(worst_down, abs(v - saved_down[(seed, pattern, rate, method, rm)]))
                            n_down += 1
    check("recomputed per-engine RUL predictions", worst_pred <= tol, f"{n_pred} predictions, max |Δ| {worst_pred:.2e}")
    check("recomputed per-window imputation errors", worst_imp <= 1e-6, f"{n_imp} windows, max |Δ| {worst_imp:.2e}")
    if saved_down:
        check("recomputed downstream RUL RMSEs", worst_down <= tol, f"{n_down} values, max |Δ| {worst_down:.2e}")


def _bundle(b):
    return {"config": b["config"], "state": b["state"]}


sys.exit(run("verify_results", a, main))
