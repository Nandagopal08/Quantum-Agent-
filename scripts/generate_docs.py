"""Generate documentation/RESULTS.md, documentation/API.md and the agent-graph diagram from real artefacts.

Every number in RESULTS.md is read from reports saved by actual runs (experiments/comparison_reports,
optimization/qaoa_results); nothing is typed by hand. Usage (with the same storage env vars as the API):
    python scripts/generate_docs.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.agents.graph import mermaid  # noqa: E402
from backend.config import get_config  # noqa: E402
from backend.storage import load_metadata, paths  # noqa: E402
from backend.storage.fileio import read_json  # noqa: E402

DOCS = ROOT / "documentation"


def f(v, d=2):
    return "—" if v is None else f"{v:,.{d}f}"


def pct(v, d=1):
    return "—" if v is None else f"{100 * v:.{d}f}%"


def stamp(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def results() -> str:
    cfg = get_config()
    rep = paths.area("comparison_reports")
    out = ["# Verified results", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/generate_docs.py` from reports written by actual runs "
           f"(storage mode: `{cfg.storage_mode}`). Re-run the script after new experiments; do not edit numbers by hand.", ""]
    meta = load_metadata()
    out += ["## Datasets", "", "| Dataset | Archive | SHA-256 | Extracted files | Downloaded |", "|---|---|---|---|---|"]
    for name, m in meta.items():
        out.append(f"| {m.get('title', name)} | {m.get('file_size_bytes', 0) / 1e6:,.1f} MB | `{(m.get('sha256') or '')[:16]}…` | "
                   f"{m.get('extracted_file_count', '—')} | {m.get('download_date', '—')} |")
    out.append("")
    for p in sorted(rep.glob("rul_*_comparison.json")):
        d = read_json(p)
        out += [f"## RUL prediction — C-MAPSS {d['subset']}", "",
                f"Official test set: one prediction per engine at its last cycle vs `RUL_{d['subset']}.txt` (RUL capped at 125). "
                f"PoF = P(RUL ≤ 30) from the calibrated Gaussian; coverage = share of test engines inside the 90 % conformal interval. "
                f"Report: `{p.name}` ({stamp(p)}).", "",
                "| Model | RMSE | MAE | R² | NASA score | PoF ROC-AUC | 90 % interval coverage | Train time (s) |",
                "|---|---|---|---|---|---|---|---|"]
        for r in sorted(d["rows"], key=lambda r: r["rmse"]):
            out.append(f"| {r['model']} | {f(r['rmse'])} | {f(r['mae'])} | {f(r['r2'], 3)} | {f(r['nasa_score'], 0)} | "
                       f"{f(r['pof_roc_auc'], 3)} | {pct(r['interval_coverage_90'], 0)} | {f(r['train_seconds'], 0)} |")
        out.append("")
    p = rep / "ims_anomaly_comparison.json"
    if p.exists():
        d = read_json(p)
        out += ["## Bearing anomaly detection — NASA IMS", "",
                f"Labels — {d['labelling']}. Lead time = hours from the first sustained alarm (3 consecutive "
                f"snapshots above the 99th-percentile baseline threshold) to the end of the test. Report: `{p.name}` ({stamp(p)}).", "",
                "| Test | Model | ROC-AUC | Precision | Recall | F1 | False-alarm rate | Lead time (h) |", "|---|---|---|---|---|---|---|---|"]
        for r in d["rows"]:
            lead = ", ".join(f"b{b}: {h}" for b, h in r["lead_time_hours"].items())
            out.append(f"| {r['test']} | {r['model']} | {f(r['roc_auc'], 3)} | {f(r['precision'], 3)} | {f(r['recall'], 3)} | "
                       f"{f(r['f1'], 3)} | {pct(r['false_alarm_rate'])} | {lead} |")
        out += ["", "`random_forest_supervised` is trained on the other two tests and evaluated on the listed one.", ""]
    qs = sorted(rep.glob("quantum_vs_classical_*.json"))
    if qs:
        out += ["## Quantum vs classical (PennyLane statevector simulator)", "",
                "Classical baselines use the same PCA-reduced inputs. Single runs; **no quantum advantage is claimed.**", "",
                "| Run | Task | Qubits | Depth | Gates | Params | Quantum | Best classical (same inputs) | Train (s) | Report |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for p in qs:
            d = read_json(p)
            q, c = d["quantum"], d["comparison"]
            out.append(f"| {d['run']} | {d['task']} | {q['qubits']} | {q['circuit_depth']} | {q['gate_count']} | {q['trainable_params']} | "
                       f"{c['metric']} {f(c['quantum'], 3)} | {c['best_classical_model']} {f(c['best_classical'], 3)} | "
                       f"{f(q['train_seconds'], 1)} | {stamp(p)} |")
        out.append("")
    for p in sorted(rep.glob("qseq_benchmark_*.json")):  # (qseq_raw_* holds raw predictions, not summarised)
        out += qseq_section(read_json(p), p, rep)
    qa = sorted(paths.area("qaoa_results").glob("schedule_*.json"))
    if qa:
        out += ["## Maintenance scheduling: QAOA vs classical", "",
                "Costs evaluated with the original cost model; gap relative to the exact MILP (HiGHS) optimum. "
                "QAOA's plan is the cheapest feasible bitstring among its most probable outcomes; `P(plan)` is the "
                "probability the circuit itself assigned to it.", "",
                "| Run | Assets | QUBO vars | MILP cost | Greedy gap | Annealing gap | QAOA gap | QAOA p / depth | P(plan) | QAOA time (s) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for p in qa:
            d = read_json(p)
            r = {x["solver"]: x for x in d["results"]}
            q = r.get("qaoa", {})
            out.append(f"| {p.stem} | {len(d['problem']['assets'])} | {d['qubo_variables']} | {f(r['milp_highs']['total_expected_cost'], 0)} | "
                       f"{f(r['greedy'].get('gap_percent'))}% | {f(r['simulated_annealing'].get('gap_percent'))}% | "
                       f"{f(q.get('gap_percent'))}% | {q.get('layers', '—')} / {q.get('circuit_depth', '—')} | "
                       f"{pct(q.get('probability_of_selected'), 3)} | {f(q.get('runtime_seconds'), 1)} |")
        out.append("")
    shap = sorted(paths.area("shap").glob("rul_*.json"))
    if shap:
        out += ["## Explainability (top drivers)", "", "| Model | Top SHAP features | Top permutation-importance sensors |", "|---|---|---|"]
        for p in shap:
            d = read_json(p)
            s = ", ".join(i["feature"] for i in (d.get("shap_global") or {}).get("importance", [])[:4]) or "n/a (sequence model)"
            pi = ", ".join(f"{i['feature_group']} ({i['sensor_name']})" for i in d["permutation_importance"][:3])
            out.append(f"| {d['model']} | {s} | {pi} |")
        out.append("")
    return "\n".join(out)


LABEL = {"q_lstm": "Quantum LSTM", "q_transformer": "Quantum Transformer", "q_cnn_lstm": "Quantum CNN-LSTM",
         "c_lstm": "LSTM (classical twin)", "c_transformer": "Transformer (classical twin)",
         "c_cnn_lstm": "CNN-LSTM (classical twin)", "mean": "mean fill", "locf": "LOCF", "linear": "linear interpolation"}


def pm(v, sd, d=2):
    return f(v, d) if sd is None else f"{f(v, d)} ± {f(sd, d)}"


def _ci(c, d=3):
    return "—" if not c or c[0] is None else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def _stats_rows(rows, d=3):
    out = ["| A vs B | RMSE(A) − RMSE(B) | 95% CI (engines) | 95% CI (seeds) | Holm p | Verdict |", "|---|---|---|---|---|---|"]
    for c in rows:
        out.append(f"| {LABEL.get(c['a'], c['a'])} vs {LABEL.get(c['b'], c['b'])} | {f(c['rmse_diff'], d)} | "
                   f"{_ci(c['bootstrap_ci95'], d)} | {_ci(c['seed_t_ci95'], d)} | {f(c.get('wilcoxon_p_holm'), 3)} | "
                   f"{c.get('verdict', '—')} |")
    return out


def qseq_section(d: dict, p: Path, rep: Path) -> list[str]:
    cfg = d["config"]
    n = len(d["seeds"])
    dd = d["data"]
    out = [f"## Quantum sequence models — C-MAPSS {cfg['subset']} (QLSTM, Quantum Transformer, Quantum CNN-LSTM)", "",
           f"{cfg['n_qubits']}-qubit circuits × {cfg['layers']} variational layers (PennyLane `default.qubit`); "
           f"{dd['window']}-cycle windows of {len(d['sensors'])} sensors; {dd['n_train_windows']} training windows; "
           f"seeds {', '.join(map(str, d['seeds']))} (mean ± sample SD over {n}); run `{d.get('run_tag', '—')}`, "
           f"{f(d.get('total_seconds', 0) / 60, 0)} min. Each classical twin is the same architecture with the circuit "
           f"replaced by a same-width classical layer. Report: `{p.name}` ({stamp(p)}). **No quantum advantage is claimed.**", "",
           f"Labels: RMSE/MAE/R²/NASA use RUL capped at {dd.get('rul_cap', 125)} (the training target; "
           f"{dd.get('test_engines_above_cap', '?')} of {dd['n_test_engines']} test engines have a larger true RUL). "
           "*Uncapped* compares the same predictions with the raw `RUL_FD001.txt` values. *Ablated*: every circuit (or the "
           "twin's counterpart layer) replaced by its batch mean at inference.", "",
           "### RUL prediction (complete data, 100 official test engines)", "",
           "| Model | RMSE | MAE | R² | NASA score | RMSE uncapped | RMSE ablated | Params (circuit) | Circuit | Train (s) | Infer (ms) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(d["rul"]["summary"], key=lambda r: r["rmse"]):
        params = f"{r['trainable_params']}" + (f" ({r['quantum_params']})" if r.get("quantum") else "")
        c = r.get("circuit") or {}
        circ = (f"{r.get('n_qubits')}q×{r.get('circuit_layers', '?')}L, depth {c.get('depth')}, {c.get('gates')} gates, "
                f"{r.get('circuits')} circuits") if r.get("quantum") else "—"
        out.append(f"| {LABEL.get(r['model'], r['model'])} | {pm(r['rmse'], r['rmse_std'])} | {pm(r['mae'], r.get('mae_std'))} | "
                   f"{f(r['r2'], 3)} | {f(r['nasa_score'], 0)} | {f(r.get('rmse_uncapped'))} | {f(r.get('rmse_ablated'))} | "
                   f"{params} | {circ} | {f(r['train_seconds'], 0)} | {f(1000 * r['inference_seconds'], 0) if r.get('inference_seconds') is not None else '—'} |")
    raw = read_json(rep / f"qseq_raw_{cfg['subset']}.json")
    if raw and raw.get("run_tag") == d.get("run_tag"):
        from backend.quantum_models.seq_experiments import rul_accuracy_table

        acc = rul_accuracy_table(raw["rul_predictions"])
        out += ["", f"### Accuracy: \"fails within {acc[0]['horizon']} cycles\" (100 official test engines, "
                    f"{acc[0]['positives']} positive)", "",
                "Classification view of the same RUL predictions: an engine is predicted to fail when RUL_pred ≤ 30 "
                "cycles; ROC-AUC ranks engines by −RUL_pred; *within ±10/±20* = share of engines whose predicted RUL is "
                "within 10/20 cycles of the true (capped) RUL. Mean ± SD over seeds, from `qseq_raw_FD001.json`.", "",
                "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False-alarm rate | Within ±10 | Within ±20 |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in acc:
            out.append(f"| {LABEL.get(r['model'], r['model'])} | " + " | ".join(
                pm(r[k], r[f"{k}_std"], 3) for k in ("accuracy", "precision", "recall", "f1", "roc_auc",
                                                     "false_alarm_rate", "within_10", "within_20")) + " |")
    st = d.get("statistics") or {}
    if st.get("rul"):
        out += ["", "### Paired comparison: quantum vs classical twin (RUL)", "", st.get("method", ""), ""] + _stats_rows(st["rul"], 2)
    ev = d["imputation"]["evaluation"]
    rates = sorted({r["rate"] for r in ev})
    for pattern in sorted({r["pattern"] for r in ev}):
        rows = {(r["method"], r["rate"]): r for r in ev if r["pattern"] == pattern}
        mid = rates[len(rates) // 2]
        methods = sorted({m for m, _ in rows}, key=lambda m: rows[(m, mid)]["rmse"])
        title = "random missingness (MCAR)" if pattern == "mcar" else "sensor outages (block gaps)"
        out += ["", f"### Imputation RMSE on removed values — {title}", "",
                f"Standardised sensor units; {dd['n_eval_windows']} windows from {dd.get('n_eval_engines', '?')} test engines; "
                f"identical seeded masks for every method. Sorted by the {int(mid * 100)} % column.", "",
                "| Method | " + " | ".join(f"{int(x * 100)}% missing" for x in rates) + " |",
                "|---|" + "---|" * len(rates)]
        for m in methods:
            out.append(f"| {LABEL.get(m, m)} | " + " | ".join(pm(rows[(m, x)]["rmse"], rows[(m, x)]["rmse_std"], 4)
                                                          for x in rates) + " |")
        fam = [c for c in st.get("imputation", []) if c["pattern"] == pattern and c["rate"] == mid]
        if fam:
            out += ["", f"Paired comparisons at {int(mid * 100)} % missing ({pattern}):", ""] + _stats_rows(fam, 4)
    models = d["imputation"].get("models") or []
    if models and "final_train_loss" in models[0]:
        out += ["", "### Learned imputers — training diagnostics", "",
                "| Imputer | Val RMSE | Final train loss | Circuit/counterpart grad norm, epoch 1 → last | RMSE vs ablated (MCAR 30 %) |",
                "|---|---|---|---|---|"]
        for m in sorted(models, key=lambda m: m["model"]):
            ab = m.get("ablation_mcar30") or {}
            out.append(f"| {LABEL.get(m['model'], m['model'])} | {pm(m['val_rmse'], m.get('val_rmse_std'), 4)} | "
                       f"{f(m.get('final_train_loss'), 4)} | {f(m.get('first_epoch_grad_norm'), 4)} → {f(m.get('last_epoch_grad_norm'), 4)} | "
                       f"{f(ab.get('rmse'), 4)} vs {f(ab.get('rmse_ablated'), 4)} |")
    down = d.get("downstream_rul") or []
    if down:
        rul_models = [m for m in ("q_lstm", "c_lstm", "q_transformer", "c_transformer", "q_cnn_lstm", "c_cnn_lstm")
                      if any(r["rul_model"] == m for r in down)]
        for pattern in sorted({r["pattern"] for r in down if r["pattern"] != "none"}):
            top = max(r["rate"] for r in down if r["pattern"] == pattern)
            cell = {(r["method"], r["rul_model"]): r for r in down
                    if (r["pattern"] == pattern and r["rate"] == top) or r["method"] == "complete_data"}
            methods = ["complete_data"] + sorted({r["method"] for r in down if r["method"] != "complete_data"},
                                                 key=lambda m: cell.get((m, rul_models[0]), {}).get("rul_rmse", 1e9))
            out += ["", f"### Downstream RUL RMSE (cycles) at {int(top * 100)} % missing — {pattern}", "",
                    "Rows: imputer; columns: RUL model. Mean ± SD over seeds.", "",
                    "| Imputer | " + " | ".join(LABEL.get(m, m) for m in rul_models) + " |", "|---|" + "---|" * len(rul_models)]
            for meth in methods:
                vals = [cell.get((meth, m)) for m in rul_models]
                out.append(f"| {'**complete data**' if meth == 'complete_data' else LABEL.get(meth, meth)} | " +
                           " | ".join(pm(v["rul_rmse"], v["rul_rmse_std"]) if v else "—" for v in vals) + " |")
    if d.get("quantum_vs_twin"):
        out += ["", "### Quantum vs classical twin (means)", "", "| Architecture | RUL RMSE quantum / twin | Mean imputation RMSE quantum / twin |",
                "|---|---|---|"]
        for r in d["quantum_vs_twin"]:
            out.append(f"| {r['arch']} | {f(r.get('rul_rmse_quantum'))} / {f(r.get('rul_rmse_twin'))} | "
                       f"{f(r.get('impute_rmse_quantum_mean'), 4)} / {f(r.get('impute_rmse_twin_mean'), 4)} |")
    qk = read_json(rep / f"qseq_qiskit_verification_{cfg['subset']}.json")
    if qk and qk.get("rows"):
        out += ["", "### Qiskit cross-check (simulator vs simulator — no quantum hardware)", ""]
        if qk.get("circuit"):
            c = qk["circuit"]
            out += [f"Circuit level ({c['cases']} cases incl. {', '.join(c['edge_cases'])}): "
                    + "; ".join(f"{lbl} max {c[k]['max_abs']:.1e} / mean {c[k]['mean_abs']:.1e}"
                                for k, lbl in (("forward", "forward"), ("grad_inputs", "∂/∂inputs (parameter shift vs autograd)"),
                                               ("grad_weights", "∂/∂weights"), ("parameter_mapping", "parameter mapping"))) + ".", ""]
        out += ["| Model | Task | Windows | Qiskit circuit evaluations | Max |Δ| | Mean |Δ| |", "|---|---|---|---|---|---|"]
        for r in qk["rows"]:
            mean = r.get("mean_abs_output_diff")
            out.append(f"| {LABEL.get(r['model'], r['model'])} | {r.get('task', 'rul')} | {r['windows']} | {r['qiskit_circuit_evaluations']:,} | "
                       f"{r['max_abs_output_diff']:.1e} | {'—' if mean is None else f'{mean:.1e}'} |")
    out.append("")
    return out


def api_doc() -> str:
    from backend.main import app

    spec = app.openapi()
    out = ["# API reference", "", "Generated from the FastAPI OpenAPI schema by `scripts/generate_docs.py`. Interactive docs: "
           "`http://localhost:8000/docs`. When `QADT_API_TOKEN` is set, send `Authorization: Bearer <token>` "
           "(or `?token=` for Server-Sent Events).", ""]
    by_tag: dict[str, list] = {}
    for path, methods in spec["paths"].items():
        for method, op in methods.items():
            by_tag.setdefault((op.get("tags") or ["other"])[0], []).append((method.upper(), path, op))
    for tag, ops in by_tag.items():
        out += [f"## {tag}", "", "| Method | Path | Description |", "|---|---|---|"]
        for m, p, op in ops:
            desc = (op.get("description") or op.get("summary") or "").strip().splitlines()[0] if (op.get("description") or op.get("summary")) else ""
            out.append(f"| {m} | `{p}` | {desc} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate RESULTS.md / API.md / agent_graph.mmd from saved reports.")
    ap.add_argument("--publish", action="store_true",
                    help="write into documentation/ (the repository); otherwise into <project>/reports/docs/")
    ap.add_argument("--out", default=None, help="explicit output folder")
    a = ap.parse_args()
    out = Path(a.out) if a.out else (DOCS if a.publish else paths.reports / "docs")
    if a.publish and not get_config().is_google_drive and "--allow-dev-publish" not in sys.argv:
        print(f"note: publishing results from storage mode '{get_config().storage_mode}' ({get_config().project_root})")
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS.md").write_text(results())
    (out / "API.md").write_text(api_doc())
    (out / "agent_graph.mmd").write_text(mermaid())
    print(f"wrote {out}/RESULTS.md, API.md, agent_graph.mmd")
