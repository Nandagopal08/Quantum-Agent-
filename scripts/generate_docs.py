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
    DOCS.mkdir(exist_ok=True)
    (DOCS / "RESULTS.md").write_text(results())
    (DOCS / "API.md").write_text(api_doc())
    (DOCS / "agent_graph.mmd").write_text(mermaid())
    print("wrote documentation/RESULTS.md, API.md, agent_graph.mmd")
