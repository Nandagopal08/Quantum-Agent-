"""Train and evaluate every C-MAPSS RUL model; writes a comparison report. Usage:
    python scripts/train_all_rul.py [FD001] [--models lstm,gru] [--epochs 40]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.classical_models.rul import ALL_MODELS, train_rul_model
from backend.storage import save_result

ap = argparse.ArgumentParser()
ap.add_argument("subset", nargs="?", default="FD001")
ap.add_argument("--models", default=",".join(ALL_MODELS))
ap.add_argument("--epochs", type=int, default=40)
a = ap.parse_args()
rows = []
for m in a.models.split(","):
    r = train_rul_model(m, a.subset, epochs=a.epochs, progress=lambda f, msg: print(f"  {m}: {msg}", flush=True) if "epoch" in msg and msg.split()[1].split("/")[0] in ("1", "5", "10", "20", "30", "40") else None)
    t, c = r["test"], r["failure_classification"]
    rows.append({"model": m, **{k: t[k] for k in ("rmse", "mae", "r2", "nasa_score")}, "pof_roc_auc": c["roc_auc"],
                 "pof_f1": c["f1"], "interval_coverage_90": r["interval_coverage_90"], "train_seconds": r["train_seconds"]})
    print(json.dumps(rows[-1]), flush=True)
save_result("comparison_reports", f"rul_{a.subset}_comparison", {"subset": a.subset, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows})
