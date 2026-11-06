"""Extract IMS features and train/evaluate every bearing anomaly detector. Usage:
    python scripts/train_all_anomaly.py [--tests 1st_test,2nd_test,3rd_test]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.classical_models.anomaly import DETECTORS, train_detector, train_supervised_cross_test
from backend.datasets import ims
from backend.storage import save_result

ap = argparse.ArgumentParser()
ap.add_argument("--tests", default="2nd_test,1st_test,3rd_test")
a = ap.parse_args()
tests = a.tests.split(",")
rows = []
for t in tests:
    t0 = time.time(); ims.extract_features(t); print(f"{t}: features in {time.time() - t0:.0f}s", flush=True)
    ims.eda(t)
    for d in DETECTORS:
        r = train_detector(d, t)
        m = r["metrics"]
        rows.append({"test": t, "model": d, **{k: m[k] for k in ("roc_auc", "precision", "recall", "f1", "accuracy", "false_alarm_rate")},
                     "lead_time_hours": {b: v["lead_time_hours"] for b, v in r["detection"].items()}, "train_seconds": r["train_seconds"]})
        print(json.dumps(rows[-1]), flush=True)
if len(tests) >= 3:
    for ev in tests:
        r = train_supervised_cross_test([t for t in tests if t != ev], ev)
        m = r["metrics"]
        rows.append({"test": ev, "model": "random_forest_supervised", **{k: m[k] for k in ("roc_auc", "precision", "recall", "f1", "accuracy", "false_alarm_rate")},
                     "lead_time_hours": {b: v["lead_time_hours"] for b, v in r["detection"].items()}, "train_tests": r["train_tests"]})
        print(json.dumps(rows[-1]), flush=True)
save_result("comparison_reports", "ims_anomaly_comparison", {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": rows,
            "labelling": "pseudo-labels: first 25% healthy, last 10% of failed bearings anomalous (see backend/datasets/ims.py)"})
