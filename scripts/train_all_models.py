"""Train and evaluate the classical models: C-MAPSS RUL regressors and IMS anomaly detectors.
Deep models checkpoint every epoch to storage and resume automatically when re-run.

    python scripts/train_all_models.py [--subset FD001] [--rul-models lstm,xgboost] [--epochs 40]
                                       [--ims-tests 2nd_test,1st_test,3rd_test] [--skip-rul] [--skip-anomaly]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("--subset", default="FD001")
ap.add_argument("--rul-models", default=None, help="default: all 8 (random_forest … transformer)")
ap.add_argument("--epochs", type=int, default=40, help="epochs for deep RUL models")
ap.add_argument("--anomaly-epochs", type=int, default=60)
ap.add_argument("--ims-tests", default="2nd_test,1st_test,3rd_test")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--skip-rul", action="store_true")
ap.add_argument("--skip-anomaly", action="store_true")
a = ap.parse_args()


def main():
    from backend.workflows import train_anomaly_suite, train_rul_suite
    out = {}
    if not a.skip_rul:
        out["rul"] = train_rul_suite(a.subset, csv(a.rul_models), a.epochs, a.seed)
    if not a.skip_anomaly:
        out["anomaly"] = train_anomaly_suite(csv(a.ims_tests), a.anomaly_epochs)
    return out


sys.exit(run("train_all_models", a, main))
