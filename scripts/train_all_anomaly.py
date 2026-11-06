"""Extract IMS features and train/evaluate every bearing anomaly detector (kept for compatibility; see
train_all_models.py).

    python scripts/train_all_anomaly.py [--tests 1st_test,2nd_test,3rd_test] [--epochs 60]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("--tests", default="2nd_test,1st_test,3rd_test")
ap.add_argument("--epochs", type=int, default=60)
a = ap.parse_args()


def main():
    from backend.workflows import train_anomaly_suite
    return train_anomaly_suite(csv(a.tests), a.epochs)


sys.exit(run("train_all_anomaly", a, main))
