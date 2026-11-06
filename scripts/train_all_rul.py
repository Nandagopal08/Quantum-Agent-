"""Train and evaluate every C-MAPSS RUL model (kept for compatibility; see train_all_models.py).

    python scripts/train_all_rul.py [FD001] [--models lstm,gru] [--epochs 40]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("subset", nargs="?", default="FD001")
ap.add_argument("--models", default=None)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()


def main():
    from backend.workflows import train_rul_suite
    return train_rul_suite(a.subset, csv(a.models), a.epochs, a.seed)


sys.exit(run("train_all_rul", a, main))
