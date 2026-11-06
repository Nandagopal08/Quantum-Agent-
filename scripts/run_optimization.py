"""Maintenance scheduling: QAOA (PennyLane) vs exact MILP, greedy and simulated annealing.

    python scripts/run_optimization.py [--assets FD001-test-unit34,FD001-test-unit81] [--n-assets 5] [--qaoa-layers 2]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("--assets", default=None)
ap.add_argument("--subset", default="FD001")
ap.add_argument("--n-assets", type=int, default=5)
ap.add_argument("--qaoa-layers", type=int, default=None)
a = ap.parse_args()


def main():
    from backend.workflows import run_optimization
    return run_optimization(csv(a.assets), a.subset, a.n_assets, a.qaoa_layers)


sys.exit(run("run_optimization", a, main))
