"""Digital-twin simulation of future operation (values are simulated and labelled as such).

    python scripts/run_digital_twin.py [--assets FD001-test-unit34,IMS-2nd_test-bearing1] [--horizon 60] [--load 1.2]
                                       [--maintenance-at 20]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("--assets", default=None, help="default: the highest-risk test engines")
ap.add_argument("--subset", default="FD001")
ap.add_argument("--n-assets", type=int, default=3)
ap.add_argument("--horizon", type=int, default=60)
ap.add_argument("--load", type=float, default=1.0)
ap.add_argument("--maintenance-at", type=int, default=None)
a = ap.parse_args()


def main():
    from backend.workflows import run_digital_twin
    return run_digital_twin(csv(a.assets), a.subset, a.n_assets, a.horizon, a.load, a.maintenance_at)


sys.exit(run("run_digital_twin", a, main))
