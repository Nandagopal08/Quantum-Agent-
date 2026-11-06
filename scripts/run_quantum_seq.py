"""Quantum sequence-model benchmark only (shortcut for run_quantum_experiments.py --suite sequence).

    python scripts/run_quantum_seq.py [FD001] [--seeds 42,43,44] [--epochs 10] [--imputation-epochs 8]
                                      [--models q_lstm,c_lstm] [--rates 0.1,0.2,0.3,0.4,0.5] [--quick]
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("subset", nargs="?", default="FD001")
ap.add_argument("--seeds", default="42")
ap.add_argument("--models", default=None)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--imputation-epochs", type=int, default=8)
ap.add_argument("--n-train", type=int, default=1200)
ap.add_argument("--rates", default="0.1,0.2,0.3,0.4,0.5")
ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline; saves nothing")
a = ap.parse_args()


def main():
    from backend.workflows import quantum_sequence_suite
    if a.quick:
        return quantum_sequence_suite(a.subset, [42], csv(a.models), 1, 1, 128, missing_rates=csv(a.rates, float),
                                      save=False)
    return quantum_sequence_suite(a.subset, csv(a.seeds, int), csv(a.models), a.epochs, a.imputation_epochs, a.n_train,
                                  missing_rates=csv(a.rates, float))


sys.exit(run("run_quantum_seq", a, main))
