"""Quantum experiments (PennyLane; Qiskit cross-check).

Suites:
  qml       VQC, quantum-kernel SVM and hybrid QNN on PCA-encoded C-MAPSS vs classical baselines on the same inputs
  sequence  Quantum LSTM / Quantum Transformer / Quantum CNN-LSTM vs classical twins: RUL on complete data, then
            imputation at several missing rates (MCAR and block outages) and downstream RUL
  qiskit    re-run the saved quantum sequence models with Qiskit's statevector and compare with PennyLane

    python scripts/run_quantum_experiments.py --suite all
    python scripts/run_quantum_experiments.py --suite sequence --seeds 42,43,44 --epochs 10 --rates 0.1,0.3,0.5
    python scripts/run_quantum_experiments.py --suite qml --qml-models vqc --qubits 4,6
"""
import sys

from _cli import csv, parser, run

ap = parser(__doc__)
ap.add_argument("--suite", choices=("qml", "sequence", "qiskit", "all"), default="all")
ap.add_argument("--subset", default="FD001")
g = ap.add_argument_group("qml suite")
g.add_argument("--qml-models", default="quantum_kernel,vqc,qnn_regressor")
g.add_argument("--qubits", default="4,6")
g.add_argument("--qml-epochs", type=int, default=20)
g.add_argument("--qml-train", type=int, default=200)
g = ap.add_argument_group("sequence suite")
g.add_argument("--seq-models", default=None, help="default: q_lstm,q_transformer,q_cnn_lstm,c_lstm,c_transformer,c_cnn_lstm")
g.add_argument("--seeds", default="42")
g.add_argument("--epochs", type=int, default=10, help="RUL epochs")
g.add_argument("--imputation-epochs", type=int, default=8)
g.add_argument("--n-train", type=int, default=1200, help="training windows")
g.add_argument("--n-qubits", type=int, default=4)
g.add_argument("--layers", type=int, default=2)
g.add_argument("--rates", default="0.1,0.2,0.3,0.4,0.5", help="missing rates to evaluate")
a = ap.parse_args()


def main():
    from backend import workflows as w
    out = {}
    if a.suite in ("qml", "all"):
        out["qml"] = w.quantum_qml_suite(a.subset, csv(a.qml_models), csv(a.qubits, int), n_train=a.qml_train,
                                         epochs=a.qml_epochs)
    if a.suite in ("sequence", "all"):
        out["sequence"] = w.quantum_sequence_suite(a.subset, csv(a.seeds, int), csv(a.seq_models), a.epochs,
                                                   a.imputation_epochs, a.n_train, a.n_qubits, a.layers,
                                                   csv(a.rates, float), verify_qiskit=True)
    if a.suite == "qiskit":
        out["qiskit"] = w.verify_with_qiskit(a.subset)
    return out


sys.exit(run("run_quantum_experiments", a, main))
