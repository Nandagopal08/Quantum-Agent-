"""Hybrid quantum-classical experiments with PennyLane (statevector simulator, optional hardware device).

Task: C-MAPSS "fails within ``failure_horizon`` cycles" classification and RUL regression.
Inputs are reduced with PCA (fitted on training units only) to ``n_qubits`` components and scaled to
[0, π] for angle encoding.

Models
* ``vqc``            – Variational Quantum Classifier: AngleEmbedding + StronglyEntanglingLayers, ⟨Z₀⟩ → logit.
* ``quantum_kernel`` – fidelity kernel k(x, x') = |⟨φ(x)|φ(x')⟩|² of an IQP feature map, used in an SVM.
* ``qnn_regressor``  – hybrid QNN: classical input layer → quantum layer → classical head, predicts RUL.

Each experiment also trains classical baselines on the *same* PCA features so the comparison is fair,
and records qubits, circuit depth, parameter counts and wall-clock times. Nothing here claims a
quantum advantage; results are reported as measured.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..classical_models.metrics import classification_metrics, regression_metrics
from ..datasets import cmapss
from ..storage import ExperimentLogger, save_model, save_result

log = logging.getLogger("qadt.quantum")
Progress = Callable[[float, str], None]
QUANTUM_MODELS = ("vqc", "quantum_kernel", "qnn_regressor")


@dataclass
class QuantumConfig:
    model: str = "vqc"
    subset: str = "FD001"
    n_qubits: int = 4
    layers: int = 2
    n_train: int = 300
    epochs: int = 30
    lr: float = 0.05
    seed: int = 42
    device: str = os.environ.get("QADT_QUANTUM_DEVICE", "default.qubit")  # e.g. "qiskit.ibmq" if configured
    shots: int | None = None


# ---------------------------------------------------------------- data


def _prepare(cfg: QuantumConfig):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    data = cmapss.prepare(cmapss.CmapssConfig(subset=cfg.subset, seed=cfg.seed), save=False)
    rng = np.random.default_rng(cfg.seed)
    tr = data.train
    # balanced subsample so the small quantum models see both classes
    pos = tr[tr["fail_within_h"] == 1].index.to_numpy()
    neg = tr[tr["fail_within_h"] == 0].index.to_numpy()
    k = min(cfg.n_train // 2, len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    sample = tr.loc[rng.permutation(idx)]
    pca = PCA(n_components=cfg.n_qubits, random_state=cfg.seed).fit(tr[data.feature_cols])
    scaler = MinMaxScaler((0, np.pi)).fit(pca.transform(tr[data.feature_cols]))

    def enc(df):
        return np.clip(scaler.transform(pca.transform(df[data.feature_cols])), 0, np.pi)

    test = data.test_last
    return {
        "X_train": enc(sample), "y_train_cls": sample["fail_within_h"].to_numpy(),
        "y_train_rul": sample["rul"].to_numpy(float),
        "X_test": enc(test), "y_test_cls": (test["rul_true"] <= data.config.failure_horizon).astype(int).to_numpy(),
        "y_test_rul": test["rul"].to_numpy(float), "rul_cap": data.config.rul_cap,
        "explained_variance": float(pca.explained_variance_ratio_.sum()), "pca": pca, "scaler": scaler,
        "feature_cols": data.feature_cols, "units": test["unit"].to_numpy(),
    }


def _device(cfg: QuantumConfig, wires: int):
    import pennylane as qml

    return qml.device(cfg.device, wires=wires, shots=cfg.shots) if cfg.shots else qml.device(cfg.device, wires=wires)


def _circuit_specs(qnode, *args) -> dict:
    """Depth and gate count after decomposition to device-native gates."""
    import pennylane as qml

    try:
        res = qml.specs(qnode, level="device")(*args).resources
        return {"depth": int(res.depth), "gates": int(res.num_gates), "gate_types": dict(res.gate_types)}
    except Exception as exc:  # specs is informational; never fail the experiment on it
        log.warning("circuit specs unavailable: %s", exc)
        return {"depth": None, "gates": None, "gate_types": {}}


def _platt(decision_train: np.ndarray, y: np.ndarray):
    """Platt scaling: map SVM decision values to probabilities (replaces deprecated SVC(probability=True))."""
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression().fit(decision_train.reshape(-1, 1), y)


# ---------------------------------------------------------------- models


def _vqc(cfg: QuantumConfig, d: dict, progress: Progress, logger: ExperimentLogger) -> dict:
    import pennylane as qml
    import torch

    torch.manual_seed(cfg.seed)
    dev = _device(cfg, cfg.n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop" if cfg.shots is None else "parameter-shift")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(cfg.n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(cfg.n_qubits))
        return qml.expval(qml.PauliZ(0))

    shape = qml.StronglyEntanglingLayers.shape(n_layers=cfg.layers, n_wires=cfg.n_qubits)
    weights = torch.nn.Parameter(0.1 * torch.randn(shape))
    bias = torch.nn.Parameter(torch.zeros(1))
    scale = torch.nn.Parameter(torch.ones(1) * 3)
    opt = torch.optim.Adam([weights, bias, scale], lr=cfg.lr)
    X = torch.tensor(d["X_train"], dtype=torch.float64)
    y = torch.tensor(d["y_train_cls"], dtype=torch.float64)

    def forward(xb):
        return torch.stack([circuit(x, weights) for x in xb]) * scale + bias

    t0 = time.time()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(forward(X[b]), y[b])
            loss.backward()
            opt.step()
        logger.log(epoch=epoch, loss=loss.item())
        progress(0.1 + 0.7 * (epoch + 1) / cfg.epochs, f"VQC epoch {epoch + 1}/{cfg.epochs} loss {loss.item():.3f}")
    train_s = time.time() - t0
    t1 = time.time()
    with torch.no_grad():
        proba = torch.sigmoid(forward(torch.tensor(d["X_test"], dtype=torch.float64))).numpy()
    return {"scores": proba, "train_seconds": train_s, "inference_seconds": time.time() - t1,
            "trainable_params": int(weights.numel() + 2),
            "circuit": _circuit_specs(circuit, X[0], weights.detach()),
            "state": {"weights": weights.detach().numpy(), "bias": float(bias), "scale": float(scale)}}


def _quantum_kernel(cfg: QuantumConfig, d: dict, progress: Progress, logger: ExperimentLogger) -> dict:
    import pennylane as qml
    from sklearn.svm import SVC

    dev = _device(cfg, cfg.n_qubits)

    @qml.qnode(dev)
    def feature_state(x):
        qml.IQPEmbedding(x, wires=range(cfg.n_qubits), n_repeats=cfg.layers)
        return qml.state()

    def states(X):
        return np.array([feature_state(x) for x in X])

    t0 = time.time()
    progress(0.2, "computing quantum feature states (train)")
    s_train = states(d["X_train"])
    # fidelity kernel from exact statevectors (equivalent to the overlap-test circuit on a simulator)
    K_train = np.abs(s_train.conj() @ s_train.T) ** 2
    svc = SVC(kernel="precomputed", class_weight="balanced", random_state=cfg.seed).fit(K_train, d["y_train_cls"])
    platt = _platt(svc.decision_function(K_train), d["y_train_cls"])
    train_s = time.time() - t0
    progress(0.7, "computing quantum kernel (test)")
    t1 = time.time()
    s_test = states(d["X_test"])
    K_test = np.abs(s_test.conj() @ s_train.T) ** 2
    proba = platt.predict_proba(svc.decision_function(K_test).reshape(-1, 1))[:, 1]
    logger.log(event="kernel", train_kernel_mean=float(K_train.mean()))
    return {"scores": proba, "train_seconds": train_s, "inference_seconds": time.time() - t1,
            "trainable_params": int(len(svc.support_)), "circuit": _circuit_specs(feature_state, d["X_train"][0]),
            "kernel_stats": {"mean": float(K_train.mean()), "offdiag_std": float(K_train[~np.eye(len(K_train), dtype=bool)].std())},
            "state": {"svc": svc, "train_states": s_train, "platt": platt}}


def _qnn_regressor(cfg: QuantumConfig, d: dict, progress: Progress, logger: ExperimentLogger) -> dict:
    import pennylane as qml
    import torch

    torch.manual_seed(cfg.seed)
    dev = _device(cfg, cfg.n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop" if cfg.shots is None else "parameter-shift")
    def qlayer(inputs, weights):
        qml.AngleEmbedding(inputs * (np.pi / 2), wires=range(cfg.n_qubits), rotation="Y")  # tanh output → ±π/2
        qml.BasicEntanglerLayers(weights, wires=range(cfg.n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(cfg.n_qubits)]

    layer = qml.qnn.TorchLayer(qlayer, {"weights": (cfg.layers, cfg.n_qubits)})
    net = torch.nn.Sequential(torch.nn.Linear(cfg.n_qubits, cfg.n_qubits), torch.nn.Tanh(),
                              layer, torch.nn.Linear(cfg.n_qubits, 1))
    net = net.double()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    cap = d["rul_cap"]
    X = torch.tensor(d["X_train"], dtype=torch.float64)
    y = torch.tensor(d["y_train_rul"] / cap, dtype=torch.float64)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(X[b]).squeeze(-1), y[b])
            loss.backward()
            opt.step()
        logger.log(epoch=epoch, loss=loss.item())
        progress(0.1 + 0.7 * (epoch + 1) / cfg.epochs, f"QNN epoch {epoch + 1}/{cfg.epochs} loss {loss.item():.4f}")
    train_s = time.time() - t0
    t1 = time.time()
    with torch.no_grad():
        pred = np.clip(net(torch.tensor(d["X_test"], dtype=torch.float64)).squeeze(-1).numpy() * cap, 0, None)
    return {"rul_pred": pred, "train_seconds": train_s, "inference_seconds": time.time() - t1,
            "trainable_params": int(sum(p.numel() for p in net.parameters())),
            "circuit": _circuit_specs(qlayer, X[0], torch.zeros(cfg.layers, cfg.n_qubits, dtype=torch.float64)),
            "state": {"state_dict": {k: v.detach().numpy() for k, v in net.state_dict().items()}}}


# ---------------------------------------------------------------- classical baselines on identical inputs


def _classical_baselines(d: dict, seed: int, regression: bool) -> list[dict]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.svm import SVC, SVR

    out = []
    if regression:
        models = {"ridge": Ridge(), "svr_rbf": SVR(C=50),
                  "mlp_small": MLPRegressor(hidden_layer_sizes=(8,), max_iter=2000, random_state=seed)}
        for name, m in models.items():
            t0 = time.time()
            m.fit(d["X_train"], d["y_train_rul"])
            pred = np.clip(m.predict(d["X_test"]), 0, None)
            out.append({"model": name, "train_seconds": time.time() - t0,
                        "metrics": regression_metrics(d["y_test_rul"], pred)})
    else:
        models = {"logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
                  "svc_rbf": CalibratedClassifierCV(SVC(class_weight="balanced", random_state=seed), ensemble=False),
                  "mlp_small": MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=seed)}
        for name, m in models.items():
            t0 = time.time()
            m.fit(d["X_train"], d["y_train_cls"])
            out.append({"model": name, "train_seconds": time.time() - t0,
                        "metrics": classification_metrics(d["y_test_cls"], m.predict_proba(d["X_test"])[:, 1])})
    return out


def run_experiment(cfg: QuantumConfig | None = None, progress: Progress | None = None, save: bool = True) -> dict:
    cfg = cfg or QuantumConfig()
    if cfg.model not in QUANTUM_MODELS:
        raise ValueError(f"model must be one of {QUANTUM_MODELS}")
    if not 2 <= cfg.n_qubits <= 8:
        raise ValueError("n_qubits must be between 2 and 8 for simulator experiments")
    progress = progress or (lambda f, m: None)
    run = f"q_{cfg.model}_{cfg.subset}_{cfg.n_qubits}q"
    logger = ExperimentLogger(run, config=cfg.__dict__)
    progress(0.03, "preparing PCA-encoded data")
    d = _prepare(cfg)
    fn = {"vqc": _vqc, "quantum_kernel": _quantum_kernel, "qnn_regressor": _qnn_regressor}[cfg.model]
    q = fn(cfg, d, progress, logger)
    regression = cfg.model == "qnn_regressor"
    if regression:
        q_metrics = regression_metrics(d["y_test_rul"], q["rul_pred"])
    else:
        q_metrics = classification_metrics(d["y_test_cls"], q["scores"])
    progress(0.9, "training classical baselines on the same features")
    baselines = _classical_baselines(d, cfg.seed, regression)
    key = "rmse" if regression else "roc_auc"
    best_classical = (min if regression else max)(baselines, key=lambda b: b["metrics"][key] or 0)
    result = {
        "run": run, "model": cfg.model, "task": "rul_regression" if regression else "failure_within_horizon",
        "config": {k: v for k, v in cfg.__dict__.items()},
        "quantum": {"metrics": q_metrics, "qubits": cfg.n_qubits, "layers": cfg.layers,
                    "circuit_depth": q["circuit"]["depth"], "gate_count": q["circuit"]["gates"],
                    "gate_types": q["circuit"]["gate_types"],
                    "trainable_params": q["trainable_params"], "train_seconds": round(q["train_seconds"], 2),
                    "inference_seconds": round(q["inference_seconds"], 3), "device": cfg.device,
                    "shots": cfg.shots, **({"kernel_stats": q["kernel_stats"]} if "kernel_stats" in q else {})},
        "classical_baselines": baselines,
        "comparison": {
            "metric": key, "quantum": q_metrics[key], "best_classical_model": best_classical["model"],
            "best_classical": best_classical["metrics"][key],
            "quantum_better": (q_metrics[key] < best_classical["metrics"][key]) if regression
            else ((q_metrics[key] or 0) > (best_classical["metrics"][key] or 0)),
            "note": "Single run on a statevector simulator with the same PCA inputs for all models. "
                    "A difference here is not evidence of quantum advantage.",
        },
        "data": {"pca_components": cfg.n_qubits, "pca_explained_variance": d["explained_variance"],
                 "n_train": int(len(d["X_train"])), "n_test": int(len(d["X_test"])),
                 "test_positives": int(d["y_test_cls"].sum())},
        "test_predictions": ([{"unit": int(u), "rul_true": float(t), "rul_pred": float(p)} for u, t, p in
                              zip(d["units"], d["y_test_rul"], q["rul_pred"])] if regression else
                             [{"unit": int(u), "label": int(t), "score": float(s)} for u, t, s in
                              zip(d["units"], d["y_test_cls"], q["scores"])]),
    }
    if save:
        save_model({"config": cfg.__dict__, "state": q["state"], "pca": d["pca"], "scaler": d["scaler"],
                    "feature_cols": d["feature_cols"]}, "quantum_ai", run, config=cfg.__dict__, metrics=q_metrics)
        save_result("comparison_reports", f"quantum_vs_classical_{run}", result)
    progress(1.0, "done")
    return result
