"""IMS bearing anomaly detection: Isolation Forest, One-Class SVM, Autoencoder, LSTM Autoencoder,
plus a cross-test supervised classifier when more than one IMS test is available.

All unsupervised detectors are fitted on the healthy baseline period only. Scores are oriented so that
higher = more anomalous; the alarm threshold is the ``threshold_quantile`` of baseline scores.
Evaluation uses the labelling assumption documented in :mod:`backend.datasets.ims`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..datasets import ims
from ..storage import ExperimentLogger, save_model
from ..storage.artifacts import load_model
from .metrics import classification_metrics

log = logging.getLogger("qadt.anomaly")

DETECTORS = ("isolation_forest", "one_class_svm", "autoencoder", "lstm_autoencoder")
Progress = Callable[[float, str], None]


@dataclass
class AnomalyModel:
    name: str
    test_id: str
    estimator: Any
    feature_cols: list[str]
    threshold: float = 0.0
    window: int = 8
    metrics: dict = field(default_factory=dict)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Anomaly score per row of a prepared IMS feature frame (higher = more anomalous)."""
        X = df[self.feature_cols].to_numpy(np.float32)
        if self.name == "isolation_forest":
            return -self.estimator.score_samples(X)
        if self.name == "one_class_svm":
            return -self.estimator.decision_function(X)
        import torch

        net = self.estimator
        net.eval()
        with torch.no_grad():
            if self.name == "autoencoder":
                t = torch.from_numpy(X)
                return ((net(t) - t) ** 2).mean(1).numpy()
            scores = np.empty(len(df))
            for _, idx in df.groupby("bearing").groups.items():
                pos = df.index.get_indexer(idx)
                order = pos[np.argsort(df["file_index"].to_numpy()[pos])]
                seqs = _windows(X[order], self.window)
                t = torch.from_numpy(seqs)
                scores[order] = ((net(t) - t) ** 2).mean((1, 2)).numpy()
            return scores


def _windows(X: np.ndarray, w: int) -> np.ndarray:
    """Window ending at each row; the first rows are front-padded with the first snapshot."""
    padded = np.vstack([np.repeat(X[:1], w - 1, axis=0), X])
    return np.stack([padded[i:i + w] for i in range(len(X))]).astype(np.float32)


def _train_torch_ae(net, data: np.ndarray, epochs: int, seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    t = torch.from_numpy(data.astype(np.float32))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(t))
        for i in range(0, len(t), 64):
            b = t[perm[i:i + 64]]
            opt.zero_grad()
            loss = ((net(b) - b) ** 2).mean()
            loss.backward()
            opt.step()


def detection_lead_time(df: pd.DataFrame, scores: np.ndarray, threshold: float, consecutive: int = 3) -> dict:
    """Hours between the first sustained alarm on each failed bearing and the end of the test."""
    out = {}
    df = df.assign(_score=scores)
    end = df["timestamp"].max()
    for bearing, g in df[df["failed_bearing"] == 1].groupby("bearing"):
        g = g[~g["is_baseline"]].sort_values("file_index")
        alarm = (g["_score"] >= threshold).to_numpy().astype(int)
        run = np.convolve(alarm, np.ones(consecutive, int), "valid") if len(alarm) >= consecutive else np.array([])
        hits = np.flatnonzero(run == consecutive)
        if hits.size:
            first = g["timestamp"].iloc[hits[0]]
            out[str(bearing)] = {"first_alarm": str(first),
                                 "lead_time_hours": round((end - first).total_seconds() / 3600, 2),
                                 "life_fraction_at_alarm": round(float(g["life_fraction"].iloc[hits[0]]), 4)}
        else:
            out[str(bearing)] = {"first_alarm": None, "lead_time_hours": 0.0, "life_fraction_at_alarm": None}
    return out


def train_detector(name: str, test_id: str = "2nd_test", config: ims.ImsConfig | None = None,
                   threshold_quantile: float = 0.99, epochs: int = 60, seed: int = 42,
                   progress: Progress | None = None, save: bool = True) -> dict:
    if name not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}")
    progress = progress or (lambda f, m: None)
    t0 = time.time()
    progress(0.05, f"extracting IMS {test_id} features")
    data = ims.prepare_anomaly(test_id, config)
    cols = data.feature_cols
    base_df = data.features[data.features["is_baseline"]]
    X = base_df[cols].to_numpy(np.float32)
    logger = ExperimentLogger(f"anomaly_{name}_{test_id}", config={"model": name, "test": test_id})
    progress(0.3, f"training {name} on {len(X)} healthy snapshots")

    if name == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        est = IsolationForest(n_estimators=300, contamination="auto", random_state=seed).fit(X)
    elif name == "one_class_svm":
        from sklearn.svm import OneClassSVM
        est = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X)
    elif name == "autoencoder":
        from .nets import DenseAutoencoder
        est = DenseAutoencoder(len(cols))
        _train_torch_ae(est, X, epochs, seed)
    else:
        from .nets import LstmAutoencoder
        est = LstmAutoencoder(len(cols))
        seqs = np.concatenate([_windows(g.sort_values("file_index")[cols].to_numpy(np.float32), 8)
                               for _, g in base_df.groupby("bearing")])
        _train_torch_ae(est, seqs, max(epochs // 2, 10), seed)

    model = AnomalyModel(name=name, test_id=test_id, estimator=est, feature_cols=cols)
    progress(0.8, "scoring and evaluating")
    all_scores = model.score(data.features)
    model.threshold = float(np.quantile(all_scores[data.features["is_baseline"].to_numpy()], threshold_quantile))
    ev = data.features["label"].notna().to_numpy()
    metrics = classification_metrics(data.features.loc[ev, "label"], all_scores[ev], model.threshold)
    lead = detection_lead_time(data.features, all_scores, model.threshold)
    result = {
        "model": name, "test": test_id, "metrics": metrics, "detection": lead,
        "threshold": model.threshold, "threshold_quantile": threshold_quantile,
        "train_rows": int(len(X)), "train_seconds": round(time.time() - t0, 2),
        "labelling": {"baseline_fraction": data.config.baseline_fraction,
                      "degraded_fraction": data.config.degraded_fraction,
                      "note": "pseudo-labels from known failed bearings; see backend/datasets/ims.py"},
        "score_trace": (data.features.assign(score=all_scores)
                        .groupby(["file_index", "bearing"])["score"].mean().reset_index()
                        .iloc[::max(1, len(data.features) // 2000)].round(5).to_dict(orient="records")),
    }
    model.metrics = {"metrics": metrics, "detection": lead}
    logger.log(event="evaluation", **{k: v for k, v in metrics.items() if not isinstance(v, dict)})
    if save:
        save_model(model, "classical_ai", f"anomaly_{name}_{test_id}", config={"model": name, "test": test_id},
                   metrics={k: metrics[k] for k in ("roc_auc", "f1", "precision", "recall", "false_alarm_rate")})
    progress(1.0, "done")
    return result


def train_supervised_cross_test(train_tests: list[str], eval_test: str, config: ims.ImsConfig | None = None,
                                seed: int = 42, save: bool = True) -> dict:
    """Random-forest classifier trained on pseudo-labelled tests, evaluated on a different test."""
    from sklearn.ensemble import RandomForestClassifier

    frames = [ims.prepare_anomaly(t, config, save=False) for t in train_tests]
    cols = frames[0].feature_cols
    tr = pd.concat([f.eval_df for f in frames])
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(tr[cols], tr["label"].astype(int))
    ev = ims.prepare_anomaly(eval_test, config, save=False)
    proba = clf.predict_proba(ev.eval_df[cols])[:, 1]
    metrics = classification_metrics(ev.eval_df["label"].astype(int), proba, 0.5)
    all_p = clf.predict_proba(ev.features[cols])[:, 1]
    result = {"model": "random_forest_supervised", "train_tests": train_tests, "test": eval_test,
              "metrics": metrics, "detection": detection_lead_time(ev.features, all_p, 0.5)}
    if save:
        save_model(AnomalyModel("random_forest_supervised", eval_test, clf, cols, 0.5, metrics=result),
                   "classical_ai", f"anomaly_supervised_{eval_test}", metrics=metrics)
    return result


def load_detector(name: str, test_id: str = "2nd_test") -> AnomalyModel:
    return load_model("classical_ai", f"anomaly_{name}_{test_id}")
