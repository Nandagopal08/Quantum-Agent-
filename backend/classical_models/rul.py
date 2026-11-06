"""Remaining-useful-life models for C-MAPSS.

Tabular models (Random Forest, XGBoost, LightGBM, CatBoost) use per-cycle features; sequence models
(LSTM, GRU, CNN-LSTM, Transformer) use sliding windows. All are wrapped in :class:`RulModel`, which
adds a calibrated uncertainty so the risk engine can turn an RUL prediction into a probability of
failure:

* ``sigma``   – std of validation residuals (Gaussian predictive spread);
* ``q90``     – split-conformal 90 % absolute-error quantile on validation units (prediction interval).

P(failure within h cycles) = Φ((h − RUL_pred) / sigma).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..datasets import cmapss
from ..storage import CheckpointManager, ExperimentLogger, save_model
from ..storage.artifacts import load_model
from .metrics import classification_metrics, regression_metrics

log = logging.getLogger("qadt.rul")

TABULAR = ("random_forest", "xgboost", "lightgbm", "catboost")
SEQUENCE = ("lstm", "gru", "cnn_lstm", "transformer")
ALL_MODELS = TABULAR + SEQUENCE

Progress = Callable[[float, str], None]


@dataclass
class RulModel:
    name: str
    subset: str
    kind: str                      # "tabular" | "sequence"
    feature_cols: list[str]
    window: int
    estimator: Any                 # sklearn-style regressor or torch module
    sigma: float = 0.0
    q90: float = 0.0
    rul_cap: int = 125
    failure_horizon: int = 30
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""

    # ---- inference
    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Predict RUL for every row of ``df`` (sequence models use the window ending at that row)."""
        if self.kind == "tabular":
            return np.clip(self.estimator.predict(df[self.feature_cols]), 0, None)
        X, _, _ = cmapss.make_sequences(df, self.feature_cols, self.window)
        out = _torch_predict(self.estimator, X)
        # make_sequences emits windows ending at rows window..n; earlier rows reuse the first window
        preds, i = np.empty(len(df)), 0
        for _, g in df.groupby("unit", sort=True):
            n_win = max(len(g) - self.window + 1, 1)
            p = out[i:i + n_win]
            preds[np.flatnonzero(df["unit"].to_numpy() == g["unit"].iloc[0])] = np.concatenate(
                [np.repeat(p[:1], len(g) - len(p)), p])
            i += n_win
        return np.clip(preds, 0, None)

    def predict_last(self, df: pd.DataFrame) -> np.ndarray:
        """One prediction per unit at its last available cycle."""
        if self.kind == "tabular":
            last = df.groupby("unit", sort=True).tail(1)
            return np.clip(self.estimator.predict(last[self.feature_cols]), 0, None)
        X, _, _ = cmapss.make_sequences(df, self.feature_cols, self.window, last_only=True)
        return np.clip(_torch_predict(self.estimator, X), 0, None)

    def failure_probability(self, rul_pred, horizon: int | None = None) -> np.ndarray:
        h = self.failure_horizon if horizon is None else horizon
        return norm.cdf((h - np.asarray(rul_pred, float)) / max(self.sigma, 1e-6))

    def interval(self, rul_pred) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(rul_pred, float)
        return np.clip(r - self.q90, 0, None), r + self.q90


# ---------------------------------------------------------------- tabular


def _make_tabular(name: str, seed: int):
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, n_jobs=-1, early_stopping_rounds=40)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, random_seed=seed, verbose=False,
                                 od_type="Iter", od_wait=50, thread_count=-1)
    raise ValueError(name)


def _fit_tabular(name: str, data: cmapss.CmapssData, seed: int):
    est = _make_tabular(name, seed)
    X, y = data.train[data.feature_cols], data.train["rul"]
    Xv, yv = data.val[data.feature_cols], data.val["rul"]
    if name == "xgboost":
        est.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    elif name == "lightgbm":
        from lightgbm import early_stopping
        est.fit(X, y, eval_set=[(Xv, yv)], callbacks=[early_stopping(50, verbose=False)])
    elif name == "catboost":
        est.fit(X, y, eval_set=(Xv, yv))
    else:
        est.fit(X, y)
    return est


# ---------------------------------------------------------------- sequence (PyTorch)


def _torch_predict(net, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    import torch

    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(net(torch.from_numpy(X[i:i + batch])).numpy())
    return np.concatenate(outs) if outs else np.empty(0)


def _fit_sequence(name: str, data: cmapss.CmapssData, seed: int, epochs: int, run_name: str,
                  progress: Progress | None, logger: ExperimentLogger):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    w = data.config.window
    X, y, _ = cmapss.make_sequences(data.train, data.feature_cols, w)
    Xv, yv, _ = cmapss.make_sequences(data.val, data.feature_cols, w, last_only=False)
    from .nets import build_rul_net

    net = build_rul_net(name, X.shape[2], w)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    loss_fn = torch.nn.MSELoss()

    ckpt = CheckpointManager(run_name, category="classical_ai", keep_last=2, every_n_steps=1, every_seconds=None)
    state, info = ckpt.load_latest()
    start, best, best_state, bad = 0, np.inf, None, 0
    if state:
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optim"])
        start, best, best_state, bad = info["step"] + 1, state["best"], state["best_state"], state["bad"]

    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    patience = 6
    for epoch in range(start, epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        val_rmse = float(np.sqrt(np.mean((_torch_predict(net, Xv) - yv) ** 2)))
        sched.step(val_rmse)
        if val_rmse < best - 1e-3:
            best, bad = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        train_rmse = float(np.sqrt(total / len(perm)))
        logger.log(epoch=epoch, train_rmse=train_rmse, val_rmse=val_rmse)
        ckpt.save({"model": net.state_dict(), "optim": opt.state_dict(), "best": best,
                   "best_state": best_state, "bad": bad}, epoch, metrics={"val_rmse": val_rmse})
        if progress:
            progress((epoch + 1) / epochs, f"epoch {epoch + 1}/{epochs} val RMSE {val_rmse:.2f}")
        if bad >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


# ---------------------------------------------------------------- train / evaluate


def evaluate(model: RulModel, data: cmapss.CmapssData) -> dict:
    """Official evaluation: one prediction per test engine at its last cycle vs RUL_FDxxx.txt."""
    y_true = data.test_last["rul"].to_numpy()          # capped, the standard convention
    y_true_raw = data.test_last["rul_true"].to_numpy()
    pred = model.predict_last(data.test)
    pof = model.failure_probability(pred)
    lo, hi = model.interval(pred)
    fail_label = (y_true_raw <= model.failure_horizon).astype(int)
    return {
        "test": regression_metrics(y_true, pred),
        "test_uncapped": regression_metrics(y_true_raw, pred),
        "failure_classification": classification_metrics(fail_label, pof, 0.5),
        "interval_coverage_90": float(np.mean((y_true >= lo) & (y_true <= hi))),
        "predictions": [{"unit": int(u), "rul_true": float(t), "rul_pred": float(p), "pof": float(f)}
                        for u, t, p, f in zip(data.test_last["unit"], y_true, pred, pof)],
    }


def train_rul_model(name: str, subset: str = "FD001", epochs: int = 40, seed: int = 42,
                    progress: Progress | None = None, save: bool = True) -> dict:
    if name not in ALL_MODELS:
        raise ValueError(f"model must be one of {ALL_MODELS}")
    t0 = time.time()
    progress = progress or (lambda f, m: None)
    progress(0.02, "preparing C-MAPSS data")
    data = cmapss.prepare(cmapss.CmapssConfig(subset=subset, seed=seed), save=False)
    run_name = f"rul_{name}_{subset}"
    logger = ExperimentLogger(run_name, config={"model": name, "subset": subset, "epochs": epochs, "seed": seed})
    progress(0.05, f"training {name}")
    if name in TABULAR:
        est, kind = _fit_tabular(name, data, seed), "tabular"
    else:
        est, kind = _fit_sequence(name, data, seed, epochs, run_name, progress, logger), "sequence"
    model = RulModel(name=name, subset=subset, kind=kind, feature_cols=data.feature_cols,
                     window=data.config.window, estimator=est, rul_cap=data.config.rul_cap,
                     failure_horizon=data.config.failure_horizon)
    # calibrate uncertainty on validation units (last-cycle-of-each-window points)
    val_pred = model.predict_frame(data.val)
    resid = data.val["rul"].to_numpy() - val_pred
    model.sigma = float(np.std(resid))
    model.q90 = float(np.quantile(np.abs(resid), 0.9))
    progress(0.95, "evaluating on official test set")
    result = evaluate(model, data)
    result["validation"] = regression_metrics(data.val["rul"], val_pred)
    result.update({"model": name, "subset": subset, "kind": kind, "sigma": model.sigma, "q90": model.q90,
                   "train_seconds": round(time.time() - t0, 2), "n_features": len(data.feature_cols),
                   "train_units": int(data.train["unit"].nunique()), "val_units": int(data.val["unit"].nunique())})
    model.metrics = {k: result[k] for k in ("test", "validation", "failure_classification")}
    model.trained_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    logger.log(event="evaluation", **{f"test_{k}": v for k, v in result["test"].items()})
    if save:
        save_model(model, "classical_ai", run_name, config={"model": name, "subset": subset, "epochs": epochs},
                   metrics={**result["test"], "val_rmse": result["validation"]["rmse"],
                            "roc_auc": result["failure_classification"]["roc_auc"]})
    progress(1.0, "done")
    log.info("%s %s: test RMSE %.2f, NASA score %.1f", name, subset, result["test"]["rmse"], result["test"]["nasa_score"])
    return result


def load_rul_model(name: str, subset: str = "FD001") -> RulModel:
    return load_model("classical_ai", f"rul_{name}_{subset}")
