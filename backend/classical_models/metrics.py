"""Evaluation metrics shared by every model family."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score: late predictions (pred > true) are penalised more heavily."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(skm.mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(skm.mean_squared_error(y_true, y_pred))),
        "r2": float(skm.r2_score(y_true, y_pred)),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """``scores`` are probabilities / anomaly scores (higher = more likely positive)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    pred = (scores >= threshold).astype(int)
    neg = y_true == 0
    out = {
        "accuracy": float(skm.accuracy_score(y_true, pred)),
        "precision": float(skm.precision_score(y_true, pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, pred, zero_division=0)),
        "false_alarm_rate": float(pred[neg].mean()) if neg.any() else None,
        "positives": int(y_true.sum()), "n": int(len(y_true)), "threshold": float(threshold),
    }
    out["roc_auc"] = float(skm.roc_auc_score(y_true, scores)) if 0 < y_true.sum() < len(y_true) else None
    out["brier"] = float(np.mean((np.clip(scores, 0, 1) - y_true) ** 2)) if scores.max() <= 1 else None
    return out
