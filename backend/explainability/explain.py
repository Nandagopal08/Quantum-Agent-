"""Explainability for RUL models and bearing anomaly detectors.

* SHAP (TreeExplainer) for tree models; occlusion attributions for sequence models
  (feature group replaced by its healthy-reference value across the window → change in prediction).
* Model-native feature importance where available; permutation importance on validation units.
* Counterfactual explanations: the smallest set of sensor shifts toward healthy-reference values that
  raises predicted RUL to a target, reported in original sensor units.
* Uncertainty: calibrated σ, 90 % split-conformal interval, PoF; tree-ensemble spread for random forests.
* Bearings: per-feature deviation from the healthy baseline (z-scores) for the latest snapshot.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from ..classical_models.rul import RulModel
from ..datasets import cmapss
from ..storage import paths, save_figure, save_result

log = logging.getLogger("qadt.explain")


def _sensor_of(feature: str) -> str | None:
    base = feature.split("_r")[0]
    return base if base in cmapss.SENSOR_COLS else None


def _groups(features: list[str]) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {}
    for f in features:
        s = _sensor_of(f)
        g.setdefault(s or f, []).append(f)
    return g


def _healthy_reference(data: cmapss.CmapssData) -> pd.Series:
    healthy = data.train[data.train["rul"] >= data.config.rul_cap]
    return healthy[data.feature_cols].mean()


def _unit_frame(data: cmapss.CmapssData, unit: int) -> pd.DataFrame:
    g = data.test[data.test["unit"] == unit].sort_values("cycle")
    if g.empty:
        raise ValueError(f"unit {unit} not in the test set")
    return g


def _predict_modified(model: RulModel, frame: pd.DataFrame, changes: dict[str, float]) -> float:
    """Predict RUL at the last cycle after setting the given feature columns to new values
    (across the whole window for sequence models)."""
    f = frame.copy()
    for col, val in changes.items():
        if model.kind == "sequence":
            f.loc[f.index[-model.window:], col] = val
        else:
            f.loc[f.index[-1], col] = val
    return float(model.predict_last(f)[0])


# ---------------------------------------------------------------- global


def shap_global(model: RulModel, data: cmapss.CmapssData, max_rows: int = 300) -> dict | None:
    if model.kind != "tabular":
        return None
    import shap

    X = data.test_last[model.feature_cols]
    bg = data.train[model.feature_cols].sample(min(max_rows, len(data.train)), random_state=0)
    explainer = shap.TreeExplainer(model.estimator, bg if model.name == "random_forest" else None)
    values = np.asarray(explainer.shap_values(X))
    mean_abs = np.abs(values).mean(0)
    order = np.argsort(mean_abs)[::-1]
    return {"method": "shap_tree", "expected_value": float(np.ravel(explainer.expected_value)[0]),
            "importance": [{"feature": model.feature_cols[i], "mean_abs_shap": float(mean_abs[i])} for i in order[:20]],
            "values": values, "X": X}


def native_importance(model: RulModel) -> list[dict] | None:
    imp = getattr(model.estimator, "feature_importances_", None)
    if imp is None:
        return None
    imp = np.asarray(imp, float)
    imp = imp / imp.sum() if imp.sum() else imp
    order = np.argsort(imp)[::-1][:20]
    return [{"feature": model.feature_cols[i], "importance": float(imp[i])} for i in order]


def permutation_importance(model: RulModel, data: cmapss.CmapssData, repeats: int = 3, seed: int = 0) -> list[dict]:
    """Increase in validation RMSE (last cycle of each window / row) when a sensor group is shuffled."""
    rng = np.random.default_rng(seed)
    val = data.val
    y = val["rul"].to_numpy()

    def rmse(df):
        return float(np.sqrt(np.mean((model.predict_frame(df) - y) ** 2)))

    base = rmse(val)
    out = []
    for name, cols in _groups(model.feature_cols).items():
        deltas = []
        for _ in range(repeats):
            shuffled = val.copy()
            perm = rng.permutation(len(val))
            shuffled[cols] = val[cols].to_numpy()[perm]
            deltas.append(rmse(shuffled) - base)
        out.append({"feature_group": name, "sensor_name": cmapss.SENSOR_NAMES.get(name, name),
                    "rmse_increase": float(np.mean(deltas)), "std": float(np.std(deltas))})
    return sorted(out, key=lambda r: -r["rmse_increase"])


# ---------------------------------------------------------------- local


def local_attribution(model: RulModel, data: cmapss.CmapssData, unit: int) -> dict:
    frame = _unit_frame(data, unit)
    pred = float(model.predict_last(frame)[0])
    if model.kind == "tabular":
        import shap

        explainer = shap.TreeExplainer(model.estimator)
        row = frame[model.feature_cols].iloc[[-1]]
        vals = np.asarray(explainer.shap_values(row))[0]
        grouped: dict[str, float] = {}
        for f, v in zip(model.feature_cols, vals):
            grouped[_sensor_of(f) or f] = grouped.get(_sensor_of(f) or f, 0.0) + float(v)
        method, base = "shap_tree", float(np.ravel(explainer.expected_value)[0])
    else:
        ref = _healthy_reference(data)
        grouped = {name: pred - _predict_modified(model, frame, {c: ref[c] for c in cols})
                   for name, cols in _groups(model.feature_cols).items()}
        method, base = "occlusion_to_healthy_reference", None
    items = sorted(grouped.items(), key=lambda kv: -abs(kv[1]))
    return {"unit": unit, "prediction": pred, "method": method, "base_value": base,
            "contributions": [{"feature": k, "sensor_name": cmapss.SENSOR_NAMES.get(k, k), "contribution": v}
                              for k, v in items[:12]],
            "note": ("SHAP values sum (with the base value) to the prediction." if method == "shap_tree" else
                     "Contribution = prediction − prediction with that sensor set to its healthy-reference value; "
                     "negative values mean the sensor's current reading lowers predicted RUL.")}


def counterfactual_explanation(model: RulModel, data: cmapss.CmapssData, unit: int,
                               target_rul: float | None = None, max_changes: int = 4) -> dict:
    """Greedy search over sensor groups: shift the most harmful sensors toward healthy values until the
    predicted RUL reaches ``target_rul``. Reports the changes in normalised and original units."""
    frame = _unit_frame(data, unit)
    current = float(model.predict_last(frame)[0])
    target = target_rul if target_rul is not None else min(current + 30, model.rul_cap)
    ref = _healthy_reference(data)
    groups = {k: v for k, v in _groups(model.feature_cols).items() if k in cmapss.SENSOR_COLS}
    cond = int(frame["op_condition"].iloc[-1])
    sc = data.scaler.get(cond) or next(iter(data.scaler.values()))
    applied: dict[str, float] = {}
    steps = []
    pred = current
    remaining = dict(groups)
    while pred < target and remaining and len(steps) < max_changes:
        trial = {name: _predict_modified(model, frame, {**applied, **{c: ref[c] for c in cols}})
                 for name, cols in remaining.items()}
        best = max(trial, key=trial.get)
        if trial[best] <= pred + 1e-6:
            break
        cols = remaining.pop(best)
        applied.update({c: ref[c] for c in cols})
        now_z = float(frame[best].iloc[-1])
        to_z = float(ref[best])
        steps.append({
            "sensor": best, "sensor_name": cmapss.SENSOR_NAMES[best],
            "current_value": now_z * sc["std"][best] + sc["mean"][best],
            "counterfactual_value": to_z * sc["std"][best] + sc["mean"][best],
            "current_normalised": now_z, "counterfactual_normalised": to_z,
            "predicted_rul_after": trial[best],
        })
        pred = trial[best]
    return {"unit": unit, "current_rul": current, "target_rul": target, "achieved_rul": pred,
            "target_reached": pred >= target, "changes": steps,
            "interpretation": (f"If {', '.join(s['sensor_name'] for s in steps)} returned to healthy-engine levels, "
                               f"predicted RUL would rise from {current:.1f} to {pred:.1f} cycles." if steps else
                               "No single sensor shift toward healthy values increases the prediction."),
            "note": "Counterfactual sensor values are hypothetical, not measurements."}


def uncertainty(model: RulModel, data: cmapss.CmapssData, unit: int) -> dict:
    frame = _unit_frame(data, unit)
    pred = float(model.predict_last(frame)[0])
    lo, hi = model.interval([pred])
    out = {"unit": unit, "prediction": pred, "sigma": model.sigma, "interval_90": [float(lo[0]), float(hi[0])],
           "pof": float(model.failure_probability([pred])[0]), "horizon": model.failure_horizon,
           "method": "split-conformal 90% interval on validation units; Gaussian σ from validation residuals"}
    if model.name == "random_forest":
        row = frame[model.feature_cols].iloc[[-1]].to_numpy()
        trees = np.array([t.predict(row)[0] for t in model.estimator.estimators_])
        out["ensemble_std"] = float(trees.std())
        out["ensemble_p10_p90"] = [float(np.quantile(trees, 0.1)), float(np.quantile(trees, 0.9))]
    return out


# ---------------------------------------------------------------- bearings


def bearing_deviation(features: pd.DataFrame, feature_cols: list[str], bearing: int, file_index: int | None = None,
                      top: int = 8) -> dict:
    g = features[features["bearing"] == bearing].sort_values("file_index")
    row = g.iloc[-1] if file_index is None else g[g["file_index"] == file_index].iloc[0]
    devs = sorted(((c.removeprefix("z_"), float(row[c])) for c in feature_cols), key=lambda kv: -abs(kv[1]))
    return {"bearing": bearing, "file_index": int(row["file_index"]), "timestamp": str(row["timestamp"]),
            "method": "z-score vs healthy baseline (log-scaled features)",
            "top_deviations": [{"feature": k, "z": v} for k, v in devs[:top]]}


# ---------------------------------------------------------------- report


def explain_rul_model(model: RulModel, data: cmapss.CmapssData, unit: int | None = None, save: bool = True,
                      progress: Callable[[float, str], None] | None = None) -> dict:
    progress = progress or (lambda f, m: None)
    unit = unit if unit is not None else int(data.test_last.sort_values("rul_true")["unit"].iloc[0])
    progress(0.1, "global attributions")
    shap_g = shap_global(model, data)
    progress(0.4, "permutation importance")
    report = {
        "model": model.name, "subset": data.config.subset, "unit": unit,
        "shap_global": None if shap_g is None else {k: shap_g[k] for k in ("method", "expected_value", "importance")},
        "native_importance": native_importance(model),
        "permutation_importance": permutation_importance(model, data),
    }
    progress(0.7, "local attribution and counterfactual")
    report["local"] = local_attribution(model, data, unit)
    report["counterfactual"] = counterfactual_explanation(model, data, unit)
    report["uncertainty"] = uncertainty(model, data, unit)
    if save:
        name = f"rul_{model.name}_{data.config.subset}"
        save_result("shap", name, report)
        save_result("counterfactuals", f"{name}_unit{unit}", report["counterfactual"])
        if shap_g is not None:
            _shap_plot(shap_g, name)
            report["shap_plot"] = str(paths.area("shap") / f"{name}_summary.png")
    progress(1.0, "done")
    return report


def _shap_plot(shap_g: dict, name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = shap_g["importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([i["feature"] for i in imp], [i["mean_abs_shap"] for i in imp], color="#4f6bd8")
    ax.set_xlabel("mean |SHAP value| (cycles of RUL)")
    ax.set_title(f"SHAP feature importance — {name}")
    fig.tight_layout()
    save_figure(fig, f"{name}_summary", area="shap")
    plt.close(fig)
