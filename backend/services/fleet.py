"""Fleet service: resolves asset IDs to data, trained models, predictions, risk and digital twins.

Asset IDs
* ``FD001-test-unit34``            C-MAPSS test engine 34 (official test set) — turbofan
* ``IMS-2nd_test-bearing1``        IMS bearing 1 of the second test — bearing

Bearings are evaluated "as of" a snapshot index (default: the last one), so the dashboard can replay
history and see how risk evolved.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..classical_models.anomaly import load_detector
from ..classical_models.rul import ALL_MODELS, RulModel, load_rul_model
from ..datasets import cmapss, ims
from ..digital_twin.twin import BearingTwin, SimulationParams, TurbofanTwin
from ..risk.engine import assess_turbofan
from ..storage import list_models

TURBOFAN_RE = re.compile(r"^(FD00[1-4])-(test|train)-unit(\d+)$")
BEARING_RE = re.compile(r"^IMS-(\w+_test)-bearing(\d)$")
_lock = threading.RLock()


class AssetNotFound(ValueError):
    pass


class ModelNotTrained(RuntimeError):
    pass


@dataclass
class ParsedAsset:
    asset_id: str
    kind: str
    subset: str | None = None
    split: str | None = None
    unit: int | None = None
    test: str | None = None
    bearing: int | None = None


def parse_asset(asset_id: str) -> ParsedAsset:
    if m := TURBOFAN_RE.match(asset_id):
        return ParsedAsset(asset_id, "turbofan", subset=m[1], split=m[2], unit=int(m[3]))
    if m := BEARING_RE.match(asset_id):
        return ParsedAsset(asset_id, "bearing", test=m[1], bearing=int(m[2]))
    raise AssetNotFound(f"unrecognised asset id {asset_id!r}")


# ---------------------------------------------------------------- cached resources


@lru_cache(maxsize=4)
def cmapss_data(subset: str) -> cmapss.CmapssData:
    with _lock:
        return cmapss.prepare(cmapss.CmapssConfig(subset=subset), save=False)


@lru_cache(maxsize=16)
def rul_model(name: str, subset: str) -> RulModel:
    try:
        return load_rul_model(name, subset)
    except FileNotFoundError:
        raise ModelNotTrained(f"RUL model {name} for {subset} has not been trained yet") from None


@lru_cache(maxsize=8)
def ims_features(test: str):
    return ims.prepare_anomaly(test, save=False)


@lru_cache(maxsize=16)
def detector(name: str, test: str):
    try:
        return load_detector(name, test)
    except FileNotFoundError:
        raise ModelNotTrained(f"anomaly detector {name} for {test} has not been trained yet") from None


def clear_caches() -> None:
    for f in (cmapss_data, rul_model, ims_features, detector):
        f.cache_clear()


def trained_rul_models(subset: str | None = None) -> list[str]:
    names = []
    for info in list_models("classical_ai"):
        n = info["name"]
        if n.startswith("rul_") and (subset is None or n.endswith(subset)):
            names.append(n.removeprefix("rul_").rsplit("_", 1)[0])
    return [m for m in ALL_MODELS if m in names]


def trained_detectors(test: str | None = None) -> list[str]:
    out = []
    for info in list_models("classical_ai"):
        n = info["name"]
        if n.startswith("anomaly_") and not n.startswith("anomaly_supervised") and (test is None or n.endswith(test)):
            out.append(n.removeprefix("anomaly_").rsplit("_", 2)[0])
    return sorted(set(out))


def best_rul_model(subset: str) -> str:
    """Lowest test RMSE among trained models (from the metrics stored with each model)."""
    best, best_rmse = None, np.inf
    for info in list_models("classical_ai"):
        n = info["name"]
        if n.startswith("rul_") and n.endswith(subset):
            rmse = info.get("metrics", {}).get("rmse", np.inf)
            if rmse < best_rmse:
                best, best_rmse = n.removeprefix("rul_").rsplit("_", 1)[0], rmse
    if best is None:
        raise ModelNotTrained(f"no RUL model trained for {subset}; train one on the Model Training page")
    return best


def best_detector(test: str) -> str:
    best, best_auc = None, -1.0
    for info in list_models("classical_ai"):
        n = info["name"]
        if n.startswith("anomaly_") and n.endswith(test) and not n.startswith("anomaly_supervised"):
            auc = info.get("metrics", {}).get("roc_auc") or 0
            if auc > best_auc:
                best, best_auc = n.removeprefix("anomaly_").rsplit("_", 2)[0], auc
    if best is None:
        raise ModelNotTrained(f"no anomaly detector trained for {test}")
    return best


# ---------------------------------------------------------------- twins and assessments


def turbofan_twin(asset_id: str, model: str | None = None) -> TurbofanTwin:
    a = parse_asset(asset_id)
    if a.kind != "turbofan":
        raise AssetNotFound(f"{asset_id} is not a turbofan")
    m = rul_model(model or best_rul_model(a.subset), a.subset)
    return TurbofanTwin(m, cmapss_data(a.subset), a.unit, a.split)


def bearing_twin(asset_id: str, model: str | None = None) -> BearingTwin:
    a = parse_asset(asset_id)
    if a.kind != "bearing":
        raise AssetNotFound(f"{asset_id} is not a bearing")
    det = detector(model or best_detector(a.test), a.test)
    return BearingTwin(det, ims_features(a.test).features, a.test, a.bearing)


def asset_prediction(asset_id: str, model: str | None = None, as_of: int | None = None) -> dict:
    """Numerical prediction + risk for one asset. All numbers come from trained models."""
    a = parse_asset(asset_id)
    if a.kind == "turbofan":
        tw = turbofan_twin(asset_id, model)
        st = tw.state()
        lo, hi = tw.model.interval([st["rul_pred"]])
        return {"asset_id": asset_id, "equipment_type": "turbofan", "model": tw.model.name,
                "rul_pred": st["rul_pred"], "rul_true": st["rul_true"], "sigma": tw.model.sigma, "q90": tw.model.q90,
                "rul_interval_90": [float(lo[0]), float(hi[0])], "rul_cap": tw.model.rul_cap,
                "current_cycle": st["current_cycle"], "risk": st["risk"], "time_unit": "cycle"}
    tw = bearing_twin(asset_id, model)
    st = tw.state(as_of)
    ttt = st["time_to_threshold_hours"]
    return {"asset_id": asset_id, "equipment_type": "bearing", "model": tw.detector.name,
            "anomaly_score": st["anomaly_score"], "threshold": tw.detector.threshold,
            "rul_pred": ttt, "sigma": (0.3 * ttt if ttt else None), "file_index": st["file_index"],
            "timestamp": st["timestamp"], "risk": st["risk"], "time_unit": "hour",
            "rul_definition": "hours until the extrapolated anomaly score crosses the alarm threshold (simulated)"}


def fleet_overview(subset: str = "FD001", model: str | None = None, limit: int | None = None) -> dict:
    """Predictions and risk for every official test engine of a subset (one batched model call)."""
    data = cmapss_data(subset)
    name = model or best_rul_model(subset)
    m = rul_model(name, subset)
    preds = m.predict_last(data.test)
    units = sorted(data.test["unit"].unique())
    last = data.test.groupby("unit").tail(1).set_index("unit")
    assets = []
    for u, p in zip(units, preds):
        r = assess_turbofan(f"{subset}-test-unit{u}", float(p), m.sigma, m.q90, m.rul_cap)
        assets.append({"asset_id": r.asset_id, "unit": int(u), "rul_pred": float(p),
                       "rul_true": float(last.loc[u, "rul_true"]), "current_cycle": int(last.loc[u, "cycle"]),
                       "pof": r.probability_of_failure, "risk_score": r.risk_score, "risk_level": r.risk_level,
                       "health": r.health_score, "urgency": r.urgency})
    assets.sort(key=lambda x: -x["risk_score"])
    counts = {lvl: sum(a["risk_level"] == lvl for a in assets) for lvl in ("low", "medium", "high", "critical")}
    return {"subset": subset, "model": name, "sigma": m.sigma, "assets": assets[:limit] if limit else assets,
            "risk_counts": counts, "n_assets": len(assets), "model_metrics": m.metrics.get("test", {})}


def bearing_overview(test: str = "2nd_test", model: str | None = None, as_of: int | None = None) -> dict:
    feats = ims_features(test).features
    name = model or best_detector(test)
    det = detector(name, test)
    out = []
    for b in sorted(feats["bearing"].unique()):
        tw = BearingTwin(det, feats, test, int(b))
        st = tw.state(as_of)
        r = st["risk"]
        out.append({"asset_id": tw.asset_id, "bearing": int(b), "anomaly_score": st["anomaly_score"],
                    "threshold": det.threshold, "pof": r["probability_of_failure"], "risk_score": r["risk_score"],
                    "risk_level": r["risk_level"], "health": r["health_score"], "urgency": r["urgency"],
                    "time_to_threshold_hours": st["time_to_threshold_hours"], "timestamp": st["timestamp"],
                    "failed_in_dataset": int(b) in ims.TESTS[test]["failed"]})
    return {"test": test, "model": name, "as_of": as_of, "assets": out,
            "max_file_index": int(feats["file_index"].max())}


def simulate(asset_id: str, params: SimulationParams, model: str | None = None, as_of: int | None = None) -> dict:
    a = parse_asset(asset_id)
    if a.kind == "turbofan":
        return turbofan_twin(asset_id, model).simulate(params)
    return bearing_twin(asset_id, model).simulate(params, as_of)


def default_assets(subset: str = "FD001", n: int = 5) -> list[str]:
    """Highest-risk test engines — a sensible default scope for agent runs and optimisation."""
    return [a["asset_id"] for a in fleet_overview(subset)["assets"][:n]]
