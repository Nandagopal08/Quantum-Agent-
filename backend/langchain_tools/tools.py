"""LangChain tools over the real models and computations.

Agents (LangGraph nodes) invoke these tools for every number they report, and the optional
tool-calling chat agent can call them too. Each call and result is published to the run's event
stream so the dashboard shows exactly which tool produced which value.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..counterfactual.scenarios import AssetState, compare_scenarios
from ..digital_twin.twin import SimulationParams
from ..events import bus, current_run
from ..explainability import explain as xai
from ..optimization.problem import Asset, default_problem
from ..optimization.solvers import compare_solvers
from ..services import fleet
from ..settings import get_settings


def _summarise(value: Any, depth: int = 0) -> Any:
    """Compact copy of a tool result for the event stream (long series are truncated)."""
    if isinstance(value, dict):
        return {k: _summarise(v, depth + 1) for k, v in list(value.items())[:25]
                if k not in ("series", "timing_curve", "expectation_history", "values", "X")}
    if isinstance(value, list):
        return [_summarise(v, depth + 1) for v in value[:8]] + (["…"] if len(value) > 8 else [])
    if isinstance(value, float):
        return round(value, 4)
    return value


def _traced(name: str, fn):
    def wrapper(**kwargs):
        run_id = current_run.get()
        t0 = time.time()
        if run_id:
            bus.publish(f"run:{run_id}", {"type": "tool_call", "tool": name, "args": kwargs})
        try:
            result = fn(**kwargs)
        except Exception as exc:
            if run_id:
                bus.publish(f"run:{run_id}", {"type": "tool_error", "tool": name, "error": str(exc)})
            raise
        if run_id:
            bus.publish(f"run:{run_id}", {"type": "tool_result", "tool": name, "seconds": round(time.time() - t0, 3),
                                          "result": _summarise(result)})
        return result
    wrapper.__name__ = name
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------- tool implementations


def check_data_quality(asset_id: str) -> dict:
    """Check the latest sensor data of an asset: missing values, out-of-distribution readings, history length."""
    a = fleet.parse_asset(asset_id)
    if a.kind == "turbofan":
        data = fleet.cmapss_data(a.subset)
        df = data.test if a.split == "test" else data.train
        g = df[df["unit"] == a.unit]
        if g.empty:
            return {"asset_id": asset_id, "ok": False, "issues": ["asset has no data"]}
        recent = g[data.sensors_used].tail(data.config.window)
        z = recent.abs().max()
        issues = []
        if recent.isna().any().any():
            issues.append("missing sensor values in recent window")
        drift = [s for s in data.sensors_used if z[s] > 6]
        if drift:
            issues.append(f"readings > 6σ from training distribution: {', '.join(drift)}")
        flat = [s for s in data.sensors_used if recent[s].std() == 0]
        if flat:
            issues.append(f"flat-lined sensors: {', '.join(flat)}")
        if len(g) < data.config.window:
            issues.append(f"short history ({len(g)} cycles < {data.config.window}); window is padded")
        return {"asset_id": asset_id, "ok": not any("missing" in i or "flat" in i for i in issues),
                "cycles_available": int(len(g)), "max_abs_z_recent": float(z.max()), "issues": issues,
                "source": "measured"}
    feats = fleet.ims_features(a.test).features
    g = feats[feats["bearing"] == a.bearing]
    zc = [c for c in g.columns if c.startswith("z_")]
    last = g.sort_values("file_index").tail(10)
    issues = []
    if last[zc].isna().any().any():
        issues.append("missing feature values in recent snapshots")
    return {"asset_id": asset_id, "ok": not issues, "snapshots_available": int(len(g)),
            "max_abs_z_recent": float(last[zc].abs().max().max()), "issues": issues, "source": "measured"}


def predict_rul(asset_id: str, model: str | None = None) -> dict:
    """Predict remaining useful life (turbofan, cycles) or time-to-alarm (bearing, hours) with the trained model."""
    p = fleet.asset_prediction(asset_id, model)
    keep = ("asset_id", "equipment_type", "model", "rul_pred", "rul_true", "sigma", "q90", "rul_interval_90",
            "rul_cap", "current_cycle", "anomaly_score", "threshold", "file_index", "timestamp", "time_unit",
            "rul_definition")
    return {k: p[k] for k in keep if k in p}


def assess_risk(asset_id: str, model: str | None = None) -> dict:
    """Probability of failure, consequence, risk score (PoF × CoF), risk level, health and urgency."""
    return fleet.asset_prediction(asset_id, model)["risk"]


def simulate_digital_twin(asset_id: str, horizon: int = 60, load: float = 1.0,
                          maintenance_at: int | None = None) -> dict:
    """Run the digital-twin simulation of future operation (simulated values)."""
    sim = fleet.simulate(asset_id, SimulationParams(horizon=horizon, load=load, maintenance_at=maintenance_at))
    last = sim["series"][-1]
    return {"asset_id": asset_id, "horizon": horizon, "load": load, "maintenance_at": maintenance_at,
            "final_cumulative_pof": last.get("cumulative_pof"), "final_rul_median": last.get("rul_median"),
            "hours_to_threshold": sim.get("hours_to_threshold"), "source": "simulated", "series": sim["series"]}


def compare_maintenance_scenarios(asset_id: str, horizon: float | None = None) -> dict:
    """Compare continue / immediate / delayed / reduced-load / optimal-timing maintenance scenarios."""
    p = fleet.asset_prediction(asset_id)
    if p["rul_pred"] is None:
        return {"asset_id": asset_id, "error": "no upward degradation trend; scenarios need a time-to-failure estimate"}
    state = AssetState(asset_id, p["equipment_type"], float(p["rul_pred"]), float(p["sigma"] or 1.0),
                       float(p.get("rul_cap", 125)))
    return compare_scenarios(state, horizon=horizon)


def optimize_maintenance_schedule(asset_ids: list[str], qaoa_layers: int | None = None) -> dict:
    """Schedule maintenance across assets with QAOA and classical baselines (MILP, greedy, annealing)."""
    s = get_settings()
    assets = []
    for aid in asset_ids:
        p = fleet.asset_prediction(aid)
        if p["rul_pred"] is None:
            continue
        assets.append(Asset(aid, p["equipment_type"], float(p["rul_pred"]), float(p["sigma"] or 1.0),
                            float(p.get("rul_cap", 125))))
    if not assets:
        return {"error": "no assets with a time-to-failure estimate"}
    problem = default_problem(assets, s)
    _, _, nv = problem.to_qubo()
    max_q = s["optimization"]["max_qubits"]
    while nv > max_q and len(problem.assets) > 1:   # keep the highest-risk assets that fit on the simulator
        problem = default_problem(sorted(problem.assets, key=lambda a: a.rul_pred)[:-1], s)
        _, _, nv = problem.to_qubo()
    return compare_solvers(problem, qaoa_layers or s["optimization"]["qaoa_layers"], s["optimization"]["qaoa_steps"],
                           max_q)


def explain_prediction(asset_id: str) -> dict:
    """Feature attributions, counterfactual sensor explanation and uncertainty for a turbofan prediction."""
    a = fleet.parse_asset(asset_id)
    if a.kind == "turbofan":
        tw = fleet.turbofan_twin(asset_id)
        return {"asset_id": asset_id, "local": xai.local_attribution(tw.model, tw.data, a.unit),
                "counterfactual": xai.counterfactual_explanation(tw.model, tw.data, a.unit),
                "uncertainty": xai.uncertainty(tw.model, tw.data, a.unit)}
    d = fleet.ims_features(a.test)
    return {"asset_id": asset_id, "deviation": xai.bearing_deviation(d.features, d.feature_cols, a.bearing)}


# ---------------------------------------------------------------- LangChain tool objects


class AssetArgs(BaseModel):
    asset_id: str = Field(description="Asset id, e.g. FD001-test-unit34 or IMS-2nd_test-bearing1")


ModelName = Literal["random_forest", "xgboost", "lightgbm", "catboost", "lstm", "gru", "cnn_lstm", "transformer",
                    "isolation_forest", "one_class_svm", "autoencoder", "lstm_autoencoder"]


class PredictArgs(AssetArgs):
    model: ModelName | None = Field(None, description="Omit to use the best trained model (recommended).")


class TwinArgs(AssetArgs):
    horizon: int = Field(60, ge=1, le=500)
    load: float = Field(1.0, gt=0, le=2.0)
    maintenance_at: int | None = Field(None, ge=0)


class ScenarioArgs(AssetArgs):
    horizon: float | None = Field(None, gt=0, le=300,
                                  description="Planning horizon in the asset's time unit. Omit to use the configured default.")


class ScheduleArgs(BaseModel):
    asset_ids: list[str] = Field(min_length=1, max_length=8)
    qaoa_layers: int | None = Field(None, ge=1, le=6)


_SPECS = [
    (check_data_quality, AssetArgs), (predict_rul, PredictArgs), (assess_risk, PredictArgs),
    (simulate_digital_twin, TwinArgs), (compare_maintenance_scenarios, ScenarioArgs),
    (optimize_maintenance_schedule, ScheduleArgs), (explain_prediction, AssetArgs),
]

TOOLS: dict[str, StructuredTool] = {
    fn.__name__: StructuredTool.from_function(func=_traced(fn.__name__, fn), name=fn.__name__,
                                              description=fn.__doc__, args_schema=schema)
    for fn, schema in _SPECS
}

# Read-only subset exposed to the free-form chat agent (no optimisation runs from chat).
CHAT_TOOLS = [TOOLS[n] for n in ("predict_rul", "assess_risk", "compare_maintenance_scenarios",
                                 "explain_prediction", "check_data_quality")]


def call(name: str, **kwargs) -> Any:
    return TOOLS[name].invoke(kwargs)


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x
