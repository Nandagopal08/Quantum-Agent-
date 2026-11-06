"""LangChain tools used by the missing-data agents. Every number an agent reports comes from one of these.

Windows travel through the agent state as plain lists (``T × C``, ``None`` = missing) so runs can be
persisted as JSON and replayed in the dashboard.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...langchain_tools.tools import _traced
from ...missing_data import BASELINE_IMPUTERS, PATTERNS, impute_baseline, make_mask, missingness_profile
from ...quantum_models import seq_experiments as sx
from ...quantum_models.qseq import SEQ_MODELS

DEFAULT_RANKING = ["linear", "locf", "mean"]      # used only when no benchmark report exists


# ---------------------------------------------------------------- cached data / models


@lru_cache(maxsize=4)
def _sequences(subset: str, window: int):
    cfg = sx.SeqExperimentConfig(subset=subset, window=window)
    from ...datasets import cmapss

    data = cmapss.prepare(cmapss.CmapssConfig(subset=subset, window=window, seed=cfg.seed), save=False)
    means = data.train[data.sensors_used].to_numpy(float).mean(0)
    return data, means


@lru_cache(maxsize=16)
def _model(task: str, name: str, subset: str):
    return sx.load_seq_model(task, name, subset)


def clear_caches() -> None:
    _sequences.cache_clear()
    _model.cache_clear()


def _report(subset: str) -> dict | None:
    return sx.load_report(subset)


def _window_len(subset: str) -> int:
    r = _report(subset)
    return int(r["data"]["window"]) if r else sx.SeqExperimentConfig().window


def available_imputers(subset: str) -> list[str]:
    out = list(BASELINE_IMPUTERS)
    for name in SEQ_MODELS:
        try:
            _model("impute", name, subset)
            out.append(name)
        except FileNotFoundError:
            pass
    return out


def _to_arrays(values: list[list[Any]]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.array([[np.nan if v is None else float(v) for v in row] for row in values], float)
    mask = (~np.isnan(raw)).astype(float)
    return np.nan_to_num(raw), mask


def _to_lists(x: np.ndarray, mask: np.ndarray | None = None) -> list[list[float | None]]:
    return [[None if (mask is not None and mask[t, c] == 0) else round(float(x[t, c]), 6)
             for c in range(x.shape[1])] for t in range(x.shape[0])]


# ---------------------------------------------------------------- tool implementations


def window_and_truth(asset_id: str, missing_rate: float = 0.3, pattern: str = "mcar", seed: int = 0
                     ) -> tuple[dict, dict]:
    """(observation, hidden truth) for an engine window with injected missingness. Not a tool: the truth must never
    reach an agent or the event stream; the runner keeps it in a per-run vault for the Reporter's scoring."""
    parts = asset_id.split("-")
    if len(parts) != 3 or not parts[2].startswith("unit"):
        raise ValueError("asset_id must look like FD001-test-unit34")
    subset, split, unit = parts[0], parts[1], int(parts[2][4:])
    W = _window_len(subset)
    data, _ = _sequences(subset, W)
    df = data.test if split == "test" else data.train if split == "train" else None
    if df is None:
        raise ValueError("split must be test or train")
    g = df[df["unit"] == unit]
    if g.empty:
        raise ValueError(f"{asset_id}: no such engine")
    arr = g[data.sensors_used].to_numpy(float)
    if len(arr) < W:
        arr = np.vstack([np.repeat(arr[:1], W - len(arr), 0), arr])
    x = arr[-W:]
    mask = make_mask((1, *x.shape), missing_rate, pattern, np.random.default_rng(seed))[0]
    obs = {"asset_id": asset_id, "subset": subset, "unit": unit, "sensors": data.sensors_used,
           "cycles": int(len(g)), "window": W, "values": _to_lists(x, mask), "mask": mask.tolist(),
           "injected": {"missing_rate": missing_rate, "pattern": pattern, "seed": seed},
           "source": "measured (with simulated missingness)"}
    truth = {"values": _to_lists(x), "rul_true": float(g["rul"].iloc[-1]) if "rul" in g else None,
             "injected": obs["injected"]}
    return obs, truth


def observe_asset(asset_id: str, missing_rate: float = 0.3, pattern: str = "mcar", seed: int = 0) -> dict:
    """Latest sensor window of a C-MAPSS engine (e.g. FD001-test-unit34) with missingness injected at the given
    rate and pattern (mcar | block). Returns the observed window only (None = missing)."""
    return window_and_truth(asset_id, missing_rate, pattern, seed)[0]


def expected_input(subset: str = "FD001") -> dict:
    """Window length and sensor list the trained sequence models expect."""
    W = _window_len(subset)
    data, _ = _sequences(subset, W)
    return {"window": W, "sensors": list(data.sensors_used)}


def profile_missingness(mask: list[list[float]], sensors: list[str]) -> dict:
    """Overall and per-sensor missing rate, gap lengths and the likely missingness pattern (mcar or block)."""
    return missingness_profile(np.asarray(mask, float), sensors)


def rank_imputers(missing_rate: float, pattern: str, subset: str = "FD001", exclude: list[str] | None = None) -> dict:
    """Rank the available imputation methods by their benchmark RMSE at the nearest evaluated missing rate and
    the given pattern. Evidence comes from the saved quantum sequence-model benchmark report."""
    exclude = set(exclude or [])
    avail = available_imputers(subset)
    report = _report(subset)
    pat = pattern if pattern in PATTERNS else "mcar"
    if not report:
        ranked = [{"method": m, "rmse": None} for m in DEFAULT_RANKING if m in avail and m not in exclude]
        return {"ranking": ranked, "evidence": "no benchmark report; default order (linear, locf, mean)",
                "benchmark_rate": None, "pattern": pat}
    table = report["imputation"]["ranking"].get(pat, {})
    rates = sorted(float(r) for r in table)
    nearest = min(rates, key=lambda r: abs(r - missing_rate))
    ranked = [r for r in table[str(nearest)] if r["method"] in avail and r["method"] not in exclude]
    note = ""
    if missing_rate > max(rates) + 0.05 or missing_rate < min(rates) - 0.05:
        note = f" (outside the benchmarked range {min(rates):.0%}–{max(rates):.0%}; nearest rate used)"
    return {"ranking": ranked, "benchmark_rate": nearest, "pattern": pat,
            "seeds": report.get("seeds"), "evidence": f"{report['name']}: {pat} at {nearest:.0%} missing{note}"}


def run_imputation(method: str, values: list[list[Any]], subset: str = "FD001", holdout_fraction: float = 0.15,
                   seed: int = 0) -> dict:
    """Impute a window with one method. Also runs a label-free self-check: a fraction of the *observed* values is
    hidden, re-imputed with the same method and with linear interpolation, and the errors are reported."""
    x, mask = _to_arrays(values)
    _, means = _sequences(subset, x.shape[0])
    models = {}
    if method not in BASELINE_IMPUTERS:
        models[method] = _model("impute", method, subset)["model"]

    def imp(m):
        return sx.impute_any(method, x[None], m[None], means, models)[0]

    full = imp(mask)
    rng = np.random.default_rng(seed + 7)
    obs = np.argwhere(mask > 0)
    k = max(1, int(round(holdout_fraction * len(obs)))) if len(obs) > 1 else 0
    check: dict[str, Any] = {"holdout_values": k}
    if k:
        pick = obs[rng.choice(len(obs), k, replace=False)]
        m2 = mask.copy()
        m2[pick[:, 0], pick[:, 1]] = 0
        e_m = imp(m2)[pick[:, 0], pick[:, 1]] - x[pick[:, 0], pick[:, 1]]
        e_l = impute_baseline("linear", x[None], m2[None], means)[0][pick[:, 0], pick[:, 1]] - x[pick[:, 0], pick[:, 1]]
        check.update(holdout_rmse=float(np.sqrt(np.mean(e_m ** 2))),
                     linear_holdout_rmse=float(np.sqrt(np.mean(e_l ** 2))),
                     effective_missing_rate=float(1 - m2.mean()))
    return {"method": method, "values": _to_lists(full), "self_check": check,
            "max_abs_z": float(np.abs(full).max()), "imputed_count": int((mask == 0).sum())}


def forecast_rul(values: list[list[float]], subset: str = "FD001", model: str | None = None) -> dict:
    """Predict remaining useful life (cycles) from a complete (imputed) window with a saved quantum sequence
    model; omit ``model`` to use the best quantum model on validation engines. Also returns its classical twin."""
    report = _report(subset)
    if model is None:
        if report is None:
            raise FileNotFoundError("no quantum sequence benchmark report; run scripts/run_quantum_seq.py")
        q = [r for r in report["rul"]["summary"] if r["model"].startswith("q_")]
        model = min(q, key=lambda r: r["val_rmse"])["model"]
    x = np.asarray(values, float)[None]
    out = {"model": model}
    for name in (model, "c_" + model.split("_", 1)[1] if model.startswith("q_") else None):
        if not name:
            continue
        try:
            b = _model("rul", name, subset)
        except FileNotFoundError:
            continue
        pred = float(sx.predict_rul_windows(b["model"], x, b["cap"])[0])
        rec = {"rul_pred": pred, "sigma": b["sigma"], "interval_90": [max(0.0, pred - 1.645 * b["sigma"]),
                                                                       pred + 1.645 * b["sigma"]],
               "cap": b["cap"]}
        out["prediction" if name == model else "classical_twin"] = rec | {"model": name}
    if "prediction" not in out:
        raise FileNotFoundError(f"RUL model {model} is not trained")
    return out


# ---------------------------------------------------------------- LangChain tool objects


class ObserveArgs(BaseModel):
    asset_id: str = Field(description="C-MAPSS engine id, e.g. FD001-test-unit34")
    missing_rate: float = Field(0.3, ge=0, lt=1)
    pattern: str = Field("mcar", pattern="^(mcar|block)$")
    seed: int = 0


class ProfileArgs(BaseModel):
    mask: list[list[float]]
    sensors: list[str]


class RankArgs(BaseModel):
    missing_rate: float = Field(ge=0, le=1)
    pattern: str
    subset: str = "FD001"
    exclude: list[str] | None = None


class ImputeArgs(BaseModel):
    method: str
    values: list[list[float | None]]
    subset: str = "FD001"
    holdout_fraction: float = Field(0.15, gt=0, lt=0.5)
    seed: int = 0


class ForecastArgs(BaseModel):
    values: list[list[float]]
    subset: str = "FD001"
    model: str | None = None


_SPECS = [(observe_asset, ObserveArgs), (profile_missingness, ProfileArgs), (rank_imputers, RankArgs),
          (run_imputation, ImputeArgs), (forecast_rul, ForecastArgs)]
TOOLS: dict[str, StructuredTool] = {
    fn.__name__: StructuredTool.from_function(func=_traced(fn.__name__, fn), name=fn.__name__,
                                              description=fn.__doc__, args_schema=schema)
    for fn, schema in _SPECS
}


def call(name: str, **kwargs) -> Any:
    return TOOLS[name].invoke(kwargs)
