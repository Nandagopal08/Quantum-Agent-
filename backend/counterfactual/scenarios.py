"""Counterfactual maintenance scenarios over a planning horizon H (all outputs are simulated).

Time to failure from now:  T ~ N(RUL̂, σ) truncated to (0, ∞) — the asset is running now, so it has
not failed yet (model prediction + calibrated uncertainty)
Operating load L changes the wear rate to L^α, so effective time to failure is T / L^α.
Preventive maintenance (PM) at time m, if the asset has not failed first, restores remaining life to
T' ~ N(restoration_fraction × RUL cap, 0.1 × that).

Cost components (no double counting):
  maintenance_cost       = P(PM performed) × PM cost
  downtime_cost          = P(PM performed) × PM downtime × downtime cost per unit
  expected_failure_cost  = P(failure within H) × CoF,  CoF = (corrective cost + failure downtime × rate) × criticality
  production_loss        = (1 − L) × H × lost output per unit         (only when load is reduced)
  total_expected_cost    = sum of the above
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import truncnorm

from ..risk.engine import consequence_of_failure
from ..settings import equipment_costs, get_settings

SCENARIOS = ("continue_operation", "immediate_maintenance", "delay_maintenance", "reduce_load",
             "optimal_maintenance_timing")


@dataclass
class AssetState:
    asset_id: str
    equipment_type: str          # turbofan | bearing
    rul_pred: float
    sigma: float
    rul_cap: float = 125.0


def positive_normal(mean: float, sd: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """N(mean, sd) truncated to (0, ∞)."""
    sd = max(sd, 1e-6)
    return truncnorm.rvs((0 - mean) / sd, np.inf, loc=mean, scale=sd, size=n, random_state=rng)


def _simulate(state: AssetState, horizon: float, maintenance_at: float | None, load: float,
              n: int, seed: int, settings: dict) -> dict:
    c = equipment_costs(state.equipment_type, settings)
    alpha = settings["simulation"]["load_exponent"]
    rng = np.random.default_rng(seed)
    rate = load ** alpha
    t_fail = positive_normal(state.rul_pred, state.sigma, n, rng) / rate
    restored_mean = c["restoration_fraction"] * state.rul_cap
    t_restored = positive_normal(restored_mean, 0.1 * restored_mean, n, rng) / rate

    if maintenance_at is None:
        failed = t_fail <= horizon
        pm_done = np.zeros(n, bool)
        remaining = np.maximum(t_fail - horizon, 0)
    else:
        m = maintenance_at
        fail_before = t_fail <= m
        pm_done = ~fail_before
        after = m + c["preventive_downtime"]
        fail_after = pm_done & (after + t_restored <= horizon)
        failed = fail_before | fail_after
        remaining = np.where(pm_done, np.maximum(after + t_restored - horizon, 0), 0)

    p_fail, p_pm = float(failed.mean()), float(pm_done.mean())
    cof = consequence_of_failure(state.equipment_type, settings)
    maintenance_cost = p_pm * c["preventive_maintenance_cost"]
    downtime_cost = p_pm * c["preventive_downtime"] * c["downtime_cost_per_unit"]
    failure_cost = p_fail * cof
    production_loss = max(0.0, 1 - load) * horizon * c["lost_output_per_unit"]
    return {
        "failure_probability": p_fail, "pm_probability": p_pm,
        "expected_rul_end_of_horizon": float(remaining.mean()),
        "expected_life_from_now": float(np.where(pm_done, (maintenance_at or 0) + t_restored, t_fail).mean()),
        "maintenance_cost": maintenance_cost, "downtime_cost": downtime_cost,
        "expected_failure_cost": failure_cost, "production_loss": production_loss,
        "total_expected_cost": maintenance_cost + downtime_cost + failure_cost + production_loss,
    }


def compare_scenarios(state: AssetState, horizon: float | None = None, delay: float | None = None,
                      reduced_load: float = 0.7, samples: int | None = None, seed: int | None = None,
                      settings: dict | None = None) -> dict:
    s = settings or get_settings()
    H = float(horizon or s["simulation"]["planning_horizon"])
    n = samples or s["simulation"]["monte_carlo_samples"]
    sd = s["simulation"]["seed"] if seed is None else seed
    d = float(delay if delay is not None else round(min(max(state.rul_pred * 0.5, 1), H)))
    unit = equipment_costs(state.equipment_type, s)["time_unit"]

    def run(name, label, m, load, desc):
        return {"scenario": name, "label": label, "maintenance_at": m, "load": load, "description": desc,
                "source": "simulated", **_simulate(state, H, m, load, n, sd, s)}

    results = [
        run("continue_operation", "Continue normal operation", None, 1.0,
            f"No maintenance for {H:g} {unit}s at nominal load."),
        run("immediate_maintenance", "Immediate preventive maintenance", 0.0, 1.0,
            "Preventive maintenance now, then nominal operation."),
        run("delay_maintenance", f"Delay maintenance by {d:g} {unit}s", d, 1.0,
            f"Operate at nominal load, preventive maintenance after {d:g} {unit}s."),
        run("reduce_load", f"Reduce operating load to {reduced_load:.0%}", None, reduced_load,
            f"No maintenance; load reduced to {reduced_load:.0%} (wear rate {reduced_load ** s['simulation']['load_exponent']:.2f}×)."),
    ]
    # optimal maintenance timing: scan candidate times, keep the cheapest expected total cost
    grid = np.unique(np.linspace(0, H, 25).round(1))
    curve = []
    for m in grid:
        r = _simulate(state, H, float(m), 1.0, n, sd, s)
        curve.append({"maintenance_at": float(m), "total_expected_cost": r["total_expected_cost"],
                      "failure_probability": r["failure_probability"]})
    best_m = min(curve, key=lambda p: p["total_expected_cost"])["maintenance_at"]
    results.append(run("optimal_maintenance_timing", f"Maintenance at optimal time ({best_m:g} {unit}s)", best_m, 1.0,
                       f"Preventive maintenance at the time that minimises expected total cost over {H:g} {unit}s."))

    ranked = sorted(results, key=lambda r: r["total_expected_cost"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return {
        "asset_id": state.asset_id, "equipment_type": state.equipment_type, "horizon": H, "time_unit": unit,
        "inputs": {"rul_pred": state.rul_pred, "sigma": state.sigma, "samples": n, "seed": sd},
        "consequence_of_failure": consequence_of_failure(state.equipment_type, s),
        "scenarios": results, "best": ranked[0]["scenario"], "timing_curve": curve,
        "explanation": explain(results, ranked, s["currency"], unit),
        "assumptions": __doc__.strip(),
    }


def explain(results: list[dict], ranked: list[dict], currency: str, unit: str) -> list[str]:
    """Plain-language comparison built only from the computed numbers."""
    by = {r["scenario"]: r for r in results}
    best, base = ranked[0], by["continue_operation"]
    lines = [f"Lowest expected total cost: {best['label']} at {best['total_expected_cost']:,.0f} {currency} "
             f"(failure probability {best['failure_probability']:.1%})."]
    if best["scenario"] != "continue_operation":
        saving = base["total_expected_cost"] - best["total_expected_cost"]
        lines.append(f"Compared with continuing as-is ({base['total_expected_cost']:,.0f} {currency}, failure "
                     f"probability {base['failure_probability']:.1%}), it saves {saving:,.0f} {currency}.")
        dp = base["failure_probability"] - best["failure_probability"]
        spend = best["maintenance_cost"] + best["downtime_cost"] + best["production_loss"]
        lines.append(f"It spends {spend:,.0f} {currency} on maintenance, downtime and lost output to cut failure "
                     f"probability by {dp:.1%}, which lowers expected failure cost by "
                     f"{base['expected_failure_cost'] - best['expected_failure_cost']:,.0f} {currency}.")
    else:
        lines.append("Failure risk over the horizon is low enough that maintenance spend is not yet justified.")
    imm, dly = by["immediate_maintenance"], by["delay_maintenance"]
    cheaper = "delaying" if dly["total_expected_cost"] < imm["total_expected_cost"] else "maintaining immediately"
    lines.append(f"Between immediate and delayed maintenance, {cheaper} is cheaper "
                 f"({imm['total_expected_cost']:,.0f} vs {dly['total_expected_cost']:,.0f} {currency}); delaying "
                 f"raises failure probability from {imm['failure_probability']:.1%} to {dly['failure_probability']:.1%}.")
    rl = by["reduce_load"]
    lines.append(f"Reducing load lowers failure probability to {rl['failure_probability']:.1%} but costs "
                 f"{rl['production_loss']:,.0f} {currency} in lost output.")
    lines.append("All values are simulated from the model's RUL prediction and the configured cost assumptions.")
    return lines
