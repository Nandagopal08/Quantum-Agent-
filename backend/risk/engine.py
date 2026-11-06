"""Equipment health and risk assessment.

    Risk score = Probability of Failure (within horizon) × Consequence of Failure

* Turbofan: PoF = Φ((h − RUL̂) / σ), with σ the model's validation-calibrated residual spread.
* Bearing:  no RUL labels exist, so PoF comes from the anomaly score relative to the detector's
  alarm threshold: severity s = (score − baseline median) / (threshold − baseline median) and
  PoF = 1 / (1 + exp(−k (s − s₀))). This mapping is a documented heuristic, reported as such.

Consequence of failure = (corrective cost + failure downtime × downtime cost) × criticality.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.stats import norm

from ..settings import equipment_costs, get_settings

LEVELS = ("low", "medium", "high", "critical")


@dataclass
class RiskAssessment:
    asset_id: str
    equipment_type: str
    rul_pred: float | None
    rul_interval: tuple[float, float] | None
    probability_of_failure: float
    consequence_of_failure: float
    risk_score: float
    risk_level: str
    health_score: float
    urgency: str
    horizon: float
    method: str
    drivers: list[str]
    requires_approval: bool

    def to_dict(self) -> dict:
        return asdict(self)


def consequence_of_failure(kind: str, settings: dict | None = None) -> float:
    c = equipment_costs(kind, settings)
    return (c["corrective_maintenance_cost"] + c["failure_downtime"] * c["downtime_cost_per_unit"]) * c["criticality"]


def risk_level(risk_score: float, cof: float, settings: dict | None = None) -> str:
    th = (settings or get_settings())["risk"]["thresholds"]
    frac = risk_score / cof if cof else 0.0
    if frac >= th["critical"]:
        return "critical"
    if frac >= th["high"]:
        return "high"
    if frac >= th["medium"]:
        return "medium"
    return "low"


def urgency(level: str, rul: float | None, settings: dict | None = None) -> str:
    u = (settings or get_settings())["risk"]["urgency_rul"]
    if level == "critical" or (rul is not None and rul <= u["immediate"]):
        return "immediate"
    if level == "high" or (rul is not None and rul <= u["soon"]):
        return "soon"
    if level == "medium" or (rul is not None and rul <= u["planned"]):
        return "planned"
    return "monitor"


def needs_approval(level: str, settings: dict | None = None) -> bool:
    gate = (settings or get_settings())["risk"]["human_approval_level"]
    return LEVELS.index(level) >= LEVELS.index(gate)


def assess_turbofan(asset_id: str, rul_pred: float, sigma: float, q90: float, rul_cap: float = 125,
                    horizon: float | None = None, settings: dict | None = None) -> RiskAssessment:
    s = settings or get_settings()
    h = horizon if horizon is not None else s["risk"]["failure_horizon"]
    pof = float(norm.cdf((h - rul_pred) / max(sigma, 1e-6)))
    cof = consequence_of_failure("turbofan", s)
    risk = pof * cof
    level = risk_level(risk, cof, s)
    drivers = [f"predicted RUL {rul_pred:.1f} cycles (90% interval {max(rul_pred - q90, 0):.0f}–{rul_pred + q90:.0f})",
               f"P(failure within {h:g} cycles) = {pof:.1%} from N(RUL̂, σ={sigma:.1f})",
               f"consequence of failure {cof:,.0f} {s['currency']}"]
    return RiskAssessment(
        asset_id=asset_id, equipment_type="turbofan", rul_pred=float(rul_pred),
        rul_interval=(float(max(rul_pred - q90, 0)), float(rul_pred + q90)),
        probability_of_failure=pof, consequence_of_failure=cof, risk_score=risk, risk_level=level,
        health_score=float(100 * min(max(rul_pred / rul_cap, 0), 1)), urgency=urgency(level, rul_pred, s),
        horizon=h, method="gaussian_rul_residuals", drivers=drivers, requires_approval=needs_approval(level, s))


def bearing_severity(score: float, baseline_median: float, threshold: float) -> float:
    return (score - baseline_median) / max(threshold - baseline_median, 1e-12)


def assess_bearing(asset_id: str, score: float, baseline_median: float, threshold: float,
                   rul_hours: float | None = None, k: float = 3.0, s0: float = 1.5,
                   settings: dict | None = None) -> RiskAssessment:
    s = settings or get_settings()
    sev = bearing_severity(score, baseline_median, threshold)
    pof = 1.0 / (1.0 + math.exp(-k * (sev - s0)))
    cof = consequence_of_failure("bearing", s)
    risk = pof * cof
    level = risk_level(risk, cof, s)
    drivers = [f"anomaly severity {sev:.2f} (1.0 = alarm threshold)",
               f"PoF {pof:.1%} from logistic mapping (k={k}, s₀={s0}; heuristic)",
               f"consequence of failure {cof:,.0f} {s['currency']}"]
    if rul_hours is not None:
        drivers.insert(1, f"extrapolated time to alarm-threshold crossing ≈ {rul_hours:.1f} h (simulated)")
    return RiskAssessment(
        asset_id=asset_id, equipment_type="bearing", rul_pred=rul_hours, rul_interval=None,
        probability_of_failure=pof, consequence_of_failure=cof, risk_score=risk, risk_level=level,
        health_score=float(100 * max(0.0, min(1.0, 1 - max(sev, 0) / 3))), urgency=urgency(level, None, s),
        horizon=s["risk"]["failure_horizon"], method="anomaly_severity_logistic", drivers=drivers,
        requires_approval=needs_approval(level, s))
