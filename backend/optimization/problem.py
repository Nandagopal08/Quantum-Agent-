"""Maintenance scheduling problem and its QUBO formulation.

Decision variables: x[i, t] = 1 if asset i receives preventive maintenance in slot t (slot start τ_t).
At most one slot per asset; at most K assets per slot (crew / hangar capacity).

Expected cost of asset i (deterministic, from the RUL model's Gaussian uncertainty):
    not maintained:     C0_i     = P_i(T ≤ H) · CoF_i
    maintained at τ_t:  C_it     = PM_i + PMdown_i · rate_i + P_i(T ≤ τ_t) · CoF_i
                                 + P(fail after restoration before H) · CoF_i
Objective:  Σ_i C0_i + Σ_{i,t} x_it (C_it − C0_i)

QUBO penalties (A = penalty weight):
    one slot per asset:  A · Σ_i Σ_{t<t'} x_it x_it'
    capacity per slot:   A · (Σ_i x_it + Σ_b w_b s_tb − K)²   with binary slack bits s_tb
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..risk.engine import consequence_of_failure
from ..settings import equipment_costs, get_settings


@dataclass
class Asset:
    asset_id: str
    equipment_type: str
    rul_pred: float
    sigma: float
    rul_cap: float = 125.0
    criticality: float | None = None   # overrides the equipment-type default


def p_fail_by(t: float, mean: float, sd: float) -> float:
    """P(T ≤ t) for T ~ N(mean, sd) truncated to (0, ∞)."""
    sd = max(sd, 1e-6)
    z0 = norm.cdf(-mean / sd)
    return float(max(norm.cdf((t - mean) / sd) - z0, 0) / max(1 - z0, 1e-12))


def _slack_weights(k: int) -> list[int]:
    if k <= 0:
        return []
    bits = int(np.floor(np.log2(k))) + 1
    w = [2 ** b for b in range(bits - 1)]
    w.append(k - sum(w))
    return w


@dataclass
class SchedulingProblem:
    assets: list[Asset]
    slot_times: list[float]
    capacity: int
    horizon: float
    cost_none: np.ndarray = field(init=False)
    cost_maint: np.ndarray = field(init=False)
    pof_none: np.ndarray = field(init=False)
    pof_maint: np.ndarray = field(init=False)
    downtime_maint: np.ndarray = field(init=False)

    def __post_init__(self):
        s = get_settings()
        n, T = len(self.assets), len(self.slot_times)
        self.cost_none, self.pof_none = np.zeros(n), np.zeros(n)
        self.cost_maint, self.pof_maint, self.downtime_maint = np.zeros((n, T)), np.zeros((n, T)), np.zeros(n)
        for i, a in enumerate(self.assets):
            c = equipment_costs(a.equipment_type, s)
            crit_scale = (a.criticality / c["criticality"]) if a.criticality else 1.0
            cof = consequence_of_failure(a.equipment_type, s) * crit_scale
            self.pof_none[i] = p_fail_by(self.horizon, a.rul_pred, a.sigma)
            self.cost_none[i] = self.pof_none[i] * cof
            restored = c["restoration_fraction"] * a.rul_cap
            self.downtime_maint[i] = c["preventive_downtime"]
            for t, tau in enumerate(self.slot_times):
                p_before = p_fail_by(tau, a.rul_pred, a.sigma)
                p_after = (1 - p_before) * p_fail_by(self.horizon - tau - c["preventive_downtime"], restored, 0.1 * restored)
                self.pof_maint[i, t] = p_before + p_after
                self.cost_maint[i, t] = ((1 - p_before) * (c["preventive_maintenance_cost"]
                                         + c["preventive_downtime"] * c["downtime_cost_per_unit"])
                                         + self.pof_maint[i, t] * cof)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_slots(self) -> int:
        return len(self.slot_times)

    @property
    def n_decision(self) -> int:
        return self.n_assets * self.n_slots

    def var(self, i: int, t: int) -> int:
        return i * self.n_slots + t

    # ---- evaluation of a plan (decision bits only)
    def evaluate(self, x: np.ndarray) -> dict:
        X = np.asarray(x[: self.n_decision]).reshape(self.n_assets, self.n_slots).astype(int)
        per_asset = X.sum(1)
        per_slot = X.sum(0)
        feasible = bool((per_asset <= 1).all() and (per_slot <= self.capacity).all())
        cost = pof = down = 0.0
        plan = []
        for i, a in enumerate(self.assets):
            ts = np.flatnonzero(X[i])
            if len(ts):
                t = int(ts[0])
                cost += self.cost_maint[i, t]
                pof += self.pof_maint[i, t]
                down += self.downtime_maint[i]
                plan.append({"asset_id": a.asset_id, "slot": t, "slot_time": self.slot_times[t], "maintain": True})
            else:
                cost += self.cost_none[i]
                pof += self.pof_none[i]
                plan.append({"asset_id": a.asset_id, "slot": None, "slot_time": None, "maintain": False})
        return {"feasible": feasible, "total_expected_cost": float(cost), "expected_failures": float(pof),
                "mean_failure_risk": float(pof / self.n_assets), "planned_downtime": float(down),
                "maintenance_count": int(X.sum()), "plan": plan}

    # ---- QUBO
    def to_qubo(self, penalty: float | None = None) -> tuple[np.ndarray, float, int]:
        """Return (Q upper-triangular, offset, n_vars). Variables: decision bits then slack bits."""
        n, T = self.n_assets, self.n_slots
        slack_w = _slack_weights(self.capacity)
        nv = n * T + T * len(slack_w)
        lin = (self.cost_maint - self.cost_none[:, None]).ravel()
        # Violating a constraint by one unit can gain at most max|lin|, so a slightly larger A suffices;
        # a much larger A would drown the objective in penalty terms (bad for QAOA and annealing).
        A = penalty if penalty is not None else float(1.5 * np.abs(lin).max() + 1.0)
        Q = np.zeros((nv, nv))
        offset = float(self.cost_none.sum())
        Q[np.arange(n * T), np.arange(n * T)] += lin
        for i in range(n):
            for t in range(T):
                for t2 in range(t + 1, T):
                    Q[self.var(i, t), self.var(i, t2)] += A
        for t in range(T):
            terms = [(self.var(i, t), 1.0) for i in range(n)]
            terms += [(n * T + t * len(slack_w) + b, float(w)) for b, w in enumerate(slack_w)]
            # A (Σ c_j y_j − K)² = A [Σ c_j² y_j + 2 Σ_{j<k} c_j c_k y_j y_k − 2K Σ c_j y_j + K²]
            K = self.capacity
            for j, (vj, cj) in enumerate(terms):
                Q[vj, vj] += A * (cj * cj - 2 * K * cj)
                for vk, ck in terms[j + 1:]:
                    a, b = min(vj, vk), max(vj, vk)
                    Q[a, b] += 2 * A * cj * ck
            offset += A * K * K
        return Q, offset, nv

    def describe(self) -> dict:
        return {"assets": [a.__dict__ for a in self.assets], "slot_times": self.slot_times,
                "capacity": self.capacity, "horizon": self.horizon,
                "cost_none": self.cost_none.round(2).tolist(), "cost_maint": self.cost_maint.round(2).tolist(),
                "pof_none": self.pof_none.round(4).tolist(), "pof_maint": self.pof_maint.round(4).tolist(),
                "formulation": __doc__.strip()}


def qubo_energy(Q: np.ndarray, offset: float, x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x @ Q @ x + offset)


def default_problem(assets: list[Asset], settings: dict | None = None) -> SchedulingProblem:
    s = settings or get_settings()
    o = s["optimization"]
    H = s["simulation"]["planning_horizon"]
    slots = [round(H * t / (o["slots"] + 1), 1) for t in range(o["slots"])]  # e.g. 0 and H/3 for 2 slots
    return SchedulingProblem(assets, slots, o["crew_capacity_per_slot"], H)
