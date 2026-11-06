"""Software digital twins for a C-MAPSS turbofan engine and an IMS bearing.

Every series point carries ``source``: ``"measured"`` (dataset values, model predictions on measured
data) or ``"simulated"`` (projected futures). Simulated values are never mixed into measured series.

Turbofan future simulation (Monte Carlo, ``n`` samples):
    RUL₀ ~ N(RUL̂, σ) truncated to (0, ∞)   (model prediction and calibrated uncertainty)
    wear rate per cycle = load^α  (α = ``load_exponent``; load 1.0 = nominal)
    RUL_{t+1} = RUL_t − rate·(1 + ε),  ε ~ N(0, 0.05)
    maintenance at cycle m restores RUL to restoration_fraction × RUL cap
Sensor projections use per-sensor linear maps sensor ≈ a + b·RUL fitted on training engines,
plus residual noise, so simulated sensors are consistent with the simulated degradation.

Bearing future simulation: an exponential trend fitted to the recent anomaly score is projected
forward (rate scaled by load^α) to estimate when it crosses the detector's alarm threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..classical_models.rul import RulModel
from ..counterfactual.scenarios import positive_normal
from ..datasets import cmapss
from ..risk.engine import assess_bearing, assess_turbofan
from ..settings import get_settings


@dataclass
class SimulationParams:
    horizon: int = 60
    load: float = 1.0
    maintenance_at: int | None = None      # time step at which maintenance happens (None = never)
    samples: int | None = None
    seed: int | None = None
    noise: float = 0.05


# ---------------------------------------------------------------- turbofan


@dataclass
class TurbofanTwin:
    model: RulModel
    data: cmapss.CmapssData
    unit: int
    split: str = "test"
    history: list[dict] = field(default_factory=list)   # maintenance history (from the database)

    @property
    def frame(self) -> pd.DataFrame:
        df = self.data.test if self.split == "test" else pd.concat([self.data.train, self.data.val])
        g = df[df["unit"] == self.unit].sort_values("cycle").reset_index(drop=True)
        if g.empty:
            raise ValueError(f"unit {self.unit} not found in {self.split} split")
        return g

    @property
    def asset_id(self) -> str:
        return f"{self.data.config.subset}-{self.split}-unit{self.unit}"

    def playback(self, raw: pd.DataFrame | None = None) -> dict:
        """Measured history with the model's prediction at every cycle."""
        g = self.frame
        pred = self.model.predict_frame(g)
        pof = self.model.failure_probability(pred)
        if raw is None:
            r_train, r_test, _ = cmapss.load_raw(self.data.config.subset)
            raw = r_test if self.split == "test" else r_train
        raw_u = raw[raw["unit"] == self.unit].sort_values("cycle").reset_index(drop=True)
        sensors = self.data.sensors_used
        series = []
        for i, row in g.iterrows():
            point = {"cycle": int(row["cycle"]), "source": "measured", "rul_pred": float(pred[i]),
                     "pof": float(pof[i]), "health": float(100 * min(pred[i] / self.model.rul_cap, 1)),
                     "sensors": {s: float(raw_u.loc[i, s]) for s in sensors}}
            if self.split == "train" or i == len(g) - 1 or "rul_true" in row:
                point["rul_true"] = float(row["rul_true"])
            series.append(point)
        return {"asset_id": self.asset_id, "unit": self.unit, "split": self.split,
                "sensors": sensors, "sensor_names": {s: cmapss.SENSOR_NAMES[s] for s in sensors},
                "series": series, "rul_true_available": True}

    def state(self) -> dict:
        g = self.frame
        rul = float(self.model.predict_last(g)[0])
        risk = assess_turbofan(self.asset_id, rul, self.model.sigma, self.model.q90, self.model.rul_cap)
        return {"asset_id": self.asset_id, "equipment_type": "turbofan", "unit": self.unit,
                "current_cycle": int(g["cycle"].iloc[-1]), "rul_pred": rul,
                "rul_true": float(g["rul_true"].iloc[-1]), "risk": risk.to_dict(),
                "model": self.model.name, "maintenance_history": self.history, "source": "measured"}

    def _sensor_maps(self) -> dict:
        """Linear sensor ≈ a + b·RUL per sensor (raw units), fitted on training engines in the degrading regime."""
        r_train, _, _ = cmapss.load_raw(self.data.config.subset)
        r_train = cmapss.add_train_rul(r_train, None)
        units = set(self.data.train["unit"].unique())
        fit = r_train[r_train["unit"].isin(units) & (r_train["rul_true"] <= self.model.rul_cap)]
        maps = {}
        for s in self.data.sensors_used:
            b, a = np.polyfit(fit["rul_true"], fit[s], 1)
            resid = fit[s] - (a + b * fit["rul_true"])
            maps[s] = (float(a), float(b), float(resid.std()))
        return maps

    def simulate(self, params: SimulationParams) -> dict:
        st = get_settings()
        sim = st["simulation"]
        costs = st["equipment"]["turbofan"]
        n = params.samples or sim["monte_carlo_samples"]
        rng = np.random.default_rng(params.seed if params.seed is not None else sim["seed"])
        rul0 = float(self.model.predict_last(self.frame)[0])
        start_cycle = int(self.frame["cycle"].iloc[-1])
        rate = params.load ** sim["load_exponent"]
        rul = positive_normal(rul0, self.model.sigma, n, rng)
        failed = np.zeros(n, bool)
        maps = self._sensor_maps()
        restored = costs["restoration_fraction"] * self.model.rul_cap
        series = []
        for t in range(1, params.horizon + 1):
            if params.maintenance_at is not None and t == params.maintenance_at:
                rul = np.where(failed, rul, positive_normal(restored, 5, n, rng))
            rul = rul - rate * (1 + rng.normal(0, params.noise, n))
            failed |= rul <= 0
            alive = np.maximum(rul, 0)
            med = float(np.median(alive))
            series.append({
                "cycle": start_cycle + t, "step": t, "source": "simulated",
                "rul_median": med, "rul_p10": float(np.quantile(alive, 0.1)),
                "rul_p90": float(np.quantile(alive, 0.9)), "cumulative_pof": float(failed.mean()),
                "health": float(100 * min(med / self.model.rul_cap, 1)),
                "maintenance": params.maintenance_at == t,
                "sensors": {s: a + b * med + rng.normal(0, sd) for s, (a, b, sd) in maps.items()},
            })
        return {"asset_id": self.asset_id, "start_cycle": start_cycle, "rul_start": rul0,
                "params": params.__dict__, "load_wear_rate": rate, "series": series,
                "final_pof": series[-1]["cumulative_pof"],
                "assumptions": __doc__.split("Turbofan future simulation")[1].split("Bearing future")[0].strip()}


# ---------------------------------------------------------------- bearing


@dataclass
class BearingTwin:
    detector: object            # AnomalyModel
    features: pd.DataFrame      # prepared IMS feature frame (ims.prepare_anomaly(...).features)
    test_id: str
    bearing: int
    history: list[dict] = field(default_factory=list)

    @property
    def asset_id(self) -> str:
        return f"IMS-{self.test_id}-bearing{self.bearing}"

    @property
    def frame(self) -> pd.DataFrame:
        g = self.features[self.features["bearing"] == self.bearing].sort_values("file_index")
        if g.empty:
            raise ValueError(f"bearing {self.bearing} not in {self.test_id}")
        return g

    def _scores(self, upto: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
        g = self.frame if upto is None else self.frame[self.frame["file_index"] <= upto]
        return g, self.detector.score(self.features.loc[g.index] if upto is None else g)

    def baseline_median(self) -> float:
        base = self.features[self.features["is_baseline"]]
        return float(np.median(self.detector.score(base)))

    def playback(self, upto: int | None = None, stride: int = 1) -> dict:
        g, sc = self._scores(upto)
        thr, base = self.detector.threshold, self.baseline_median()
        series = []
        for i, (_, row) in enumerate(g.iterrows()):
            if i % stride and i != len(g) - 1:
                continue
            series.append({"file_index": int(row["file_index"]), "timestamp": str(row["timestamp"]),
                           "source": "measured", "anomaly_score": float(sc[i]),
                           "health": float(100 * max(0, min(1, 1 - max((sc[i] - base) / max(thr - base, 1e-12), 0) / 3))),
                           "rms": float(row["rms"]), "kurtosis": float(row["kurtosis"]),
                           "env_bpfo": float(row["env_bpfo"]), "env_bpfi": float(row["env_bpfi"])})
        return {"asset_id": self.asset_id, "test": self.test_id, "bearing": self.bearing,
                "threshold": thr, "baseline_median": base, "series": series}

    def time_to_threshold(self, upto: int | None = None, load: float = 1.0, window: int = 60) -> float | None:
        """Hours until the projected score crosses the alarm threshold (simulated; None if no upward trend)."""
        g, sc = self._scores(upto)
        thr = self.detector.threshold
        if sc[-1] >= thr:
            return 0.0
        ts = pd.to_datetime(g["timestamp"]).to_numpy()
        hrs = (ts - ts[0]) / np.timedelta64(1, "h")
        w = slice(max(0, len(sc) - window), len(sc))
        floor = sc.min()
        slope, _ = np.polyfit(hrs[w], np.log(np.maximum(sc[w] - floor + 1e-6, 1e-6)), 1)
        slope *= load ** get_settings()["simulation"]["load_exponent"]
        if slope <= 1e-9:
            return None
        target = np.log(max(thr - floor + 1e-6, 1e-6))
        current = np.log(max(sc[-1] - floor + 1e-6, 1e-6))
        return float(max((target - current) / slope, 0.0))

    def state(self, upto: int | None = None) -> dict:
        g, sc = self._scores(upto)
        ttt = self.time_to_threshold(upto)
        risk = assess_bearing(self.asset_id, float(sc[-1]), self.baseline_median(), self.detector.threshold, ttt)
        return {"asset_id": self.asset_id, "equipment_type": "bearing", "test": self.test_id,
                "bearing": self.bearing, "file_index": int(g["file_index"].iloc[-1]),
                "timestamp": str(g["timestamp"].iloc[-1]), "anomaly_score": float(sc[-1]),
                "time_to_threshold_hours": ttt, "risk": risk.to_dict(), "model": self.detector.name,
                "maintenance_history": self.history, "source": "measured"}

    def simulate(self, params: SimulationParams, upto: int | None = None) -> dict:
        g, sc = self._scores(upto)
        thr = self.detector.threshold
        ts = pd.to_datetime(g["timestamp"]).to_numpy()
        hrs = (ts - ts[0]) / np.timedelta64(1, "h")
        w = slice(max(0, len(sc) - 60), len(sc))
        floor = sc.min()
        slope, _ = np.polyfit(hrs[w], np.log(np.maximum(sc[w] - floor + 1e-6, 1e-6)), 1)
        slope = max(slope, 0) * params.load ** get_settings()["simulation"]["load_exponent"]
        rng = np.random.default_rng(params.seed if params.seed is not None else get_settings()["simulation"]["seed"])
        level = sc[-1] - floor
        base = self.baseline_median() - floor
        series = []
        for t in range(1, params.horizon + 1):
            if params.maintenance_at is not None and t == params.maintenance_at:
                level = base   # replaced bearing returns to baseline behaviour
            level = level * np.exp(slope) * np.exp(rng.normal(0, params.noise))
            score = float(level + floor)
            series.append({"step": t, "hours_ahead": t, "source": "simulated", "anomaly_score": score,
                           "above_threshold": score >= thr, "maintenance": params.maintenance_at == t})
        crossing = next((p["hours_ahead"] for p in series if p["above_threshold"]), None)
        return {"asset_id": self.asset_id, "params": params.__dict__, "threshold": thr,
                "log_score_slope_per_hour": float(slope), "series": series,
                "hours_to_threshold": crossing,
                "assumptions": "Exponential trend of the last 60 anomaly scores, scaled by load^α; "
                               "maintenance resets the score to the healthy-baseline median."}
