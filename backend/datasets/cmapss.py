"""NASA C-MAPSS preprocessing, EDA and feature engineering.

Files (per sub-dataset FD001–FD004): ``train_FDxxx.txt``, ``test_FDxxx.txt``, ``RUL_FDxxx.txt``.
Each row: unit, cycle, 3 operational settings, 21 sensors.

Leakage rules applied here:
* the validation split is by engine unit, never by row;
* scalers, operating-condition clusters and dropped-sensor decisions are fitted on training units only;
* test RUL comes only from the official ``RUL_FDxxx.txt`` labels.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ..storage import find_files, paths, save_result
from ..storage.fileio import write_json
from ..storage.artifacts import save_dataframe

log = logging.getLogger("qadt.cmapss")

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]
SENSOR_NAMES = {
    "s1": "T2 fan inlet temp", "s2": "T24 LPC outlet temp", "s3": "T30 HPC outlet temp",
    "s4": "T50 LPT outlet temp", "s5": "P2 fan inlet pressure", "s6": "P15 bypass duct pressure",
    "s7": "P30 HPC outlet pressure", "s8": "Nf fan speed", "s9": "Nc core speed",
    "s10": "epr engine pressure ratio", "s11": "Ps30 HPC static pressure", "s12": "phi fuel/Ps30 ratio",
    "s13": "NRf corrected fan speed", "s14": "NRc corrected core speed", "s15": "BPR bypass ratio",
    "s16": "farB burner fuel-air ratio", "s17": "htBleed bleed enthalpy", "s18": "Nf_dmd demanded fan speed",
    "s19": "PCNfR_dmd demanded corrected fan speed", "s20": "W31 HPT coolant bleed", "s21": "W32 LPT coolant bleed",
}
# Number of operating regimes per subset (Saxena et al., 2008).
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass
class CmapssConfig:
    subset: str = "FD001"
    rul_cap: int = 125          # piecewise-linear RUL target
    window: int = 30            # sequence length for deep models
    rolling: int = 5            # rolling-feature window (cycles)
    val_fraction: float = 0.2   # fraction of *training units* held out for validation
    min_sensor_std: float = 1e-3
    seed: int = 42
    failure_horizon: int = 30   # "will fail within N cycles" label for probability-of-failure models


@dataclass
class CmapssData:
    config: CmapssConfig
    train: pd.DataFrame            # training units (features + RUL), normalised
    val: pd.DataFrame              # held-out training units
    test: pd.DataFrame             # official test set, all cycles
    test_last: pd.DataFrame        # last cycle of each test unit (the official evaluation points)
    feature_cols: list[str]
    sensors_used: list[str]
    dropped_sensors: list[str]
    scaler: dict = field(repr=False)


# ---------------------------------------------------------------- raw loading


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, :len(COLUMNS)]
    df.columns = COLUMNS
    return df


@lru_cache(maxsize=8)
def _locate(subset: str) -> dict:
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    found = {}
    for kind in ("train", "test", "RUL"):
        matches = find_files("cmapss", f"{kind}_{subset}.txt")
        if not matches:
            raise FileNotFoundError(f"{kind}_{subset}.txt not found in the extracted C-MAPSS dataset")
        found[kind] = matches[0]
    return found


def load_raw(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return raw train, raw test and the official true RUL of each test unit (dataset measurements)."""
    files = _locate(subset)
    train, test = _read(files["train"]), _read(files["test"])
    rul = pd.read_csv(files["RUL"], sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    if len(rul) != test["unit"].nunique():
        raise ValueError(f"{subset}: RUL file has {len(rul)} rows but test set has {test['unit'].nunique()} units")
    return train, test, rul


# ---------------------------------------------------------------- labels


def add_train_rul(df: pd.DataFrame, cap: int | None) -> pd.DataFrame:
    """Run-to-failure: RUL = last cycle of the unit − current cycle (optionally capped)."""
    df = df.copy()
    df["rul_true"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["rul"] = df["rul_true"].clip(upper=cap) if cap else df["rul_true"]
    return df


def add_test_rul(df: pd.DataFrame, rul_at_end: np.ndarray, cap: int | None) -> pd.DataFrame:
    """Test units stop before failure; true RUL at row = official RUL + (last cycle − cycle)."""
    df = df.copy()
    last = df.groupby("unit")["cycle"].transform("max")
    end_rul = df["unit"].map(dict(zip(sorted(df["unit"].unique()), rul_at_end)))
    df["rul_true"] = end_rul + (last - df["cycle"])
    df["rul"] = df["rul_true"].clip(upper=cap) if cap else df["rul_true"]
    return df


# ---------------------------------------------------------------- operating conditions


def fit_conditions(train: pd.DataFrame, n: int, seed: int):
    if n <= 1:
        return None
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n, n_init=10, random_state=seed).fit(train[SETTING_COLS].round(2))


def assign_conditions(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    df["op_condition"] = 0 if model is None else model.predict(df[SETTING_COLS].round(2))
    return df


# ---------------------------------------------------------------- pipeline


def _split_units(units: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    units = np.array(sorted(units))
    val = rng.choice(units, size=max(1, int(round(len(units) * fraction))), replace=False)
    return np.setdiff1d(units, val), np.sort(val)


def _fit_scaler(df: pd.DataFrame, cols: list[str]) -> dict:
    """Per operating-condition mean/std, fitted on training rows only."""
    scaler = {}
    for cond, g in df.groupby("op_condition"):
        std = g[cols].std().replace(0, 1).fillna(1)
        scaler[int(cond)] = {"mean": g[cols].mean().to_dict(), "std": std.to_dict()}
    return scaler


def _apply_scaler(df: pd.DataFrame, cols: list[str], scaler: dict) -> pd.DataFrame:
    df = df.copy()
    out = np.empty((len(df), len(cols)))
    for cond, idx in df.groupby("op_condition").groups.items():
        s = scaler.get(int(cond)) or next(iter(scaler.values()))
        mean = np.array([s["mean"][c] for c in cols])
        std = np.array([s["std"][c] for c in cols])
        out[df.index.get_indexer(idx)] = (df.loc[idx, cols].to_numpy() - mean) / std
    df[cols] = out
    return df


def _add_rolling(df: pd.DataFrame, cols: list[str], window: int) -> tuple[pd.DataFrame, list[str]]:
    g = df.groupby("unit")[cols]
    mean = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    std = g.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0)
    mean.columns = [f"{c}_rm{window}" for c in cols]
    std.columns = [f"{c}_rs{window}" for c in cols]
    out = pd.concat([df, mean.loc[df.index], std.loc[df.index]], axis=1)
    return out, [*mean.columns, *std.columns]


def prepare(config: CmapssConfig | None = None, save: bool = True) -> CmapssData:
    """Full C-MAPSS pipeline: load → label → condition-normalise → features → leakage-free split."""
    cfg = config or CmapssConfig()
    raw_train, raw_test, rul_end = load_raw(cfg.subset)
    train = add_train_rul(raw_train, cfg.rul_cap)
    test = add_test_rul(raw_test, rul_end, cfg.rul_cap)

    train_units, val_units = _split_units(train["unit"].unique(), cfg.val_fraction, cfg.seed)
    fit_rows = train[train["unit"].isin(train_units)]

    cond_model = fit_conditions(fit_rows, N_CONDITIONS[cfg.subset], cfg.seed)
    train, test = assign_conditions(train, cond_model), assign_conditions(test, cond_model)
    fit_rows = train[train["unit"].isin(train_units)]

    # Drop sensors that carry no information *within* an operating condition (decided on training units).
    within_std = fit_rows.groupby("op_condition")[SENSOR_COLS].std().max()
    dropped = [c for c in SENSOR_COLS if not within_std[c] > cfg.min_sensor_std]
    sensors = [c for c in SENSOR_COLS if c not in dropped]

    scaler = _fit_scaler(fit_rows, sensors)
    train, test = _apply_scaler(train, sensors, scaler), _apply_scaler(test, sensors, scaler)
    train, roll_cols = _add_rolling(train, sensors, cfg.rolling)
    test, _ = _add_rolling(test, sensors, cfg.rolling)
    for df in (train, test):
        df["cycle_norm"] = df["cycle"] / 400.0
        df["fail_within_h"] = (df["rul_true"] <= cfg.failure_horizon).astype(int)
    features = [*sensors, *roll_cols, "cycle_norm"]

    data = CmapssData(
        config=cfg,
        train=train[train["unit"].isin(train_units)].reset_index(drop=True),
        val=train[train["unit"].isin(val_units)].reset_index(drop=True),
        test=test.reset_index(drop=True),
        test_last=test.groupby("unit").tail(1).reset_index(drop=True),
        feature_cols=features, sensors_used=sensors, dropped_sensors=dropped, scaler=scaler,
    )
    if save:
        _save(data, raw_train)
    log.info("C-MAPSS %s: %d train / %d val / %d test units, %d features", cfg.subset,
             len(train_units), len(val_units), test["unit"].nunique(), len(features))
    return data


def _save(data: CmapssData, raw_train: pd.DataFrame) -> None:
    s = data.config.subset
    save_dataframe(raw_train, paths.cleaned("cmapss") / f"train_{s}_raw")
    for name, df in (("train", data.train), ("val", data.val), ("test", data.test)):
        save_dataframe(df, paths.splits("cmapss") / s / name)
    write_json(paths.splits("cmapss") / s / "pipeline.json", {
        "config": asdict(data.config), "feature_cols": data.feature_cols, "sensors_used": data.sensors_used,
        "dropped_sensors": data.dropped_sensors, "scaler": data.scaler,
        "train_units": sorted(map(int, data.train["unit"].unique())),
        "val_units": sorted(map(int, data.val["unit"].unique())),
    })


# ---------------------------------------------------------------- sequences for deep models


def make_sequences(df: pd.DataFrame, features: list[str], window: int, last_only: bool = False):
    """Sliding windows per unit. Units shorter than ``window`` are front-padded with their first row."""
    xs, ys, units = [], [], []
    for unit, g in df.groupby("unit", sort=True):
        arr = g[features].to_numpy(np.float32)
        y = g["rul"].to_numpy(np.float32)
        if len(arr) < window:
            pad = np.repeat(arr[:1], window - len(arr), axis=0)
            arr, y = np.vstack([pad, arr]), np.concatenate([np.repeat(y[:1], window - len(y)), y])
        ends = [len(arr)] if last_only else range(window, len(arr) + 1)
        for end in ends:
            xs.append(arr[end - window:end])
            ys.append(y[end - 1])
            units.append(unit)
    return np.stack(xs), np.array(ys, np.float32), np.array(units)


# ---------------------------------------------------------------- EDA


def eda(subset: str = "FD001", save: bool = True) -> dict:
    """Engine-wise, sensor and operating-condition analysis on the raw training data (measured values)."""
    train, test, rul_end = load_raw(subset)
    train = add_train_rul(train, None)
    life = train.groupby("unit")["cycle"].max()
    sensor_stats = []
    for c in SENSOR_COLS:
        col = train[c]
        sensor_stats.append({
            "sensor": c, "name": SENSOR_NAMES[c], "mean": float(col.mean()), "std": float(col.std()),
            "min": float(col.min()), "max": float(col.max()),
            "corr_with_rul": float(np.nan_to_num(col.corr(train["rul_true"]))),
            "informative": bool(col.std() > 1e-3),
        })
    cond_model = fit_conditions(train, N_CONDITIONS[subset], 42)
    conds = assign_conditions(train, cond_model)["op_condition"].value_counts().sort_index()

    # Mean degradation trajectory of the 4 most RUL-correlated sensors, aligned on cycles-to-failure.
    top = sorted(sensor_stats, key=lambda r: -abs(r["corr_with_rul"]))[:4]
    traj = (train[train["rul_true"] <= 200].groupby("rul_true")[[t["sensor"] for t in top]].mean()
            .sort_index(ascending=False).reset_index())
    example_unit = int(life.idxmax())
    report = {
        "subset": subset,
        "source": "NASA C-MAPSS (measured simulation outputs)",
        "train_units": int(train["unit"].nunique()), "test_units": int(test["unit"].nunique()),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "operating_conditions": int(N_CONDITIONS[subset]),
        "condition_counts": {str(k): int(v) for k, v in conds.items()},
        "life_cycles": {"min": int(life.min()), "max": int(life.max()), "mean": float(life.mean()),
                        "median": float(life.median())},
        "life_histogram": np.histogram(life, bins=12)[0].tolist(),
        "life_bin_edges": np.histogram(life, bins=12)[1].round(1).tolist(),
        "test_rul": {"min": int(rul_end.min()), "max": int(rul_end.max()), "mean": float(rul_end.mean())},
        "sensors": sensor_stats,
        "top_sensors": [t["sensor"] for t in top],
        "degradation_trajectory": traj.round(4).to_dict(orient="records"),
        "example_unit": {
            "unit": example_unit,
            "series": train[train["unit"] == example_unit][["cycle", *[t["sensor"] for t in top]]]
            .round(4).to_dict(orient="records"),
        },
    }
    if save:
        save_result("dataset_reports", f"cmapss_{subset}_eda", report)
    return report


def available_subsets() -> list[str]:
    out = []
    for s in SUBSETS:
        try:
            _locate(s)
            out.append(s)
        except FileNotFoundError:
            pass
    return out
