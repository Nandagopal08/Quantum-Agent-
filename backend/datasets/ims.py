"""NASA IMS bearing run-to-failure data: vibration feature extraction and anomaly-detection preparation.

Layout after extraction: ``1st_test/``, ``2nd_test/`` and ``4th_test/txt/`` (the third test ships as
``3rd_test.rar`` but unpacks to ``4th_test``). Each file is a 1-second snapshot, 20 480 samples at
20 kHz, named by timestamp (``2003.10.22.12.06.24``); columns are accelerometer channels.

IMS has no per-snapshot labels; only *which* bearing failed at the end of each test is known
(Qiu et al., 2006). For anomaly-detection evaluation we therefore use an explicit, configurable
labelling assumption, stored with every result:

* the first ``baseline_fraction`` of each test is treated as healthy (label 0) and used for training;
* for the bearing(s) that failed, the last ``degraded_fraction`` of the test is labelled anomalous (1);
* snapshots in between are unlabelled and excluded from metrics.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

from ..storage import paths, require_dataset, save_result
from ..storage.artifacts import load_dataframe, save_dataframe

log = logging.getLogger("qadt.ims")

FS = 20_000
TIMESTAMP_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")

# Rexnord ZA-2115 at 2000 rpm (Qiu et al., 2006): characteristic defect frequencies in Hz.
DEFECT_FREQS = {"bpfo": 236.4, "bpfi": 296.9, "bsf": 279.8, "ftf": 14.8}

TESTS = {
    "1st_test": {"channels": 8, "bearing_channels": {1: [0, 1], 2: [2, 3], 3: [4, 5], 4: [6, 7]},
                 "failed": {3: "inner race defect", 4: "roller element defect"}},
    "2nd_test": {"channels": 4, "bearing_channels": {1: [0], 2: [1], 3: [2], 4: [3]},
                 "failed": {1: "outer race failure"}},
    "3rd_test": {"channels": 4, "bearing_channels": {1: [0], 2: [1], 3: [2], 4: [3]},
                 "failed": {3: "outer race failure"}},
}
FOLDER_ALIASES = {"1st_test": "1st_test", "2nd_test": "2nd_test", "3rd_test": "3rd_test", "4th_test": "3rd_test"}


@dataclass
class ImsConfig:
    baseline_fraction: float = 0.25
    degraded_fraction: float = 0.10
    max_files: int | None = None   # subsample long tests (evenly spaced) for quick experiments
    workers: int = 8


# ---------------------------------------------------------------- discovery


def discover_tests(root: Path | None = None) -> dict[str, Path]:
    """Map test id → folder that directly contains its timestamp-named snapshot files."""
    root = root or require_dataset("ims_bearings")
    found: dict[str, Path] = {}
    for folder in sorted({p.parent for p in root.rglob("*") if p.is_file() and TIMESTAMP_RE.match(p.name)}):
        alias = next((FOLDER_ALIASES[part] for part in reversed(folder.parts) if part in FOLDER_ALIASES), None)
        if alias and alias not in found:
            found[alias] = folder
    return found


def list_snapshots(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if TIMESTAMP_RE.match(p.name)), key=lambda p: p.name)


def parse_timestamp(name: str) -> datetime:
    return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")


def read_snapshot(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float32).to_numpy()


# ---------------------------------------------------------------- features


def _band_amp(freqs: np.ndarray, spec: np.ndarray, f0: float, width: float = 5.0) -> float:
    m = (freqs >= f0 - width) & (freqs <= f0 + width)
    return float(spec[m].max()) if m.any() else 0.0


def signal_features(x: np.ndarray, fs: int = FS) -> dict[str, float]:
    """Time-domain, statistical, frequency-domain and envelope-spectrum features of one channel."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    absx = np.abs(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(absx.max())
    mean_abs = float(absx.mean()) or 1e-12
    feats = {
        "rms": rms, "std": float(x.std()), "peak": peak, "p2p": float(x.max() - x.min()),
        "kurtosis": float(stats.kurtosis(x)), "skewness": float(stats.skew(x)),
        "crest": peak / (rms or 1e-12), "shape": rms / mean_abs, "impulse": peak / mean_abs,
        "clearance": peak / (np.mean(np.sqrt(absx)) ** 2 or 1e-12),
        "entropy": float(stats.entropy(np.histogram(x, bins=64)[0] + 1e-12)),
    }
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    spec = np.abs(np.fft.rfft(x)) / len(x)
    power = spec ** 2
    total = power.sum() or 1e-12
    feats.update({
        "spec_centroid": float((freqs * power).sum() / total),
        "spec_spread": float(np.sqrt(((freqs - (freqs * power).sum() / total) ** 2 * power).sum() / total)),
        "dominant_freq": float(freqs[1:][np.argmax(spec[1:])]),
        "spec_kurtosis": float(stats.kurtosis(spec)),
    })
    for lo, hi in ((0, 1000), (1000, 3000), (3000, 6000), (6000, 10000)):
        feats[f"band_{lo // 1000}_{hi // 1000}k"] = float(power[(freqs >= lo) & (freqs < hi)].sum() / total)
    # Envelope spectrum: band-pass around the structural resonance, demodulate, look at defect frequencies.
    sos = signal.butter(4, [2000, 8000], btype="bandpass", fs=fs, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
    env_spec = np.abs(np.fft.rfft(env - env.mean())) / len(env)
    env_freqs = np.fft.rfftfreq(len(env), 1 / fs)
    for name, f0 in DEFECT_FREQS.items():
        feats[f"env_{name}"] = _band_amp(env_freqs, env_spec, f0)
    return feats


FEATURE_NAMES = list(signal_features(np.random.default_rng(0).normal(size=2048)).keys())


def _snapshot_rows(path: Path, index: int, test_id: str) -> list[dict]:
    data = read_snapshot(path)
    spec = TESTS[test_id]
    rows = []
    for bearing, chans in spec["bearing_channels"].items():
        per_channel = [signal_features(data[:, c]) for c in chans if c < data.shape[1]]
        feats = {k: float(np.mean([f[k] for f in per_channel])) for k in per_channel[0]}
        rows.append({"test": test_id, "file_index": index, "timestamp": parse_timestamp(path.name),
                     "bearing": bearing, **feats})
    return rows


def extract_features(test_id: str, folder: Path | None = None, config: ImsConfig | None = None,
                     force: bool = False, progress=None) -> pd.DataFrame:
    """Feature table (one row per snapshot × bearing), cached in Drive under datasets/features/ims_bearings."""
    cfg = config or ImsConfig()
    cache = paths.features("ims_bearings") / f"{test_id}_features"
    suffix = f"_n{cfg.max_files}" if cfg.max_files else ""
    cache = cache.with_name(cache.name + suffix)
    if not force:
        for ext in (".parquet", ".csv"):
            if cache.with_suffix(ext).exists():
                return _with_labels(load_dataframe(cache.with_suffix(ext)), test_id, cfg)
    folder = folder or discover_tests().get(test_id)
    if folder is None:
        raise FileNotFoundError(f"IMS test {test_id} not found in the extracted dataset")
    files = list_snapshots(folder)
    if cfg.max_files and len(files) > cfg.max_files:
        files = [files[i] for i in np.linspace(0, len(files) - 1, cfg.max_files).round().astype(int)]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        for i, part in enumerate(pool.map(lambda a: _snapshot_rows(a[1], a[0], test_id), enumerate(files))):
            rows.extend(part)
            if progress and i % 25 == 0:
                progress(i / len(files))
    df = pd.DataFrame(rows).sort_values(["bearing", "file_index"]).reset_index(drop=True)
    save_dataframe(df, cache)
    log.info("IMS %s: %d snapshots × %d bearings → %d features", test_id, len(files),
             df["bearing"].nunique(), len(FEATURE_NAMES))
    return _with_labels(df, test_id, cfg)


def _with_labels(df: pd.DataFrame, test_id: str, cfg: ImsConfig) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n = df["file_index"].max() + 1
    df["life_fraction"] = df["file_index"] / max(n - 1, 1)
    failed = TESTS[test_id]["failed"]
    df["failed_bearing"] = df["bearing"].isin(failed).astype(int)
    df["label"] = np.nan
    df.loc[df["life_fraction"] <= cfg.baseline_fraction, "label"] = 0
    late = df["life_fraction"] >= 1 - cfg.degraded_fraction
    df.loc[late & (df["failed_bearing"] == 1), "label"] = 1
    df.loc[late & (df["failed_bearing"] == 0), "label"] = 0
    df["is_baseline"] = df["life_fraction"] <= cfg.baseline_fraction
    return df


@dataclass
class ImsAnomalyData:
    test_id: str
    features: pd.DataFrame
    feature_cols: list[str]
    train_X: np.ndarray          # healthy baseline, all bearings (unsupervised training)
    eval_df: pd.DataFrame        # labelled rows only
    mean: np.ndarray
    std: np.ndarray
    config: ImsConfig


def prepare_anomaly(test_id: str = "2nd_test", config: ImsConfig | None = None, save: bool = True) -> ImsAnomalyData:
    cfg = config or ImsConfig()
    df = extract_features(test_id, config=cfg)
    cols = FEATURE_NAMES
    base = df[df["is_baseline"]]
    # log-scale heavy-tailed amplitude features, standardise on the healthy baseline only
    X = np.log1p(np.abs(df[cols].to_numpy())) * np.sign(df[cols].to_numpy())
    mean = X[df["is_baseline"].to_numpy()].mean(0)
    std = X[df["is_baseline"].to_numpy()].std(0) + 1e-9
    Z = (X - mean) / std
    df = df.copy()
    df[[f"z_{c}" for c in cols]] = Z
    zcols = [f"z_{c}" for c in cols]
    data = ImsAnomalyData(test_id=test_id, features=df, feature_cols=zcols,
                          train_X=df.loc[base.index, zcols].to_numpy(), eval_df=df[df["label"].notna()],
                          mean=mean, std=std, config=cfg)
    if save:
        save_result("dataset_reports", f"ims_{test_id}_labelling", {
            "test": test_id, "assumption": __doc__.split("therefore use")[1].strip() if __doc__ else "",
            "config": asdict(cfg), "failed_bearings": TESTS[test_id]["failed"],
            "rows": int(len(df)), "baseline_rows": int(df["is_baseline"].sum()),
            "labelled_rows": int(df["label"].notna().sum()), "anomalous_rows": int((df["label"] == 1).sum()),
        })
    return data


def eda(test_id: str = "2nd_test", config: ImsConfig | None = None, save: bool = True) -> dict:
    df = extract_features(test_id, config=config)
    trend_cols = ["rms", "kurtosis", "env_bpfo", "env_bpfi", "spec_centroid"]
    trend = df.groupby(["file_index", "bearing"])[trend_cols].mean().reset_index()
    stride = max(1, int(np.ceil(trend["file_index"].nunique() / 600)))   # ≤ ~600 points per bearing
    trend = trend[(trend["file_index"] % stride == 0) | (trend["file_index"] == trend["file_index"].max())]
    ts = df.groupby("file_index")["timestamp"].first()
    report = {
        "test": test_id, "source": "NASA IMS bearing vibration (measured)",
        "snapshots": int(df["file_index"].nunique()), "bearings": int(df["bearing"].nunique()),
        "start": str(ts.min()), "end": str(ts.max()),
        "duration_hours": float((ts.max() - ts.min()).total_seconds() / 3600),
        "failed_bearings": {str(k): v for k, v in TESTS[test_id]["failed"].items()},
        "feature_names": FEATURE_NAMES,
        "feature_summary": df.groupby("bearing")[trend_cols].agg(["mean", "max"]).round(5)
        .pipe(lambda t: {f"{b}": {f"{c}_{s}": float(t.loc[b, (c, s)]) for c, s in t.columns} for b in t.index}),
        "trend": trend.round(6).to_dict(orient="records"), "trend_stride": stride,
    }
    if save:
        save_result("dataset_reports", f"ims_{test_id}_eda", report)
    return report


def available_tests() -> list[str]:
    try:
        return sorted(discover_tests())
    except Exception:  # dataset not ready
        return []
