"""Experiments for the quantum sequence models on NASA C-MAPSS sensor time series.

1. **RUL prediction** (complete data): each model reads a window of the normalised sensors of one
   engine and predicts the (capped) remaining useful life at its last cycle. Evaluated on the official
   test set (last window of each test engine vs ``RUL_FDxxx.txt``).
2. **Imputation** (missing data): values are removed with a known mask (MCAR or block outages) at
   10–50 % missing rates. Learned imputers receive ``[linear-interpolated window, mask]`` and predict a
   residual correction to the interpolation; observed values are always kept. They are trained once on
   random rates in ``train_rate_range`` and both patterns (one rate-agnostic model per architecture),
   then evaluated per rate/pattern on held-out *test* engines with fixed, seeded masks. Errors are
   computed only on the removed values, in standardised sensor units.
3. **Downstream RUL under missing data**: the test windows are masked, imputed by every method and
   fed to the best RUL model (chosen on validation units) — showing how imputation quality carries
   through to the maintenance decision.

All model selection uses validation engines; the test engines are touched only for the final numbers.
Because C-MAPSS is complete, the true values behind every mask are known exactly.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import torch

from ..classical_models.metrics import regression_metrics
from ..datasets import cmapss
from ..missing_data import BASELINE_IMPUTERS, PATTERNS, impute_baseline, linear_fill, make_mask
from ..storage import ExperimentLogger, save_model, save_result
from ..storage.artifacts import load_model
from ..storage.fileio import read_json
from ..storage import paths
from . import stats
from .qseq import (SEQ_MODELS, SeqModelConfig, ablate_circuits, block_gradient_norms, build_model, model_from_state,
                   state_to_numpy)

log = logging.getLogger("qadt.qseq")
Progress = Callable[[float, str], None]
REPORT_NAME = "qseq_benchmark_{subset}"


@dataclass
class SeqExperimentConfig:
    subset: str = "FD001"
    window: int = 24
    n_qubits: int = 4
    layers: int = 2
    hidden: int = 8
    n_train: int = 1200              # training windows (random subsample of all training-engine windows)
    n_val: int = 400
    n_eval: int = 400                # test windows for the imputation benchmark
    epochs: int = 10
    imputation_epochs: int = 8
    batch: int = 32
    lr: float = 0.01
    seed: int = 42
    models: tuple[str, ...] = SEQ_MODELS
    missing_rates: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    patterns: tuple[str, ...] = PATTERNS
    train_rate_range: tuple[float, float] = (0.05, 0.6)
    run_rul: bool = True
    run_imputation: bool = True
    n_boot: int = 2000               # bootstrap resamples for paired confidence intervals
    run_tag: str = ""                # appended to training-log names so re-runs never mix records
    save_logs: bool = True

    def model_config(self, name: str, n_features: int, task: str) -> SeqModelConfig:
        return SeqModelConfig(name=name, n_features=n_features, n_out=1 if task == "regression" else n_features,
                              task=task, window=self.window, n_qubits=self.n_qubits, layers=self.layers,
                              hidden=self.hidden)


# ---------------------------------------------------------------- data


def load_sequences(cfg: SeqExperimentConfig) -> dict:
    data = cmapss.prepare(cmapss.CmapssConfig(subset=cfg.subset, window=cfg.window, seed=cfg.seed), save=False)
    sensors = data.sensors_used
    rng = np.random.default_rng(cfg.seed)

    def windows(df, n=None, last_only=False):
        X, y, units = cmapss.make_sequences(df, sensors, cfg.window, last_only=last_only)
        if n is not None and len(X) > n:
            idx = np.sort(rng.choice(len(X), n, replace=False))
            X, y, units = X[idx], y[idx], units[idx]
        return X.astype(np.float64), y.astype(np.float64), units

    Xtr, ytr, _ = windows(data.train, cfg.n_train)
    Xva, yva, _ = windows(data.val, cfg.n_val)
    Xte, yte, ute = windows(data.test, last_only=True)
    Xev, _, uev = windows(data.test, cfg.n_eval)
    train_means = data.train[sensors].to_numpy(float).mean(0)
    # uncapped true RUL of each official test engine (the capped value is the training/evaluation target)
    true_by_unit = dict(zip(data.test_last["unit"].astype(int), data.test_last["rul_true"].astype(float)))
    return {"X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva, "X_test": Xte, "y_test": yte,
            "test_units": ute, "y_test_uncapped": np.array([true_by_unit[int(u)] for u in ute]),
            "X_eval": Xev, "eval_units": uev, "sensors": sensors, "cap": float(data.config.rul_cap),
            "means": train_means, "data": data}


class _NullLogger:
    """Stand-in for ExperimentLogger when a run must not write logs (e.g. --quick checks)."""

    def log(self, **record):
        pass


def _logger(name: str, config: dict, cfg: "SeqExperimentConfig"):
    if not cfg.save_logs:
        return _NullLogger()
    return ExperimentLogger(f"{name}_{cfg.run_tag}", config=config) if cfg.run_tag else ExperimentLogger(name, config=config)


def _t(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


# ---------------------------------------------------------------- generic training loop


def _train(model, batches: Callable[[np.random.Generator], list], val_score: Callable[[], float], epochs: int,
           lr: float, seed: int, logger: ExperimentLogger, progress: Progress, label: str) -> dict:
    """Adam + grad clipping; keeps the parameters with the best validation score (lower is better)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best, best_state, history = np.inf, None, []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        losses, grads = [], []
        for loss_fn in batches(rng):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            if epoch in (0, epochs - 1):     # gradient reaching the circuit (or the twin's counterpart), pre-clipping
                grads.append(block_gradient_norms(model)["mean"])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            score = float(val_score())
        if score < best:
            best, best_state = score, copy.deepcopy(model.state_dict())
        rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val": score,
               "seconds": round(time.time() - t0, 1)}
        if grads:
            rec["block_grad_norm"] = float(np.mean(grads))
        history.append(rec)
        logger.log(**rec)
        progress((epoch + 1) / epochs, f"{label} epoch {epoch + 1}/{epochs} loss {rec['train_loss']:.4f} val {score:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return {"best_val": best, "train_seconds": round(time.time() - t0, 1), "history": history,
            "best_epoch": int(np.argmin([h["val"] for h in history]) + 1)}


def _minibatches(n: int, batch: int, rng: np.random.Generator):
    perm = rng.permutation(n)
    return [perm[i:i + batch] for i in range(0, n, batch)]


# ---------------------------------------------------------------- task 1: RUL prediction


def predict_rul_windows(model, X: np.ndarray, cap: float, batch: int = 256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(_t(X[i:i + batch])).numpy())
    return np.clip(np.concatenate(out) * cap, 0, cap)


def train_rul(name: str, cfg: SeqExperimentConfig, d: dict, progress: Progress, seed: int) -> tuple[dict, object]:
    mcfg = cfg.model_config(name, len(d["sensors"]), "regression")
    model = build_model(mcfg, seed=seed)
    cap = d["cap"]
    Xtr, ytr = _t(d["X_train"]), _t(d["y_train"] / cap)
    logger = _logger(f"qseq_rul_{name}_{cfg.subset}_s{seed}", mcfg.to_dict() | {"seed": seed}, cfg)

    def batches(rng):
        return [(lambda b=b: torch.nn.functional.mse_loss(model(Xtr[b]), ytr[b]))
                for b in _minibatches(len(Xtr), cfg.batch, rng)]

    def val():
        p = predict_rul_windows(model, d["X_val"], cap)
        return float(np.sqrt(np.mean((p - d["y_val"]) ** 2)))

    fit = _train(model, batches, val, cfg.epochs, cfg.lr, seed, logger, progress, f"RUL {name}")
    model.reset_calls()
    t1 = time.time()
    pred = predict_rul_windows(model, d["X_test"], cap)
    infer_s = time.time() - t1
    calls = model.calls()
    val_pred = predict_rul_windows(model, d["X_val"], cap)
    sigma = float(np.std(val_pred - d["y_val"]))
    with ablate_circuits(model):            # how much do predictions depend on the circuits / counterparts?
        pred_ablated = predict_rul_windows(model, d["X_test"], cap)
    res = {"model": name, "seed": seed, "test": regression_metrics(d["y_test"], pred),
           "test_uncapped": regression_metrics(d["y_test_uncapped"], pred),
           "ablation": {"rmse_ablated": regression_metrics(d["y_test"], pred_ablated)["rmse"],
                        "mean_abs_change_cycles": float(np.mean(np.abs(pred_ablated - pred)))},
           "first_epoch_grad_norm": fit["history"][0].get("block_grad_norm"),
           "last_epoch_grad_norm": fit["history"][-1].get("block_grad_norm"),
           "val_rmse": fit["best_val"], "sigma": sigma, "train_seconds": fit["train_seconds"],
           "best_epoch": fit["best_epoch"], "inference_seconds": round(infer_s, 3),
           "circuit_calls_test": calls if mcfg.quantum else 0, **model.describe(),
           "history": fit["history"],
           "predictions": [{"unit": int(u), "rul_true": float(t), "rul_true_uncapped": float(tu), "rul_pred": float(p)}
                           for u, t, tu, p in zip(d["test_units"], d["y_test"], d["y_test_uncapped"], pred)]}
    return res, model


# ---------------------------------------------------------------- task 2: imputation


def _random_masks(shape, rng, cfg: SeqExperimentConfig) -> np.ndarray:
    lo, hi = cfg.train_rate_range
    rates = rng.uniform(lo, hi, shape[0])
    pattern_of = rng.integers(0, len(PATTERNS), shape[0])
    mask = np.ones(shape)
    for k, p in enumerate(PATTERNS):
        sel = pattern_of == k
        if sel.any():
            mask[sel] = make_mask((int(sel.sum()), *shape[1:]), rates[sel], p, rng)
    return mask


def model_impute(model, x: np.ndarray, mask: np.ndarray, means: np.ndarray, batch: int = 256) -> np.ndarray:
    """Learned imputation: residual correction on top of linear interpolation; observed values kept."""
    base = linear_fill(np.where(mask > 0, x, 0.0), mask, means)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            inp = np.concatenate([base[i:i + batch], mask[i:i + batch]], axis=-1)
            out.append(model(_t(inp)).numpy())
    pred = base + np.concatenate(out)
    return np.where(mask > 0, x, pred)


def train_imputer(name: str, cfg: SeqExperimentConfig, d: dict, progress: Progress, seed: int) -> tuple[dict, object]:
    C = len(d["sensors"])
    mcfg = cfg.model_config(name, C, "sequence")
    mcfg.n_features = 2 * C          # values + mask indicator
    mcfg.n_out = C
    model = build_model(mcfg, seed=seed)
    X = d["X_train"]
    means = d["means"]
    logger = _logger(f"qseq_impute_{name}_{cfg.subset}_s{seed}", mcfg.to_dict() | {"seed": seed}, cfg)
    vrng = np.random.default_rng(seed + 1)
    val_mask = _random_masks(d["X_val"].shape, vrng, cfg)

    def batches(rng):
        mask = _random_masks(X.shape, rng, cfg)        # fresh masks every epoch (augmentation)
        base = linear_fill(np.where(mask > 0, X, 0.0), mask, means)
        inp, tgt, miss = _t(np.concatenate([base, mask], -1)), _t(X - base), _t(1 - mask)

        def loss(b):
            err = (model(inp[b]) - tgt[b]) ** 2 * miss[b]
            return err.sum() / miss[b].sum().clamp(min=1)
        return [(lambda b=b: loss(b)) for b in _minibatches(len(X), cfg.batch, rng)]

    def val():
        imp = model_impute(model, d["X_val"], val_mask, means)
        m = val_mask == 0
        return float(np.sqrt(np.mean((imp[m] - d["X_val"][m]) ** 2)))

    fit = _train(model, batches, val, cfg.imputation_epochs, cfg.lr, seed, logger, progress, f"impute {name}")
    return {"model": name, "seed": seed, "val_rmse": fit["best_val"], "train_seconds": fit["train_seconds"],
            "best_epoch": fit["best_epoch"], "final_train_loss": fit["history"][-1]["train_loss"],
            "first_epoch_grad_norm": fit["history"][0].get("block_grad_norm"),
            "last_epoch_grad_norm": fit["history"][-1].get("block_grad_norm"),
            **model.describe(), "history": fit["history"]}, model


def impute_any(method: str, x: np.ndarray, mask: np.ndarray, means: np.ndarray, models: dict) -> np.ndarray:
    if method in BASELINE_IMPUTERS:
        return impute_baseline(method, x, mask, means)
    return model_impute(models[method], x, mask, means)


def evaluation_masks(seed: int, rate: float, pattern: str, eval_shape, test_shape) -> tuple[np.ndarray, np.ndarray]:
    """The fixed, seeded evaluation masks (imputation windows, downstream test windows) for one condition.
    Used by the benchmark and by scripts/verify_results.py, so both see identical masks."""
    rng = np.random.default_rng(seed * 1000 + int(round(rate * 100)) + (17 if pattern == "block" else 0))
    return make_mask(eval_shape, rate, pattern, rng), make_mask(test_shape, rate, pattern, rng)


def window_errors(x_true: np.ndarray, x_imp: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window sum of squared errors and count of removed values (for paired statistics)."""
    m = mask == 0
    se = np.where(m, (x_imp - x_true) ** 2, 0.0).sum(axis=(1, 2))
    return se, m.sum(axis=(1, 2)).astype(float)


def imputation_error(x_true: np.ndarray, x_imp: np.ndarray, mask: np.ndarray) -> dict:
    m = mask == 0
    if not m.any():
        return {"rmse": 0.0, "mae": 0.0, "n_missing": 0}
    e = x_imp[m] - x_true[m]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "n_missing": int(m.sum())}


# ---------------------------------------------------------------- full benchmark


def _aggregate(rows: list[dict], keys: tuple[str, ...], metric_paths: dict[str, Callable[[dict], float]]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        rec = dict(zip(keys, g))
        for name, get in metric_paths.items():
            vals = np.array([get(r) for r in rs], float)
            rec[name] = float(vals.mean())
            rec[f"{name}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else None
        rec["n_seeds"] = len(rs)
        out.append(rec)
    return out


def run_benchmark(cfg: SeqExperimentConfig | None = None, seeds: tuple[int, ...] | None = None,
                  progress: Progress | None = None, save: bool = True) -> dict:
    """Train/evaluate every model for every seed; returns (and saves) the report.

    Saved artefacts (when ``save``): ``comparison_reports/qseq_benchmark_<subset>.json`` (summary),
    ``comparison_reports/qseq_raw_<subset>.json`` (per-seed per-engine predictions and per-window imputation
    errors, from which every metric can be recomputed), and the first seed's models. Previous versions of
    reports and models are kept in ``_history`` folders (see storage.artifacts).
    """
    cfg = cfg or SeqExperimentConfig()
    seeds = tuple(seeds or (cfg.seed,))
    bad = [m for m in cfg.models if m not in SEQ_MODELS]
    if bad:
        raise ValueError(f"unknown models {bad}; expected {SEQ_MODELS}")
    if not save:
        cfg.save_logs = False
    if not cfg.run_tag:
        cfg.run_tag = time.strftime("%Y%m%d_%H%M%S")
    progress = progress or (lambda f, m: None)
    progress(0.01, "preparing C-MAPSS sensor windows")
    d = load_sequences(cfg)
    n_steps = len(seeds) * len(cfg.models) * (int(cfg.run_rul) + int(cfg.run_imputation))
    step = [0]

    def sub(label):
        def p(f, msg):
            progress(0.02 + 0.85 * (step[0] + f) / max(1, n_steps), f"[{label}] {msg}")
        return p

    t_start = time.time()
    rul_rows, imp_rows, eval_rows, down_rows = [], [], [], []
    raw = {"rul_predictions": {}, "imputation_errors": {}, "downstream": None, "eval_units": d["eval_units"].tolist(),
           "test_units": d["test_units"].tolist()}
    methods = list(BASELINE_IMPUTERS) + (list(cfg.models) if cfg.run_imputation else [])
    for seed in seeds:
        rul_models, imp_models = {}, {}
        if cfg.run_rul:
            for name in cfg.models:
                res, model = train_rul(name, cfg, d, sub(f"seed {seed}"), seed)
                raw["rul_predictions"].setdefault(name, {})[str(seed)] = res.pop("predictions")
                rul_rows.append(res)
                rul_models[name] = model
                if save:   # every seed (for independent recomputation); the first seed also under the plain name
                    bundle = {"config": model.cfg.to_dict(), "state": state_to_numpy(model), "sigma": res["sigma"],
                              "cap": d["cap"], "sensors": d["sensors"], "window": cfg.window, "seed": seed,
                              "run_tag": cfg.run_tag}
                    names = [f"qseq_rul_{name}_{cfg.subset}_s{seed}"] + ([f"qseq_rul_{name}_{cfg.subset}"]
                                                                       if seed == seeds[0] else [])
                    for nm in names:
                        save_model(bundle, "quantum_ai", nm, config=model.cfg.to_dict(), metrics=res["test"])
                step[0] += 1
        if cfg.run_imputation:
            for name in cfg.models:
                res, model = train_imputer(name, cfg, d, sub(f"seed {seed}"), seed)
                imp_rows.append(res)
                imp_models[name] = model
                if save:
                    bundle = {"config": model.cfg.to_dict(), "state": state_to_numpy(model), "means": d["means"],
                              "sensors": d["sensors"], "window": cfg.window, "seed": seed, "run_tag": cfg.run_tag}
                    names = [f"qseq_impute_{name}_{cfg.subset}_s{seed}"] + ([f"qseq_impute_{name}_{cfg.subset}"]
                                                                          if seed == seeds[0] else [])
                    for nm in names:
                        save_model(bundle, "quantum_ai", nm, config=model.cfg.to_dict(),
                                   metrics={"val_rmse": res["val_rmse"]})
                step[0] += 1

            # imputation accuracy per rate × pattern (fixed seeded masks, identical for every method)
            for pattern in cfg.patterns:
                for rate in cfg.missing_rates:
                    mask, mask_test = evaluation_masks(seed, rate, pattern, d["X_eval"].shape, d["X_test"].shape)
                    for method in methods:
                        imp = impute_any(method, d["X_eval"], mask, d["means"], imp_models)
                        se, n = window_errors(d["X_eval"], imp, mask)
                        raw["imputation_errors"].setdefault(f"{pattern}|{rate}|{method}", {})[str(seed)] = \
                            {"se": se.round(8).tolist(), "n": n.astype(int).tolist()}
                        row = {"seed": seed, "pattern": pattern, "rate": rate, "method": method,
                               "observed_rate": float(1 - mask.mean()), **imputation_error(d["X_eval"], imp, mask)}
                        if method in imp_models and pattern == "mcar" and rate == 0.3:
                            with ablate_circuits(imp_models[method]):
                                abl = impute_any(method, d["X_eval"], mask, d["means"], imp_models)
                            row["rmse_ablated"] = imputation_error(d["X_eval"], abl, mask)["rmse"]
                        eval_rows.append(row)
                        if rul_models:     # downstream: every trained RUL model, every seed (like-for-like)
                            imp_t = impute_any(method, d["X_test"], mask_test, d["means"], imp_models)
                            for rm, rmodel in rul_models.items():
                                pred = predict_rul_windows(rmodel, imp_t, d["cap"])
                                down_rows.append({"seed": seed, "pattern": pattern, "rate": rate, "method": method,
                                                  "rul_model": rm,
                                                  "rul_rmse": regression_metrics(d["y_test"], pred)["rmse"]})
        for rm, rmodel in rul_models.items():
            clean = predict_rul_windows(rmodel, d["X_test"], d["cap"])
            down_rows.append({"seed": seed, "pattern": "none", "rate": 0.0, "method": "complete_data",
                              "rul_model": rm, "rul_rmse": regression_metrics(d["y_test"], clean)["rmse"]})

    raw["downstream"] = down_rows              # per-seed values behind every downstream mean
    progress(0.9, "aggregating results and paired statistics")
    rul_summary = sorted(_aggregate(rul_rows, ("model",), {
        "rmse": lambda r: r["test"]["rmse"], "mae": lambda r: r["test"]["mae"], "r2": lambda r: r["test"]["r2"],
        "nasa_score": lambda r: r["test"]["nasa_score"], "val_rmse": lambda r: r["val_rmse"],
        "rmse_uncapped": lambda r: r["test_uncapped"]["rmse"], "mae_uncapped": lambda r: r["test_uncapped"]["mae"],
        "rmse_ablated": lambda r: r["ablation"]["rmse_ablated"],
        "train_seconds": lambda r: r["train_seconds"], "inference_seconds": lambda r: r["inference_seconds"]}),
        key=lambda r: r["val_rmse"])
    for r in rul_summary:
        first = next(x for x in rul_rows if x["model"] == r["model"])
        r.update({k: first.get(k) for k in ("quantum", "arch", "trainable_params", "quantum_params", "circuits",
                                            "circuit", "n_qubits", "circuit_layers")})
        r["first_epoch_grad_norm"] = float(np.mean([x["first_epoch_grad_norm"] or 0 for x in rul_rows
                                                    if x["model"] == r["model"]]))
    imp_eval = _aggregate(eval_rows, ("pattern", "rate", "method"),
                          {"rmse": lambda r: r["rmse"], "mae": lambda r: r["mae"],
                           "observed_rate": lambda r: r["observed_rate"]})
    abl = _aggregate([r for r in eval_rows if "rmse_ablated" in r], ("method",),
                     {"rmse": lambda r: r["rmse"], "rmse_ablated": lambda r: r["rmse_ablated"]})
    downstream = _aggregate(down_rows, ("pattern", "rate", "method", "rul_model"), {"rul_rmse": lambda r: r["rul_rmse"]})
    imp_summary = _aggregate(imp_rows, ("model",), {"val_rmse": lambda r: r["val_rmse"],
                                                    "train_seconds": lambda r: r["train_seconds"],
                                                    "final_train_loss": lambda r: r["final_train_loss"],
                                                    "first_epoch_grad_norm": lambda r: r["first_epoch_grad_norm"] or 0.0,
                                                    "last_epoch_grad_norm": lambda r: r["last_epoch_grad_norm"] or 0.0})
    for r in imp_summary:
        first = next(x for x in imp_rows if x["model"] == r["model"])
        r.update({k: first.get(k) for k in ("quantum", "arch", "trainable_params", "quantum_params", "circuits")})
        a = next((x for x in abl if x["method"] == r["model"]), None)
        if a:
            r["ablation_mcar30"] = {"rmse": a["rmse"], "rmse_ablated": a["rmse_ablated"]}

    report = {
        "name": REPORT_NAME.format(subset=cfg.subset), "dataset": f"NASA C-MAPSS {cfg.subset}",
        "run_tag": cfg.run_tag, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "seeds": list(seeds), "sensors": d["sensors"],
        "data": {"n_train_windows": len(d["X_train"]), "n_val_windows": len(d["X_val"]),
                 "n_test_engines": len(d["X_test"]), "n_eval_windows": len(d["X_eval"]), "window": cfg.window,
                 "n_eval_engines": int(len(np.unique(d["eval_units"]))),
                 "rul_cap": d["cap"], "test_engines_above_cap": int((d["y_test_uncapped"] > d["cap"]).sum()),
                 "label_convention": "targets and the primary test metrics use RUL capped at rul_cap (piecewise-linear "
                                     "convention); *_uncapped metrics compare the same predictions with the raw "
                                     "RUL_FDxxx.txt values",
                 "seed_controls": "seeds change weight initialisation, mini-batch order, training masks and evaluation "
                                  "masks; the window subsample and the engine split are fixed (paired design)"},
        "rul": {"summary": rul_summary, "runs": rul_rows,
                "predictions": {m: v[str(seeds[0])] for m, v in raw["rul_predictions"].items()}},
        "imputation": {"models": imp_summary, "evaluation": imp_eval, "runs": imp_rows,
                       "ranking": rank_imputers_table(imp_eval)},
        "downstream_rul": downstream,
        "quantum_vs_twin": _twin_comparison(rul_summary, imp_eval),
        "statistics": _paired_statistics(cfg, raw, d, seeds),
        "total_seconds": round(time.time() - t_start, 1),
        "note": ("Statevector simulation (PennyLane default.qubit). Classical twins share the architecture with the "
                 "circuit replaced by a same-width classical layer. Differences are measured, not evidence of "
                 "quantum advantage."),
    }
    if save:
        save_result("comparison_reports", report["name"], report)
        save_result("comparison_reports", f"qseq_raw_{cfg.subset}", {"name": f"qseq_raw_{cfg.subset}",
                                                                     "run_tag": cfg.run_tag, "seeds": list(seeds), **raw})
    progress(1.0, "done")
    return report


def _paired_statistics(cfg: SeqExperimentConfig, raw: dict, d: dict, seeds) -> dict:
    """Paired quantum-vs-twin and learned-vs-linear comparisons with engine-cluster bootstrap CIs."""
    out: dict = {"method": "paired; 95% percentile bootstrap over engines (clusters), same resample for all seeds; "
                           "Wilcoxon signed-rank on per-engine MSE differences averaged over seeds; Holm-adjusted "
                           "within each family; seed-level paired-t CI; 'significant' requires Holm p < 0.05 and "
                           "both CIs excluding 0",
                 "n_boot": cfg.n_boot, "rul": [], "imputation": []}
    S = [str(s) for s in seeds]
    preds = raw["rul_predictions"]
    units = np.asarray(raw["test_units"])
    for arch in ("lstm", "transformer", "cnn_lstm"):
        a, b = f"q_{arch}", f"c_{arch}"
        if a in preds and b in preds and all(s in preds[a] and s in preds[b] for s in S):
            def se(m):
                return np.array([[(p["rul_pred"] - p["rul_true"]) ** 2 for p in preds[m][s]] for s in S])
            c = stats.paired_rmse_difference(se(a), np.ones((len(S), len(units))), se(b),
                                             np.ones((len(S), len(units))), units, cfg.n_boot)
            out["rul"].append({"a": a, "b": b, "metric": "test RMSE (cycles)", **c})
    stats.finalize(out["rul"])
    errs = raw["imputation_errors"]
    eu = np.asarray(raw["eval_units"])
    pairs = [(f"q_{x}", f"c_{x}") for x in ("lstm", "transformer", "cnn_lstm")] + \
            [(m, "linear") for m in ("q_lstm", "q_transformer", "q_cnn_lstm", "c_lstm", "c_transformer", "c_cnn_lstm")]
    for pattern in cfg.patterns:
        for rate in cfg.missing_rates:
            fam = []
            for a, b in pairs:
                ka, kb = f"{pattern}|{rate}|{a}", f"{pattern}|{rate}|{b}"
                if ka in errs and kb in errs:
                    A = [errs[ka][s] for s in S]
                    B = [errs[kb][s] for s in S]
                    c = stats.paired_rmse_difference([x["se"] for x in A], [x["n"] for x in A],
                                                     [x["se"] for x in B], [x["n"] for x in B], eu, cfg.n_boot)
                    fam.append({"pattern": pattern, "rate": rate, "a": a, "b": b, "metric": "imputation RMSE", **c})
            out["imputation"].extend(stats.finalize(fam))
    return out


def rank_imputers_table(evaluation: list[dict]) -> dict:
    """{pattern: {rate: [methods by RMSE]}} — used by the imputation-planning agent."""
    out: dict[str, dict[str, list]] = {}
    for r in evaluation:
        out.setdefault(r["pattern"], {}).setdefault(str(r["rate"]), []).append((r["rmse"], r["method"]))
    return {p: {rate: [{"method": m, "rmse": e} for e, m in sorted(v)] for rate, v in rates.items()}
            for p, rates in out.items()}


def _twin_comparison(rul_summary: list[dict], imp_eval: list[dict]) -> list[dict]:
    rows = []
    by = {r["model"]: r for r in rul_summary}
    for arch in ("lstm", "transformer", "cnn_lstm"):
        q, c = by.get(f"q_{arch}"), by.get(f"c_{arch}")
        rec = {"arch": arch}
        if q and c:
            rec.update(rul_rmse_quantum=q["rmse"], rul_rmse_twin=c["rmse"])
        qi = [r["rmse"] for r in imp_eval if r["method"] == f"q_{arch}"]
        ci = [r["rmse"] for r in imp_eval if r["method"] == f"c_{arch}"]
        if qi and ci:
            rec.update(impute_rmse_quantum_mean=float(np.mean(qi)), impute_rmse_twin_mean=float(np.mean(ci)))
        if len(rec) > 1:
            rows.append(rec)
    return rows


# ---------------------------------------------------------------- loading saved models (used by agents and API)


def load_report(subset: str = "FD001") -> dict | None:
    return read_json(paths.area("comparison_reports") / f"{REPORT_NAME.format(subset=subset)}.json")


def load_seq_model(task: str, name: str, subset: str = "FD001") -> dict:
    """Returns the saved bundle with a ready ``model``. ``task`` is "rul" or "impute"."""
    bundle = load_model("quantum_ai", f"qseq_{task}_{name}_{subset}")
    bundle["model"] = model_from_state(bundle["config"], bundle["state"])
    return bundle


def rul_accuracy_table(rul_predictions: dict, horizon: int = 30) -> list[dict]:
    """Classification view of RUL predictions on the official test engines, per model (mean ± SD over seeds):
    "fails within ``horizon`` cycles" accuracy / precision / recall / F1 (predicted positive when RUL_pred ≤ horizon),
    ROC-AUC (score = −RUL_pred), and the share of engines predicted within ±10 and ±20 cycles of the (capped) RUL.
    Computed directly from the saved per-engine predictions of every seed."""
    from ..classical_models.metrics import classification_metrics

    out = []
    for model, by_seed in rul_predictions.items():
        vals = []
        for preds in by_seed.values():
            t = np.array([p.get("rul_true_uncapped", p["rul_true"]) for p in preds], float)
            y = np.array([p["rul_pred"] for p in preds], float)
            err = np.abs(y - np.array([p["rul_true"] for p in preds], float))
            c = classification_metrics((t <= horizon).astype(int), -y, threshold=-horizon)
            vals.append({"accuracy": c["accuracy"], "precision": c["precision"], "recall": c["recall"], "f1": c["f1"],
                         "roc_auc": c["roc_auc"], "false_alarm_rate": c["false_alarm_rate"],
                         "within_10": float((err <= 10).mean()), "within_20": float((err <= 20).mean()),
                         "positives": c["positives"], "n": c["n"]})
        row = {"model": model, "horizon": horizon, "n_seeds": len(vals), "n_engines": vals[0]["n"],
               "positives": vals[0]["positives"]}
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_alarm_rate", "within_10", "within_20"):
            a = np.array([v[k] for v in vals], float)
            row[k] = float(a.mean())
            row[f"{k}_std"] = float(a.std(ddof=1)) if len(a) > 1 else None
        out.append(row)
    return sorted(out, key=lambda r: -r["accuracy"])
