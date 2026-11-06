"""Classical imputation baselines (vectorised over windows and channels).

* ``mean``   – per-channel training mean.
* ``locf``   – last observation carried forward (leading gaps: next observation carried backward).
* ``linear`` – linear interpolation in time between observed neighbours (edges: nearest observation).

Channels with no observation in a window fall back to the channel mean for every method.
"""

from __future__ import annotations

import numpy as np

BASELINE_IMPUTERS = ("mean", "locf", "linear")


def _means(x: np.ndarray, mask: np.ndarray, means: np.ndarray | None) -> np.ndarray:
    if means is not None:
        return np.asarray(means, float)
    obs = mask.sum(axis=(0, 1))
    return np.where(obs > 0, (x * mask).sum(axis=(0, 1)) / np.maximum(obs, 1), 0.0)


def mean_fill(x: np.ndarray, mask: np.ndarray, means: np.ndarray | None = None) -> np.ndarray:
    mu = _means(x, mask, means)
    return np.where(mask > 0, x, mu[None, None, :])


def _fill_direction(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Forward fill along time; positions before the first observation stay NaN."""
    N, T, C = x.shape
    idx = np.where(mask > 0, np.arange(T)[None, :, None], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(x, idx, axis=1)
    seen = np.maximum.accumulate(mask > 0, axis=1)
    return np.where(seen, out, np.nan)


def locf_fill(x: np.ndarray, mask: np.ndarray, means: np.ndarray | None = None) -> np.ndarray:
    fwd = _fill_direction(x, mask)
    bwd = _fill_direction(x[:, ::-1], mask[:, ::-1])[:, ::-1]
    out = np.where(np.isnan(fwd), bwd, fwd)
    mu = _means(x, mask, means)
    return np.where(np.isnan(out), mu[None, None, :], out)


def linear_fill(x: np.ndarray, mask: np.ndarray, means: np.ndarray | None = None) -> np.ndarray:
    N, T, C = x.shape
    t = np.arange(T, dtype=float)
    obs_t = np.where(mask > 0, t[None, :, None], np.nan)
    # previous / next observed time and value for every position
    prev_t = np.fmax.accumulate(np.nan_to_num(obs_t, nan=-np.inf), axis=1)
    next_t = -np.fmax.accumulate(np.nan_to_num(-obs_t[:, ::-1], nan=-np.inf), axis=1)[:, ::-1]
    fwd = _fill_direction(x, mask)
    bwd = _fill_direction(x[:, ::-1], mask[:, ::-1])[:, ::-1]
    has_prev, has_next = np.isfinite(prev_t), np.isfinite(next_t)
    span = np.where(has_prev & has_next, next_t - prev_t, 1.0)
    w = np.where(has_prev & has_next & (span > 0), (t[None, :, None] - prev_t) / np.where(span > 0, span, 1), 0.0)
    both = np.where(has_prev & has_next, (1 - w) * np.nan_to_num(fwd) + w * np.nan_to_num(bwd), np.nan)
    out = np.where(has_prev & has_next, both, np.where(has_prev, fwd, bwd))
    mu = _means(x, mask, means)
    out = np.where(np.isnan(out), mu[None, None, :], out)
    return np.where(mask > 0, x, out)


_FUNCS = {"mean": mean_fill, "locf": locf_fill, "linear": linear_fill}


def impute_baseline(method: str, x: np.ndarray, mask: np.ndarray, means: np.ndarray | None = None) -> np.ndarray:
    if method not in _FUNCS:
        raise ValueError(f"baseline imputer must be one of {BASELINE_IMPUTERS}")
    x = np.where(mask > 0, x, 0.0)          # never read the hidden values
    return _FUNCS[method](x, mask, means)
