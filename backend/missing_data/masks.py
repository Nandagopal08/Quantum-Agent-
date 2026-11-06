"""Missingness patterns.

* ``mcar``  – missing completely at random: every value is dropped independently with probability ``rate``.
* ``block`` – sensor outages: each channel loses contiguous gaps (length 2 … T/3) until ≈ ``rate`` of its
  steps are missing. This is the harder, more realistic pattern (a sensor drops out for a while).
"""

from __future__ import annotations

import numpy as np

PATTERNS = ("mcar", "block")


def make_mask(shape: tuple[int, int, int], rate: float | np.ndarray, pattern: str = "mcar",
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Float mask (1 = observed). ``rate`` may be a scalar or one rate per window (shape ``(N,)``)."""
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}")
    rng = rng or np.random.default_rng()
    N, T, C = shape
    rates = np.broadcast_to(np.asarray(rate, float), (N,))
    if np.any((rates < 0) | (rates >= 1)):
        raise ValueError("missing rate must be in [0, 1)")
    if pattern == "mcar":
        return (rng.random(shape) >= rates[:, None, None]).astype(float)
    mask = np.ones(shape)
    max_len = max(2, T // 3)
    for n in range(N):
        target = int(round(rates[n] * T))
        if target == 0:
            continue
        for c in range(C):
            missing = 0
            for _ in range(4 * T):                       # bounded: gaps may overlap
                if missing >= target:
                    break
                length = int(rng.integers(2, max_len + 1))
                length = min(length, target - missing) if target - missing >= 2 else 1
                start = int(rng.integers(0, T - length + 1))
                seg = mask[n, start:start + length, c]
                missing += int(seg.sum())
                seg[:] = 0
    return mask


def missingness_profile(mask: np.ndarray, channel_names: list[str] | None = None) -> dict:
    """Describe a ``(T, C)`` or ``(N, T, C)`` mask: overall/per-channel rate, gap lengths, likely pattern."""
    m = mask if mask.ndim == 3 else mask[None]
    N, T, C = m.shape
    miss = 1 - m
    gaps = []
    for n in range(N):
        for c in range(C):
            run = 0
            for v in miss[n, :, c]:
                if v:
                    run += 1
                elif run:
                    gaps.append(run)
                    run = 0
            if run:
                gaps.append(run)
    per_channel = miss.mean(axis=(0, 1))
    names = channel_names or [f"c{i}" for i in range(C)]
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    rate = float(miss.mean())
    # Under MCAR with rate r the expected run length is 1/(1-r); much longer runs indicate outages.
    expected_mcar_gap = 1.0 / max(1e-6, 1.0 - rate)
    pattern = "none" if rate == 0 else ("block" if mean_gap > 1.6 * expected_mcar_gap else "mcar")
    return {
        "missing_rate": rate,
        "per_channel": {names[i]: float(per_channel[i]) for i in range(C)},
        "fully_missing_channels": [names[i] for i in range(C) if per_channel[i] >= 0.999],
        "n_gaps": len(gaps), "mean_gap_length": mean_gap, "max_gap_length": int(max(gaps) if gaps else 0),
        "expected_mcar_gap_length": expected_mcar_gap, "likely_pattern": pattern,
        "steps": T, "channels": C, "windows": N,
    }
