"""Paired statistics for model comparisons.

All comparisons are *paired*: both models are scored on the same engines (RUL) or the same masked windows
(imputation) under the same seeds. Errors are kept per evaluation unit as (sum of squared errors, count) so
RMSE can be recomputed for any resample.

* ``paired_rmse_difference`` – RMSE(A) − RMSE(B) averaged over seeds, with a 95 % percentile bootstrap CI that
  resamples *clusters* (engines). Windows of one engine overlap, so resampling windows would understate the
  uncertainty; resampling engines does not. The same resample is used for every seed (paired design).
* Wilcoxon signed-rank test on the per-cluster mean squared error difference (averaged over seeds).
* Seed-level paired t interval on the per-seed RMSE differences (only 3 seeds → wide; reported for completeness).
* Holm–Bonferroni adjustment across a family of comparisons.

A difference is called *significant* only when the Holm-adjusted Wilcoxon p < 0.05, the engine-bootstrap CI
excludes 0 **and** the seed-level paired-t CI excludes 0. The engine bootstrap treats each trained model as fixed
(it captures test-set sampling, not training randomness); the seed-level interval captures training randomness but
with 3 seeds it is wide. Requiring both is deliberately conservative. Nothing else is claimed.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def _cluster_sums(se: np.ndarray, n: np.ndarray, clusters: np.ndarray):
    """Collapse per-unit (se, n) of shape (seeds, units) to per-cluster sums (seeds, C)."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    S = np.zeros((se.shape[0], len(uniq)))
    N = np.zeros((se.shape[0], len(uniq)))
    for s in range(se.shape[0]):
        np.add.at(S[s], inv, se[s])
        np.add.at(N[s], inv, n[s])
    return S, N, uniq


def paired_rmse_difference(se_a, n_a, se_b, n_b, clusters, n_boot: int = 5000, seed: int = 0) -> dict:
    """``se_*``/``n_*``: arrays (seeds, units) of squared-error sums and counts; ``clusters``: (units,) ids."""
    se_a, n_a, se_b, n_b = (np.atleast_2d(np.asarray(x, float)) for x in (se_a, n_a, se_b, n_b))
    Sa, Na, uniq = _cluster_sums(se_a, n_a, clusters)
    Sb, Nb, _ = _cluster_sums(se_b, n_b, clusters)

    def diff(idx):
        ra = np.sqrt(Sa[:, idx].sum(1) / np.maximum(Na[:, idx].sum(1), 1))
        rb = np.sqrt(Sb[:, idx].sum(1) / np.maximum(Nb[:, idx].sum(1), 1))
        return ra - rb                                   # per seed

    full = np.arange(len(uniq))
    per_seed = diff(full)
    rng = np.random.default_rng(seed)
    boots = np.array([diff(rng.integers(0, len(uniq), len(uniq))).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Wilcoxon on per-cluster MSE differences averaged over seeds
    mse_a = (Sa / np.maximum(Na, 1)).mean(0)
    mse_b = (Sb / np.maximum(Nb, 1)).mean(0)
    d = mse_a - mse_b
    try:
        p_w = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p_w = 1.0
    k = len(per_seed)
    if k > 1:
        sd = per_seed.std(ddof=1)
        half = sps.t.ppf(0.975, k - 1) * sd / np.sqrt(k)
        seed_ci = [float(per_seed.mean() - half), float(per_seed.mean() + half)]
    else:
        seed_ci = [None, None]
    return {"rmse_diff": float(per_seed.mean()), "rmse_diff_per_seed": per_seed.tolist(),
            "bootstrap_ci95": [float(lo), float(hi)], "seed_t_ci95": seed_ci, "wilcoxon_p": p_w,
            "clusters": int(len(uniq)), "seeds": int(k), "a_better_clusters": int((d < 0).sum()),
            "b_better_clusters": int((d > 0).sum())}


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj.tolist()


def finalize(family: list[dict], alpha: float = 0.05) -> list[dict]:
    """Add Holm-adjusted p-values and a conservative verdict to a family of comparisons (in place)."""
    adj = holm([c["wilcoxon_p"] for c in family]) if family else []
    for c, p in zip(family, adj):
        c["wilcoxon_p_holm"] = p
        lo, hi = c["bootstrap_ci95"]
        slo, shi = c["seed_t_ci95"]
        excl = lo > 0 or hi < 0
        seed_excl = slo is not None and (slo > 0 or shi < 0)
        c["engine_ci_excludes_0"] = bool(excl)
        c["seed_ci_excludes_0"] = bool(seed_excl)
        c["significant"] = bool(p < alpha and excl and seed_excl)
        c["verdict"] = ("A lower error (significant)" if c["significant"] and c["rmse_diff"] < 0 else
                        "B lower error (significant)" if c["significant"] else
                        "consistent on these engines, not across seeds" if p < alpha and excl else
                        "no significant difference")
    return family
