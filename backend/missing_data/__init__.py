"""Missing-data simulation and imputation for multivariate time series.

Arrays are shaped ``(N, T, C)`` (windows × time steps × channels). Masks use 1 = observed, 0 = missing.
"""

from .masks import PATTERNS, make_mask, missingness_profile
from .imputers import BASELINE_IMPUTERS, impute_baseline, linear_fill, locf_fill, mean_fill

__all__ = ["PATTERNS", "make_mask", "missingness_profile", "BASELINE_IMPUTERS", "impute_baseline",
           "linear_fill", "locf_fill", "mean_fill"]
