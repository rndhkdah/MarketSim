"""Income-decile tiers (§3.3). Composition only — the macro C function is unchanged."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

SIGMA = 0.75  # lognormal σ; Gini ≈ 0.40
N_DECILES = 10
BUDGET_EXP = 0.9  # b_k ∝ ι_k^0.9 (the rich save more)


@dataclass(frozen=True)
class Tiers:
    """Ten regional income deciles. ``iota`` is dimensionless with mean 1."""

    iota: np.ndarray  # (10,) relative income
    budget_share: np.ndarray  # (10,) sums to 1
    sigma: float = SIGMA

    @property
    def n_tiers(self) -> int:
        return int(self.iota.size)


def relative_incomes(sigma: float = SIGMA, n: int = N_DECILES) -> np.ndarray:
    """``ι_k = n·[Φ(z_k − σ) − Φ(z_{k−1} − σ)]``, ``z_k = Φ⁻¹(k/n)``. Mean 1, sum ``n``."""
    grid = np.linspace(0.0, 1.0, n + 1)
    # Φ^{-1}(0)=−∞, Φ^{-1}(1)=+∞ — clip the ends for a stable cdf difference
    z = norm.ppf(grid)
    z[0], z[-1] = -12.0, 12.0
    cdf = norm.cdf(z - sigma)
    iota = n * np.diff(cdf)
    return iota / iota.mean()


def budget_shares(iota: np.ndarray, exp: float = BUDGET_EXP) -> np.ndarray:
    """``b_k ∝ ι_k^{exp}``. Dimensionless, sums to 1."""
    raw = np.power(np.asarray(iota, dtype=float), exp)
    return raw / raw.sum()


def gini(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Gini of a discrete distribution. ``values`` are incomes; equal weights default."""
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    w = w / w.sum()
    cum_pop = np.cumsum(w)
    cum_inc = np.cumsum(w * x)
    cum_inc = cum_inc / cum_inc[-1]
    # discrete Lorenz: G = 1 − Σ (F_k + F_{k-1})(S_k − S_{k-1}) wait trapezoid on (F, S)
    f_prev = np.concatenate([[0.0], cum_pop[:-1]])
    s_prev = np.concatenate([[0.0], cum_inc[:-1]])
    area = np.sum((cum_pop - f_prev) * (cum_inc + s_prev) / 2.0)
    return float(1.0 - 2.0 * area)


def build_tiers(sigma: float = SIGMA) -> Tiers:
    """Ten-decile pack. ``iota`` sums to 10 (mean 1)."""
    iota = relative_incomes(sigma)
    return Tiers(iota=iota, budget_share=budget_shares(iota), sigma=sigma)


def tier_real_income(iota: np.ndarray, y_r: float) -> np.ndarray:
    """``y_k = ι_k · y_r`` (real disposable income index; 1 at baseline)."""
    return np.asarray(iota, dtype=float) * float(y_r)
