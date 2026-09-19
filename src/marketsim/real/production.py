"""Production plans and capacity caps (R1, R4)."""

from __future__ import annotations

import numpy as np

from marketsim.core.config import Config


def expected_sales(se: np.ndarray, sales: np.ndarray, tau_e: float) -> np.ndarray:
    """Adaptive sales expectation: ``s_e += (sales − s_e) / τ_e``. Units: cr/month."""
    return np.asarray(se, dtype=float) + (np.asarray(sales, dtype=float) - se) / tau_e


def plan_output(
    se: np.ndarray,
    inv: np.ndarray,
    cover: np.ndarray,
    leak: np.ndarray,
    tau_inv: np.ndarray,
    backlog: np.ndarray,
    tau_b: float,
    is_stock: np.ndarray,
    k_eff: np.ndarray,
) -> np.ndarray:
    """Desired output (cr/month of capacity), clipped to ``[0, K_eff]``."""
    stock_plan = se + leak * inv + (cover * se - inv) / tau_inv + backlog / tau_b
    plan = np.where(is_stock, stock_plan, se)
    return np.clip(plan, 0.0, k_eff)


def labour_cap(
    n: np.ndarray,
    fc: np.ndarray,
    ell: np.ndarray,
    k: np.ndarray,
    ustar: np.ndarray,
    overtime: float,
    z_sup: np.ndarray,
) -> np.ndarray:
    """Labour-constrained output (cr/month). ``n_var`` can be negative → cap at 0."""
    n_var = n - fc * ell * k * ustar
    denom = (1.0 - fc) * ell * np.exp(-z_sup)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = overtime * n_var / np.where(np.abs(denom) < 1e-15, np.inf, denom)
    return np.maximum(cap, 0.0)


def input_cap(
    s_in: np.ndarray,
    a: np.ndarray,
    stor_in: np.ndarray,
    crit: np.ndarray,
) -> np.ndarray:
    """Storable-critical input cap: ``min_i S_in[i,j] / a_ij`` (cr/month)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(stor_in & crit, s_in / np.where(a > 0, a, 1.0), np.inf)
    return ratio.min(axis=0)


def flow_crit_fill(fill_prev: np.ndarray, stor_in: np.ndarray, crit: np.ndarray) -> np.ndarray:
    """Last-period fill of flow-critical inputs (dimensionless)."""
    return np.where((~stor_in) & crit, fill_prev[:, None], 1.0).min(axis=0)


def goods_output(
    plan: np.ndarray,
    k_eff: np.ndarray,
    lab: np.ndarray,
    in_cap: np.ndarray,
    flow_crit: np.ndarray,
) -> np.ndarray:
    """``x_goods = min(plan, K_eff, lab_cap, in_cap) · flow_crit``."""
    return np.minimum.reduce([plan, k_eff, lab, in_cap]) * flow_crit


def critical_mask(a: np.ndarray, mu: np.ndarray, cfg: Config) -> np.ndarray:
    """``crit[i,j]`` if supplier ``i`` is listed and its share of buyer ``j`` intermediates ≥ min_share."""
    assert cfg.dynamics is not None
    spec = cfg.dynamics.production.critical_input
    share = np.divide(a, mu[None, :], out=np.zeros_like(a), where=mu[None, :] > 1e-12)
    codes = cfg.codes
    crit_sup = np.array([c in set(spec.suppliers) for c in codes])
    return (share >= spec.min_share) & crit_sup[:, None]
