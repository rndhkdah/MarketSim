"""Cost-plus prices with a bounded step (R7). Never cap the level."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.config import Config
from marketsim.real.steady_state import RealBaseline


def unit_cost(
    a: np.ndarray,
    p: np.ndarray,
    w: float | np.ndarray,
    ell: np.ndarray,
    m: np.ndarray,
    p_imp: float,
    z_sup: np.ndarray,
) -> np.ndarray:
    """Nuclear unit cost ``nuc_j`` (price index per unit output).

    National: ``p`` is ``(S,)``. Regional own-price: ``p`` is ``(R, S)`` and
    ``w`` is ``(R, 1)`` or ``(R,)`` — intermediates still use the same A.
    """
    p_arr = np.asarray(p, dtype=float)
    if p_arr.ndim == 1:
        return a.T @ p_arr + w * ell * np.exp(-z_sup) + m * p_imp
    # (R, S): ``p`` is the delivered input price P_in[d, i] (T3.05).
    return p_arr @ a + np.asarray(w, dtype=float)[..., None] * ell * np.exp(-z_sup) + m * p_imp


def tightness(
    x: np.ndarray,
    k: np.ndarray,
    ustar: np.ndarray,
    inv: np.ndarray,
    se: np.ndarray,
    cover: np.ndarray,
    is_stock: np.ndarray,
    kappa_u: float,
    gamma: float,
    cover_floor: float,
) -> np.ndarray:
    """Utilisation and stock-cover tightness (dimensionless)."""
    cov = np.where(is_stock, inv / np.maximum(se, 1e-9), 1.0)
    stock_term = np.where(is_stock, (cover / np.maximum(cov, cover_floor)) ** gamma, 1.0)
    return np.exp(kappa_u * (x / k - ustar)) * stock_term


@dataclass
class PriceState:
    """Log-price filters and the import-price index."""

    p: np.ndarray
    pf: np.ndarray
    ps: np.ndarray
    p_imp: float


def step_prices(
    state: PriceState,
    *,
    nuc: np.ndarray,
    markup: np.ndarray,
    tight: np.ndarray,
    z_cost: np.ndarray,
    pi_e: float,
    pt: np.ndarray,
    ptlag: np.ndarray,
    fast_mean_m: float,
    step_max: float,
    g: float,
    dz_imp: float = 0.0,
) -> PriceState:
    """Two-speed drift-compensated filter; ``|Δ ln p| ≤ step_max``. ``p_imp`` drifts at ``G``."""
    drift = pi_e / 12.0
    lp_star = np.log(np.maximum(markup * nuc * tight, 1e-18)) + drift + z_cost
    a_f = 1.0 / (1.0 + fast_mean_m)
    a_s = 1.0 / (1.0 + 3.0 * ptlag)
    pf = state.pf + drift + a_f * (lp_star - state.pf - drift)
    ps = state.ps + drift + a_s * (lp_star - state.ps - drift)
    lp_new = pt * pf + (1.0 - pt) * ps
    dlp = np.clip(lp_new - np.log(np.maximum(state.p, 1e-18)), -step_max, step_max)
    p = state.p * np.exp(dlp)
    p_imp = float(state.p_imp * g * np.exp(dz_imp))
    return PriceState(p=p, pf=pf, ps=ps, p_imp=p_imp)


def sector_pass_through(cfg: Config, codes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    pt = np.array([cfg.sectors.params(c).pass_through for c in codes], dtype=float)
    lag = np.array([cfg.sectors.params(c).pass_lag_q for c in codes], dtype=float)
    return pt, lag


def baseline_price_state(real: RealBaseline) -> PriceState:
    s = real.S
    return PriceState(p=np.ones(s), pf=np.zeros(s), ps=np.zeros(s), p_imp=1.0)
