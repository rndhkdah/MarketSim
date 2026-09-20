"""Aggregate household consumption and drift-compensated expected income (R2, R9)."""

from __future__ import annotations

import numpy as np

from marketsim.core.config import Config
from marketsim.demand.system import ScalarEtaDemand
from marketsim.ledger.opening import opening_wealth
from marketsim.real.steady_state import FinancialBaseline, RealBaseline


def consumption_nominal(alpha1: float, yd_e: float, alpha2: float, wealth: float, z_dem: float = 0.0) -> float:
    """``C_nom = (α1·YD_e + α2·W)·exp(z_dem)``. Units: cr/month nominal."""
    return float((alpha1 * yd_e + alpha2 * wealth) * np.exp(z_dem))


def smooth_nominal(current: float, observed: float, tau: float, g: float) -> float:
    """Drift-compensated smoother: ``s ← s·g + (x − s·g)/τ``."""
    return current * g + (observed - current * g) / tau


def household_basket(real: RealBaseline, cfg: Config, fin: FinancialBaseline) -> ScalarEtaDemand:
    """Baseline θ is the HOUSEHOLD final-demand mix (sums to 1)."""
    c0 = real.flat(real.C0)
    theta = c0 / c0.sum()
    eta = np.array([cfg.sectors.params(c).eta for c in real.codes], dtype=float)
    eps = np.array([cfg.sectors.params(c).eps_own for c in real.codes], dtype=float)
    semi = np.array([cfg.sectors.params(c).dem_rate_semi for c in real.codes], dtype=float)
    assert cfg.dynamics is not None
    return ScalarEtaDemand(
        theta=theta,
        eta=eta,
        eps=eps,
        dem_rate_semi=semi,
        zeta=cfg.dynamics.households.rate_budget_passthrough,
        vat=fin.vat,
    )


def income_index(yd_e: float, pc: float, yd0: float) -> float:
    """``y = (YD_e / P_c) / YD0`` (real expected income / baseline)."""
    return (yd_e / pc) / yd0


def wealth_from_ledger(ledger, hh: str = "HH:0") -> float:
    """Consumption-relevant W from the opening / running ledger."""
    return opening_wealth(ledger, hh)


def split_regional(national: float, weights: np.ndarray) -> np.ndarray:
    """Split a national scalar across regions. ``weights`` sum to 1; units follow ``national``."""
    w = np.asarray(weights, dtype=float)
    return float(national) * w


def regional_incomes(yd: np.ndarray, wealth: np.ndarray) -> tuple[float, float]:
    """National totals of per-region ``YD`` and ``W`` (cr)."""
    return float(np.asarray(yd, dtype=float).sum()), float(np.asarray(wealth, dtype=float).sum())
