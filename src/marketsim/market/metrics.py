"""Tier-1 / Tier-2 market statistics (T6.21 / §6.1 gates 2–4)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import ImpactCfg

DAYS_PER_YEAR = 252


def acf(x: np.ndarray, lag: int) -> float:
    """Lag-``k`` autocorrelation (dimensionless)."""
    z = np.asarray(x, dtype=float).reshape(-1)
    k = int(lag)
    if k < 1 or k >= z.size:
        raise ValueError("lag must be in 1 .. n-1")
    z = z - z.mean()
    den = float(np.dot(z, z))
    if den <= 0.0:
        return 0.0
    return float(np.dot(z[:-k], z[k:]) / den)


def pearson_kurtosis(x: np.ndarray) -> float:
    """Pearson kurtosis (Gaussian = 3). Dimensionless."""
    z = np.asarray(x, dtype=float).reshape(-1)
    z = z - z.mean()
    m2 = float(np.mean(z**2))
    if m2 <= 0.0:
        return 0.0
    return float(np.mean(z**4) / m2**2)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation (dimensionless)."""
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        raise ValueError("corr needs equal lengths >= 2")
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0.0:
        return 0.0
    return float(np.dot(a, b) / den)


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of ``log y`` on ``log x`` (dimensionless)."""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    lx = lx - lx.mean()
    ly = ly - ly.mean()
    den = float(np.dot(lx, lx))
    if den <= 0.0:
        return 0.0
    return float(np.dot(lx, ly) / den)


def meta_order_path(
    q_over_adv: float,
    duration_d: int,
    *,
    sigma: float = 0.01,
    cfg: ImpactCfg | None = None,
) -> np.ndarray:
    """Daily ``ξ`` while executing ``q_over_adv · ADV`` over ``duration_d`` days.

    Then 20 days of no flow (reversion window). Returns ``ξ`` of length
    ``duration_d + 20``.
    """
    kn = ImpactKernel(cfg)
    days = int(duration_d)
    if days < 1:
        raise ValueError("duration_d must be >= 1")
    q_day = float(q_over_adv) / float(days)
    adv = 1.0
    path = np.empty(days + 20, dtype=float)
    for t in range(days):
        path[t] = float(kn.step(q_day, adv, sigma))
    for t in range(days, days + 20):
        path[t] = float(kn.step(0.0, adv, sigma))
    return path


def peak_impact(path: np.ndarray, duration_d: int) -> float:
    """Max ``ξ`` during the execution window (dimensionless)."""
    return float(np.max(np.abs(path[: int(duration_d)])))


def revert_fraction(path: np.ndarray, duration_d: int) -> float:
    """Share of peak impact gone 20 days after completion (dimensionless)."""
    peak = peak_impact(path, duration_d)
    if peak <= 0.0:
        return 0.0
    return float((peak - abs(path[-1])) / peak)


def round_trip_pnl(path: np.ndarray, duration_d: int) -> float:
    """Buy over the window, sell at the last print. ``≤ 0`` is no free lunch.

    Average fill is the mean ``ξ`` during execution; exit is ``ξ`` after
    20 days. Units: log-price (dimensionless).
    """
    d = int(duration_d)
    fill = float(np.mean(path[:d]))
    exit_px = float(path[-1])
    return exit_px - fill


def walk_forward(
    n: int,
    step: Callable[[int], float],
) -> np.ndarray:
    """Collect ``n`` daily simple returns from ``step(t)`` (decimal / day)."""
    out = np.empty(int(n), dtype=float)
    for t in range(int(n)):
        out[t] = float(step(t))
    return out
