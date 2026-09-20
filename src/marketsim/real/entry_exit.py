"""NPC entry / exit on excess profit rates (§3.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.config import EntryExitCfg


def profit_rate(ebitda: np.ndarray, k: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    """Real EBITDA / K (1/month). Deflate by the sector price index when ``p`` is given."""
    eb = np.asarray(ebitda, dtype=float)
    if p is not None:
        eb = eb / np.maximum(np.asarray(p, dtype=float), 1e-12)
    return eb / np.maximum(np.asarray(k, dtype=float), 1e-12)


def smooth_excess(
    rate: np.ndarray,
    rate0: np.ndarray,
    smoothed: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``s ← s + (rate/rate0 − s)/τ``; excess = s − 1 (dimensionless)."""
    ratio = np.asarray(rate, dtype=float) / np.maximum(np.asarray(rate0, dtype=float), 1e-12)
    sm = np.asarray(smoothed, dtype=float) + (ratio - smoothed) / max(float(tau), 1.0)
    return sm - 1.0, sm


@dataclass(frozen=True)
class EntryExitStep:
    """Additional start rate and scrap rate (both fraction of K per year)."""

    excess: np.ndarray
    smoothed: np.ndarray
    entry_rate: np.ndarray
    exit_rate: np.ndarray
    scrap: np.ndarray


def step_entry_exit(
    ebitda: np.ndarray,
    k: np.ndarray,
    rate0: np.ndarray,
    smoothed: np.ndarray,
    cfg: EntryExitCfg,
    p: np.ndarray | None = None,
) -> EntryExitStep:
    """Entry ``κ_e·max(excess−θ_e, 0)``; exit scraps ``κ_x·max(−excess−θ_x, 0)`` of K / year."""
    rate = profit_rate(ebitda, k, p)
    excess, sm = smooth_excess(rate, rate0, smoothed, cfg.tau_excess_m)
    if not cfg.enabled:
        z = np.zeros_like(excess)
        return EntryExitStep(excess=excess, smoothed=sm, entry_rate=z, exit_rate=z, scrap=z)
    entry = cfg.kappa_entry * np.maximum(excess - cfg.theta_entry, 0.0)
    exit_r = cfg.kappa_exit * np.maximum(-excess - cfg.theta_exit, 0.0)
    scrap = exit_r / 12.0 * np.asarray(k, dtype=float)
    return EntryExitStep(excess=excess, smoothed=sm, entry_rate=entry, exit_rate=exit_r, scrap=scrap)
