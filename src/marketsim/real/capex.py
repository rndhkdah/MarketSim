"""Capex start rate, spending / completion chains and the supply line (R3, §2.6)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from marketsim.core.config import Config
from marketsim.core.erlang import ErlangChain
from marketsim.real.steady_state import RealBaseline


class QProvider(Protocol):
    """Tobin's Q by sector. Phase 2 stub returns 1; Phase 6 replaces the body."""

    def q_tobin(self) -> np.ndarray: ...


class UnitQ:
    """Q ≡ 1 at every sector (Phase 2 default)."""

    def __init__(self, n: int) -> None:
        self._q = np.ones(n)

    def q_tobin(self) -> np.ndarray:
        return self._q


def supply_line(
    pipe: np.ndarray,
    pipe0: np.ndarray,
    k: np.ndarray,
    k0: np.ndarray,
    ustar: np.ndarray,
    sl_weight: float,
) -> np.ndarray:
    """Capacity already under construction, in utilisation-equivalent units."""
    return sl_weight * ustar * (pipe - pipe0 * k / k0) / k


def start_rate(
    *,
    delta: float,
    unit_scale: float,
    phi: float,
    psi: float,
    chi: float,
    q_tobin_coef: float,
    q_scale: float,
    q_clip: float,
    g_e: np.ndarray,
    anchor_growth: float,
    u: np.ndarray,
    ustar: np.ndarray,
    sl: np.ndarray,
    cc_gap: np.ndarray,
    ln_q: np.ndarray,
    cap_mult: float,
) -> np.ndarray:
    """Annual start rate as a fraction of K. ``cc_gap`` is in 100bp units."""
    g_exp = (1.0 - anchor_growth) * g_e
    q_term = q_scale * q_tobin_coef * 100.0 * np.clip(ln_q, -q_clip, q_clip)
    raw = delta + unit_scale * (
        phi * 100.0 * g_exp + psi * 100.0 * ((u - ustar) - sl) - chi * cc_gap + q_term
    )
    return np.clip(raw, 0.0, cap_mult * delta)


def cost_of_capital_gap(
    r: float,
    pi_e: float,
    r_n: float,
    ds: float,
    derp: float,
    nd: np.ndarray,
    pricing: str,
) -> np.ndarray:
    """``cc_gap`` in 100bp. Uniform: one gap; risk_based: spread term × nd/2.5."""
    policy = (r - pi_e - r_n) * 100.0 + derp * 100.0
    spread_term = ds * 100.0
    if pricing == "uniform":
        return np.full(nd.shape, policy + spread_term)
    return policy + (nd / 2.5) * spread_term


def seed_pipelines(
    real: RealBaseline,
    cfg: Config,
) -> tuple[ErlangChain, ErlangChain, np.ndarray]:
    """Seed spend (Erlang 3, mean min(3·lag_q, build_m)) and completion chains."""
    assert cfg.dynamics is not None
    k0 = real.flat(real.K0)
    v = real.flat(real.v)
    delta_m = cfg.dynamics.capex.delta_annual / 12.0
    starts0 = delta_m * k0
    build_m = 3.0 * np.array([cfg.sectors.params(c).build_lag_q for c in real.codes])
    spend_mean = np.minimum(3.0 * cfg.edges.capex.lag_q, build_m)
    pipe = ErlangChain(3, build_m, k0.shape)
    pipe.seed(starts0)
    spend = ErlangChain(3, spend_mean, k0.shape)
    spend.seed(v * starts0)
    return pipe, spend, pipe.content().copy()


def step_capacity(
    k: np.ndarray,
    starts: np.ndarray,
    pipe: ErlangChain,
    delta_m: float,
) -> np.ndarray:
    """``K ← (1 − δ/12)·K + completions``."""
    return (1.0 - delta_m) * k + pipe.push(starts)
