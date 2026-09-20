"""Baseline inter-regional trade shares and the stacked Leontief map (§3.2)."""

from __future__ import annotations

import numpy as np

from marketsim.regions.geometry import Geometry


def baseline_trade_shares(geom: Geometry) -> np.ndarray:
    """``T0[i, src, dst]`` — share of dest-``dst`` demand for ``i`` sourced from ``src``.

    ``T0[i,s→d] = (1 − trad_i)·1[s=d] + trad_i·Gr[i,s→d]``,
    ``Gr ∝ capacity_share[s,i] · exp(−θ·cost_sd) · (home_bias if s=d)``,
    normalised over sources ``s``. Dimensionless; ``Σ_src T0 = 1``.
    """
    cap = np.asarray(geom.capacity_share, dtype=float)  # (R, S)
    cost = np.asarray(geom.cost, dtype=float)  # (R, R)
    trad = np.asarray(geom.tradability, dtype=float)  # (S,)
    theta = float(geom.trade.gravity_theta)
    home = float(geom.trade.home_bias)
    n_r, n_s = cap.shape

    # grav[src, dst, i]
    grav = cap[:, None, :] * np.exp(-theta * cost[:, :, None])
    home_w = 1.0 + (home - 1.0) * np.eye(n_r)[:, :, None]
    grav = grav * home_w
    denom = grav.sum(axis=0, keepdims=True)
    gr = np.divide(grav, denom, out=np.zeros_like(grav), where=denom > 0)
    eye = np.eye(n_r)[:, :, None]
    t_sdi = (1.0 - trad) * eye + trad * gr
    return np.transpose(t_sdi, (2, 0, 1))


def dest_demand(a: np.ndarray, x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Intermediate + final demand at the buying region. ``(R, S)`` cr/month."""
    return x @ a.T + d


def source_orders(t_shares: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """Route dest demand to source regions: ``orders[src,i] = Σ_d T[i,src,d]·dest[d,i]``.

    ``t_shares`` is ``(S, R_src, R_dst)``; ``dest`` / return are ``(R, S)`` cr/month.
    """
    # dest[d, i] → (src, i)
    return np.einsum("isd,di->si", t_shares, dest)


def stacked_leontief(
    a: np.ndarray,
    t_shares: np.ndarray,
    lam: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """``x = (I − Λ 𝒯 𝒜)⁻¹ Λ 𝒯 d``. ``d`` and return are ``(R, S)`` cr/month.

    ``𝒜`` applies ``A`` inside each buying region; ``𝒯`` routes by ``T``;
    ``Λ`` is ``diag(1 + leak·cover)`` repeated across regions.
    """
    n_r, n_s = d.shape
    n = n_r * n_s
    a_big = np.zeros((n, n))
    t_big = np.zeros((n, n))
    idx = np.arange(n_s)
    for r in range(n_r):
        a_big[r * n_s : (r + 1) * n_s, r * n_s : (r + 1) * n_s] = a
    for src in range(n_r):
        for dst in range(n_r):
            t_big[src * n_s + idx, dst * n_s + idx] = t_shares[:, src, dst]
    lam_big = np.diag(np.tile(np.asarray(lam, dtype=float), n_r))
    d_vec = np.asarray(d, dtype=float).reshape(n)
    rhs = lam_big @ t_big @ d_vec
    x_vec = np.linalg.solve(np.eye(n) - lam_big @ t_big @ a_big, rhs)
    return x_vec.reshape(n_r, n_s)


def net_exports(t_shares: np.ndarray, dest: np.ndarray, sourced: np.ndarray) -> np.ndarray:
    """``sourced − dest`` per cell (cr/month). Positive = net exporter."""
    return sourced - dest
