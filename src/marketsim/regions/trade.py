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


def flows_from_shares(t_shares: np.ndarray, dest: np.ndarray) -> np.ndarray:
    """``F[i,src,dst] = T[i,src,dst] · dest[dst,i]`` (cr/month)."""
    dest_a = np.asarray(dest, dtype=float)
    return np.asarray(t_shares, dtype=float) * dest_a.T[:, None, :]


def link_capacity(capacity_mult: np.ndarray, baseline_flow: np.ndarray) -> np.ndarray:
    """Directed per-sector caps ``(S, R, R)``. Home flows are uncapped (``+inf``)."""
    cap = np.asarray(capacity_mult, dtype=float)[None, :, :] * np.asarray(baseline_flow, dtype=float)
    n_r = cap.shape[-1]
    cap[:, np.arange(n_r), np.arange(n_r)] = np.inf
    return cap


def route_with_spill(
    t_shares: np.ndarray,
    dest: np.ndarray,
    link_cap: np.ndarray,
    *,
    n_iter: int = 4,
) -> np.ndarray:
    """Route dest demand by ``T``, clip to link caps, spill to other sources.

    Returns attempted source→dest flows ``(S, R_src, R_dst)`` cr/month, before
    supplier rationing. Home (src = dst) is never clipped by a link.
    """
    t_arr = np.asarray(t_shares, dtype=float)
    remaining_dest = np.asarray(dest, dtype=float).copy()
    remaining_cap = np.asarray(link_cap, dtype=float).copy()
    n_s, n_r, _ = t_arr.shape
    remaining_cap[:, np.arange(n_r), np.arange(n_r)] = np.inf
    delivered = np.zeros_like(t_arr)
    for _ in range(n_iter):
        open_link = remaining_cap > 1e-15
        weight = np.where(open_link, t_arr, 0.0)
        wsum = weight.sum(axis=1, keepdims=True)
        share = np.divide(weight, wsum, out=np.zeros_like(weight), where=wsum > 0)
        attempt = share * remaining_dest.T[:, None, :]
        take = np.minimum(attempt, remaining_cap)
        delivered = delivered + take
        remaining_cap = np.maximum(remaining_cap - take, 0.0)
        remaining_dest = np.maximum(remaining_dest - take.sum(axis=1).T, 0.0)
    return delivered


def delivered_prices(t_shares: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Buyer-region delivered price ``P_in[dst, i] = Σ_src T[i,src,dst] · p[src,i]`` (index)."""
    return np.einsum("isd,si->di", t_shares, np.asarray(p, dtype=float))


def armington_target(
    t0: np.ndarray,
    p: np.ndarray,
    cost: np.ndarray,
    sigma: np.ndarray,
    avail: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """``T* ∝ T0 · (p_src·(1+cost)/P̄)^{−σ} · avail^κ``, renormalised over sources.

    ``p`` and ``avail`` are ``(R, S)``; ``cost`` is ``(R, R)``; ``sigma`` is ``(S,)``.
    Transport cost is a preference friction only (no ledger wedge).
    """
    p_a = np.asarray(p, dtype=float)
    # p_src[i, src, dst] = p[src, i] * (1 + cost[src, dst])
    landed = p_a.T[:, :, None] * (1.0 + np.asarray(cost, dtype=float)[None, :, :])
    t_now = np.asarray(t0, dtype=float)
    pbar = np.einsum("isd,isd->id", t_now, landed)
    pbar = np.maximum(pbar, 1e-12)
    rel = landed / pbar[:, None, :]
    sig = np.asarray(sigma, dtype=float)[:, None, None]
    avail_term = np.maximum(np.asarray(avail, dtype=float).T[:, :, None], 1e-12) ** float(kappa)
    raw = t_now * np.power(rel, -sig) * avail_term
    denom = raw.sum(axis=1, keepdims=True)
    return np.divide(raw, denom, out=np.zeros_like(raw), where=denom > 0)


def smooth_shares(t_shares: np.ndarray, target: np.ndarray, tau: float) -> np.ndarray:
    """``T ← T + (T* − T)/τ`` then project onto the simplex (Σ_src = 1, T ≥ 0)."""
    tau = max(float(tau), 1.0)
    t_new = np.asarray(t_shares, dtype=float) + (np.asarray(target, dtype=float) - t_shares) / tau
    t_new = np.maximum(t_new, 0.0)
    denom = t_new.sum(axis=1, keepdims=True)
    return np.divide(t_new, denom, out=np.zeros_like(t_new), where=denom > 0)


def ration_sources(flows: np.ndarray, avail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pro-rata supplier rationing across buying regions.

    ``flows`` is ``(S, src, dst)``; ``avail`` is ``(R, S)``. Returns filled
    flows and source fill rates ``(R, S)``.
    """
    req = np.asarray(flows, dtype=float).sum(axis=2)  # (S, src)
    avail_si = np.asarray(avail, dtype=float).T
    fill = np.minimum(1.0, avail_si / np.maximum(req, 1e-12))
    filled = np.asarray(flows, dtype=float) * fill[:, :, None]
    return filled, fill.T
