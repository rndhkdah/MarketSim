"""T3.05 — Armington share dynamics and regional price convergence (gate 3)."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.prices import PriceState, sector_pass_through, step_prices, tightness, unit_cost
from marketsim.real.steady_state import compute_regional_real_baseline
from marketsim.regions.geometry import build_geometry
from marketsim.regions.trade import armington_target, delivered_prices, smooth_shares


def test_shares_stay_on_simplex(cfg, io) -> None:
    geom = build_geometry(cfg.regions, cfg.codes)
    reg = compute_regional_real_baseline(io, cfg, geom)
    p = np.ones((geom.n_regions, reg.S))
    p[0, 0] = 1.2
    avail = np.ones_like(p)
    t = reg.T0.copy()
    for _ in range(12):
        star = armington_target(reg.T0, p, geom.cost, geom.armington_sigma, avail, geom.trade.availability_kappa)
        t = smooth_shares(t, star, geom.trade.tau_share_m)
        assert np.all(t >= -1e-15)
        assert np.allclose(t.sum(axis=1), 1.0, atol=1e-12)


def test_share_path_monotone_after_price_step(cfg, io) -> None:
    geom = build_geometry(cfg.regions, cfg.codes)
    reg = compute_regional_real_baseline(io, cfg, geom)
    energy = cfg.codes.index("ENERGY")
    p = np.ones((geom.n_regions, reg.S))
    avail = np.ones_like(p)
    t = reg.T0.copy()
    p[0, energy] = 1.25
    path = []
    for _ in range(18):
        star = armington_target(reg.T0, p, geom.cost, geom.armington_sigma, avail, geom.trade.availability_kappa)
        t = smooth_shares(t, star, geom.trade.tau_share_m)
        path.append(float(t[energy, 0, 1]))
    diffs = np.diff(path)
    assert path[-1] < path[0]
    assert np.all(diffs <= 1e-12)


@pytest.mark.validation
def test_materials_shock_prices_converge(cfg, io) -> None:
    """Gate 3: −10 % MATERIALS productivity in RESOURCE; tradable prices stay in band."""
    geom = build_geometry(cfg.regions, cfg.codes)
    reg = compute_regional_real_baseline(io, cfg, geom)
    dyn = cfg.dynamics
    assert dyn is not None
    r_idx = {c: i for i, c in enumerate(geom.codes)}
    mat = cfg.codes.index("MATERIALS")
    src = r_idx["RESOURCE"]
    n_r, n_s = geom.n_regions, reg.S
    z = np.zeros((n_r, n_s))
    z[src, mat] = float(np.log(0.9))
    p_st = PriceState(p=np.ones((n_r, n_s)), pf=np.zeros((n_r, n_s)), ps=np.zeros((n_r, n_s)), p_imp=1.0)
    t = reg.T0.copy()
    avail = np.ones((n_r, n_s))
    pt, ptlag = sector_pass_through(cfg, reg.codes)
    markup = reg.flat(reg.markup)
    cover = reg.flat(reg.cover)
    m = reg.flat(reg.m)
    x0 = reg.x0
    k0 = reg.K0
    se = reg.s0.copy()
    inv = reg.inv0.copy()
    hist: list[np.ndarray] = []
    for _ in range(36):
        k_eff = k0 * np.exp(z)
        x = np.minimum(x0, k_eff)
        pin = delivered_prices(t, p_st.p)
        nuc = unit_cost(io.A, pin, geom.wage_level, reg.ell, m, p_st.p_imp, z)
        tight = tightness(
            x, k0, np.array([cfg.sectors.params(c).util_target for c in reg.codes]),
            inv, se, cover, reg.is_stock,
            dyn.prices.kappa_util, dyn.prices.gamma_cover, dyn.prices.cover_floor,
        )
        p_st = step_prices(
            p_st, nuc=nuc, markup=markup, tight=tight, z_cost=np.zeros((n_r, n_s)),
            pi_e=0.0, pt=pt, ptlag=ptlag, fast_mean_m=dyn.prices.fast_mean_m,
            step_max=dyn.prices.step_max_month, g=1.0,
        )
        avail = 0.5 * avail + 0.5 * np.exp(z)
        star = armington_target(
            reg.T0, p_st.p, geom.cost, geom.armington_sigma, avail, geom.trade.availability_kappa
        )
        t = smooth_shares(t, star, geom.trade.tau_share_m)
        hist.append(p_st.p.copy())
    p24 = hist[23]
    p_mat = p24[:, mat]
    assert np.all(p_mat > 1.0)
    assert p_mat[src] == np.max(p_mat)
    pbar = float(np.mean(np.log(p_mat)))
    band = float(geom.cost.max()) + 0.05
    skip = {cfg.codes.index("CONSTRUCT"), cfg.codes.index("REALESTATE")}
    for i, code in enumerate(reg.codes):
        if i in skip:
            continue
        if geom.tradability[i] < 0.5:
            continue
        logs = np.log(hist[23][:, i])
        assert float(np.max(np.abs(logs - logs.mean()))) <= band + 1e-9, (code, logs, band)
    assert pbar > 0.0
