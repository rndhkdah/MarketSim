from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.erlang import ErlangChain
from marketsim.real.capex import (
    cost_of_capital_gap,
    seed_pipelines,
    start_rate,
    step_capacity,
    supply_line,
)
from marketsim.real.steady_state import compute_real_baseline


def _rate_args(cfg, io, *, cc_gap, sl=None, u=None, ln_q=None):
    real = compute_real_baseline(io, cfg)
    s = real.S
    cap = cfg.edges.capex
    dyn = cfg.dynamics
    ustar = np.array([cfg.sectors.params(c).util_target for c in real.codes])
    return dict(
        delta=dyn.capex.delta_annual,
        unit_scale=cap.unit_scale,
        phi=cap.coefficients.phi_accelerator,
        psi=cap.coefficients.psi_utilisation,
        chi=cap.coefficients.chi_cost_of_capital,
        q_tobin_coef=cap.coefficients.q_tobin,
        q_scale=cap.q_scale,
        q_clip=dyn.capex.q_clip,
        g_e=np.zeros(s),
        anchor_growth=dyn.expectations.anchor_growth,
        u=ustar if u is None else u,
        ustar=ustar,
        sl=np.zeros(s) if sl is None else sl,
        cc_gap=cc_gap,
        ln_q=np.zeros(s) if ln_q is None else ln_q,
        cap_mult=dyn.capex.start_rate_cap_mult,
    ), real


def test_baseline_starts_keep_k(cfg, io) -> None:
    dyn = cfg.dynamics
    args, real = _rate_args(cfg, io, cc_gap=np.zeros(io.n))
    rate = start_rate(**args)
    assert np.allclose(rate, dyn.capex.delta_annual, atol=1e-12)
    k = real.flat(real.K0).copy()
    pipe, _spend, _p0 = seed_pipelines(real, cfg)
    starts = k * rate / 12.0
    assert np.allclose(starts, (dyn.capex.delta_annual / 12.0) * k, atol=1e-12)
    for _ in range(12):
        k = step_capacity(k, starts, pipe, dyn.capex.delta_annual / 12.0)
    assert np.allclose(k, real.flat(real.K0), atol=1e-9)


def test_common_rate_100bp_cuts_45bp(cfg, io) -> None:
    args, _real = _rate_args(cfg, io, cc_gap=np.ones(io.n))
    rate = start_rate(**args)
    expect = cfg.dynamics.capex.delta_annual - 0.0045
    assert np.allclose(rate, expect, atol=1e-12)
    assert float(rate.max() - rate.min()) < 1e-15


def test_risk_based_nd5_is_90bp() -> None:
    nd = np.array([5.0])
    gap = cost_of_capital_gap(0.01, 0.00, 0.01, ds=0.01, derp=0.0, nd=nd, pricing="risk_based")
    assert gap[0] == pytest.approx(2.0)
    rate = start_rate(
        delta=0.06,
        unit_scale=0.01,
        phi=1.2,
        psi=0.8,
        chi=0.45,
        q_tobin_coef=0.3,
        q_scale=0.1,
        q_clip=0.5,
        g_e=np.zeros(1),
        anchor_growth=0.75,
        u=np.array([0.8]),
        ustar=np.array([0.8]),
        sl=np.zeros(1),
        cc_gap=gap,
        ln_q=np.zeros(1),
        cap_mult=3.0,
    )
    assert rate[0] == pytest.approx(0.06 - 0.009, abs=1e-12)


def test_supply_line_lowers_rate(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    k = real.flat(real.K0)
    pipe0 = np.ones(real.S)
    sl = supply_line(2.0 * pipe0, pipe0, k, k, np.ones(real.S) * 0.8, 0.85)
    assert np.all(sl > 0)
    args, _ = _rate_args(cfg, io, cc_gap=np.zeros(io.n), sl=sl)
    rate = start_rate(**args)
    assert np.all(rate < cfg.dynamics.capex.delta_annual - 1e-12)


def test_completions_lag_energy_slower_than_construct(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    idx = {c: i for i, c in enumerate(real.codes)}
    build_m = 3.0 * np.array([cfg.sectors.params(c).build_lag_q for c in real.codes])
    pipe = ErlangChain(3, build_m, (real.S,))
    pulse = np.zeros(real.S)
    pulse[idx["ENERGY"]] = 1.0
    pulse[idx["CONSTRUCT"]] = 1.0
    cum = np.zeros(real.S)
    t_half = {}
    for t in range(1, 200):
        cum += pipe.push(pulse if t == 1 else np.zeros(real.S))
        for code in ("ENERGY", "CONSTRUCT"):
            if code not in t_half and cum[idx[code]] >= 0.5:
                t_half[code] = t
        if len(t_half) == 2:
            break
    assert t_half["CONSTRUCT"] < t_half["ENERGY"]
    # Mean lag is 3 · build_lag_q months (6 vs 36).
    assert t_half["CONSTRUCT"] < 12
    assert t_half["ENERGY"] > 20
