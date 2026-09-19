from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.production import (
    critical_mask,
    expected_sales,
    input_cap,
    labour_cap,
    plan_output,
)
from marketsim.real.steady_state import compute_real_baseline


def test_expected_sales_adapts() -> None:
    se = np.array([10.0, 20.0])
    sales = np.array([13.0, 20.0])
    out = expected_sales(se, sales, tau_e=3.0)
    assert out[0] == pytest.approx(11.0)
    assert out[1] == pytest.approx(20.0)


def test_baseline_plan_equals_x0(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    se = real.flat(real.s0)
    inv = real.flat(real.inv0)
    cover = real.flat(real.cover)
    leak = real.flat(real.leak)
    tau_inv = real.flat(real.tau_inv)
    backlog = real.flat(real.backlog0)
    k_eff = real.flat(real.K0)
    plan = plan_output(
        se,
        inv,
        cover,
        leak,
        tau_inv,
        backlog,
        cfg.dynamics.production.tau_backlog_m,
        real.is_stock,
        k_eff,
    )
    assert np.allclose(plan, real.flat(real.x0), atol=1e-9)


def test_stock_gap_raises_plan_by_gap_over_tau(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    se = real.flat(real.s0)
    inv = real.flat(real.inv0).copy()
    cover = real.flat(real.cover)
    tau_inv = real.flat(real.tau_inv)
    stock = np.where(real.is_stock)[0][0]
    gap = 0.4 * cover[stock] * se[stock]
    inv[stock] -= gap
    zero_leak = np.zeros_like(se)
    plan = plan_output(
        se,
        inv,
        cover,
        zero_leak,
        tau_inv,
        real.flat(real.backlog0),
        cfg.dynamics.production.tau_backlog_m,
        real.is_stock,
        real.flat(real.K0) * 10.0,
    )
    # With leak off, baseline plan is s0 and the gap term is exactly gap/τ_inv.
    assert plan[stock] - se[stock] == pytest.approx(gap / tau_inv[stock], abs=1e-9)


def test_labour_cap_binds(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    ell = real.flat(real.ell)
    k0 = real.flat(real.K0)
    fc = np.array([cfg.sectors.params(c).fixed_cost for c in real.codes])
    ustar = np.array([cfg.sectors.params(c).util_target for c in real.codes])
    n = 0.01 * real.flat(real.n0)
    cap = labour_cap(n, fc, ell, k0, ustar, overtime=1.10, z_sup=np.zeros(real.S))
    assert (cap < real.flat(real.x0)).all()


def test_input_cap_binds_on_critical_stockout(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    a = io.A
    s_in = real.S_in0[0].copy()
    crit = real.crit
    stor = real.stor_in
    # Drain a critical storable input for the first buyer that has one.
    buyers = np.where((stor & crit).any(axis=0))[0]
    j = int(buyers[0])
    i = int(np.where(stor[:, j] & crit[:, j])[0][0])
    s_in[i, j] = 0.0
    cap = input_cap(s_in, a, stor, crit)
    assert cap[j] == pytest.approx(0.0, abs=1e-12)


def test_capacity_clip_binds(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    k_eff = 0.5 * real.flat(real.x0)
    plan = plan_output(
        real.flat(real.s0),
        real.flat(real.inv0),
        real.flat(real.cover),
        real.flat(real.leak),
        real.flat(real.tau_inv),
        real.flat(real.backlog0),
        cfg.dynamics.production.tau_backlog_m,
        real.is_stock,
        k_eff,
    )
    assert np.allclose(plan, k_eff, atol=1e-12)


def test_critical_mask_energy_utilities_not_bizsvc(cfg, io) -> None:
    mask = critical_mask(io.A, io.mu, cfg)
    idx = {c: i for i, c in enumerate(io.codes)}
    assert mask[idx["ENERGY"], idx["UTILITIES"]]
    assert not mask[idx["BIZSVC"]].any()
