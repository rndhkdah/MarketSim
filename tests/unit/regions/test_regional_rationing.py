"""T3.04 — regional routing, link caps, spill, source rationing, R=1 SFC."""

from __future__ import annotations

import numpy as np

from marketsim.real.orders import route_regional_orders
from marketsim.real.steady_state import compute_regional_real_baseline
from marketsim.regions.geometry import build_geometry
from marketsim.regions.trade import (
    dest_demand,
    flows_from_shares,
    link_capacity,
    source_orders,
)
from marketsim.scenarios.irf import make_economy


def _ss(cfg, io):
    geom = build_geometry(cfg.regions, cfg.codes)
    reg = compute_regional_real_baseline(io, cfg, geom)
    dest = dest_demand(io.A, reg.x0, reg.d0)
    sourced = source_orders(reg.T0, dest)
    base = flows_from_shares(reg.T0, dest)
    cap = link_capacity(geom.capacity_mult, base)
    return geom, reg, dest, sourced, base, cap


def test_conservation_deliveries_equal_sales(cfg, io) -> None:
    geom, reg, dest, sourced, base, cap = _ss(cfg, io)
    deliv, sales, fill = route_regional_orders(dest, reg.T0, sourced, cap)
    assert np.max(np.abs(deliv.sum(axis=2).T - sales)) < 1e-12
    assert np.max(np.abs(sales - sourced)) < 1e-9
    assert np.max(np.abs(fill - 1.0)) < 1e-9


def test_link_cut_spills_then_rations(cfg, io) -> None:
    geom, reg, dest, sourced, base, cap = _ss(cfg, io)
    idx = {c: i for i, c in enumerate(geom.codes)}
    energy = cfg.codes.index("ENERGY")
    i_ind, i_cap, i_res = idx["INDUSTRIAL"], idx["CAPITAL"], idx["RESOURCE"]
    base_ind_cap = float(base[energy, i_ind, i_cap])
    base_res_cap = float(base[energy, i_res, i_cap])
    assert base_ind_cap > 0.0 and base_res_cap > 0.0

    tight = cap.copy()
    tight[energy, i_ind, i_cap] = 0.10 * base_ind_cap
    tight[energy, i_cap, i_ind] = 0.10 * float(base[energy, i_cap, i_ind])
    deliv, sales, fill = route_regional_orders(dest, reg.T0, sourced, tight)

    assert float(deliv[energy, i_ind, i_cap]) <= 0.10 * base_ind_cap + 1e-9
    assert float(deliv[energy, i_res, i_cap]) > base_res_cap + 1e-9
    # extra RESOURCE→CAPITAL demand exceeds SS availability → source rationing
    assert float(fill[i_res, energy]) < 1.0 - 1e-12
    assert np.max(np.abs(deliv.sum(axis=2).T - sales)) < 1e-12


def test_r1_ledger_sfc_unaffected(config_dir) -> None:
    eco = make_economy(
        config_dir,
        pi_star=0.0,
        overrides={"dynamics.banks.mode": "passthrough"},
        check_sfc=True,
    )
    dest = eco.sales.copy()
    t0 = eco.real.T0
    cap = link_capacity(eco.geom.capacity_mult, flows_from_shares(t0, dest[None, :]))
    deliv, sales, fill = route_regional_orders(dest[None, :], t0, eco.x[None, :], cap)
    assert deliv.shape == (eco.real.S, 1, 1)
    assert np.max(np.abs(sales[0] - dest)) < 1e-9 or fill[0].min() <= 1.0
    for _ in range(6):
        eco.step_month()
    assert eco.month == 6
