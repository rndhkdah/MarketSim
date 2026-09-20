"""T3.03 — baseline T0, stacked Leontief, income fixed point, wage-level offset."""

from __future__ import annotations

import numpy as np

from marketsim.real.steady_state import (
    compute_real_baseline,
    compute_regional_real_baseline,
    national_geometry,
)
from marketsim.regions.geometry import build_geometry
from marketsim.regions.trade import baseline_trade_shares, dest_demand, source_orders


def test_t0_rows_sum_to_one(cfg) -> None:
    geom = build_geometry(cfg.regions, cfg.codes)
    t0 = baseline_trade_shares(geom)
    assert t0.shape == (len(cfg.codes), geom.n_regions, geom.n_regions)
    assert np.allclose(t0.sum(axis=1), 1.0, atol=1e-12)


def test_nontradables_are_local(cfg) -> None:
    geom = build_geometry(cfg.regions, cfg.codes)
    t0 = baseline_trade_shares(geom)
    re = cfg.codes.index("REALESTATE")
    assert geom.tradability[re] == 0.0
    assert np.allclose(t0[re], np.eye(geom.n_regions), atol=1e-15)


def test_r1_regional_matches_national(cfg, io) -> None:
    nat = compute_real_baseline(io, cfg)
    geo = national_geometry(nat.codes)
    reg = compute_regional_real_baseline(io, cfg, geo)
    assert reg.R == 1
    assert np.max(np.abs(reg.x0 - nat.x0)) < 1e-12
    assert np.max(np.abs(reg.d0 - nat.d0)) < 1e-12
    assert np.max(np.abs(reg.C0 - nat.C0)) < 1e-12


def test_national_totals_match_phase2(cfg, io) -> None:
    nat = compute_real_baseline(io, cfg)
    geom = build_geometry(cfg.regions, cfg.codes)
    reg = compute_regional_real_baseline(io, cfg, geom)
    assert reg.R == 3
    assert np.max(np.abs(reg.x0.sum(axis=0) - nat.flat(nat.x0))) < 1e-9
    assert np.max(np.abs(reg.s0.sum(axis=0) - nat.flat(nat.s0))) < 1e-9
    assert np.max(np.abs(reg.C0.sum(axis=0) - nat.flat(nat.C0))) < 1e-9
    assert np.max(np.abs(reg.G0.sum(axis=0) - nat.flat(nat.G0))) < 1e-9
    assert np.max(np.abs(reg.X0.sum(axis=0) - nat.flat(nat.X0))) < 1e-9
    assert np.max(np.abs(reg.I_fd.sum(axis=0) - nat.flat(nat.I_fd))) < 1e-9
    wage_bill = (reg.wage_level[:, None] * reg.n0).sum(axis=0)
    assert np.max(np.abs(wage_bill - nat.flat(nat.n0))) < 1e-9


def test_cells_clear(cfg, io) -> None:
    geom = build_geometry(cfg.regions, cfg.codes)
    reg = compute_regional_real_baseline(io, cfg, geom)
    lam = 1.0 + reg.flat(reg.leak) * reg.flat(reg.cover)
    dest = dest_demand(io.A, reg.x0, reg.d0)
    sourced = source_orders(reg.T0, dest)
    assert np.max(np.abs(sourced - reg.x0 / lam)) < 1e-9
    assert np.max(np.abs(sourced.sum(axis=0) - dest.sum(axis=0))) < 1e-9


def test_resource_exports_energy_capital_exports_software(cfg, io) -> None:
    geom = build_geometry(cfg.regions, cfg.codes)
    reg = compute_regional_real_baseline(io, cfg, geom)
    dest = dest_demand(io.A, reg.x0, reg.d0)
    sourced = source_orders(reg.T0, dest)
    net = sourced - dest
    r_idx = {c: i for i, c in enumerate(geom.codes)}
    energy = cfg.codes.index("ENERGY")
    software = cfg.codes.index("SOFTWARE")
    assert net[r_idx["RESOURCE"], energy] > 0.0
    assert net[r_idx["CAPITAL"], software] > 0.0


def test_wage_level_offsets_unit_labour_cost(cfg, io) -> None:
    geom = build_geometry(cfg.regions, cfg.codes)
    reg = compute_regional_real_baseline(io, cfg, geom)
    nat = compute_real_baseline(io, cfg)
    ell_s = nat.flat(nat.ell)
    ulc = geom.wage_level[:, None] * reg.ell
    assert np.max(np.abs(ulc - ell_s[None, :])) < 1e-12
    assert np.allclose(reg.wage_level, geom.wage_level)
