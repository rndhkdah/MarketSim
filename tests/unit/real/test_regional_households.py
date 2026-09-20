"""T3.07 — regional household incomes and national government purchases."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.ledger.journal import Entry, Tx
from marketsim.ledger.opening import build_phase2_ledger, post_opening_passthrough
from marketsim.ledger.sfc import assert_consistent
from marketsim.real.government import allocate_by_population
from marketsim.real.households import (
    household_entity,
    regional_disposable,
    regional_incomes,
    split_regional,
    wealth_from_ledger,
)
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline
from marketsim.regions.geometry import build_geometry


def test_regional_incomes_sum_to_national(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io)
    geom = build_geometry(cfg.regions, cfg.codes)
    yd_r = split_regional(fin.YD0, geom.population_share)
    w_r = split_regional(fin.W, geom.population_share)
    yd_n, w_n = regional_incomes(yd_r, w_r)
    assert yd_n == pytest.approx(fin.YD0, abs=1e-12)
    assert w_n == pytest.approx(fin.W, abs=1e-12)
    pretax = split_regional(fin.pretax0, geom.population_share)
    assert float(regional_disposable(pretax, fin.tau_y).sum()) == pytest.approx(fin.YD0, abs=1e-9)


def test_government_purchases_split_by_population(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    geom = build_geometry(cfg.regions, cfg.codes)
    g = real.flat(real.G0)
    g_rs = allocate_by_population(g, geom.population_share)
    assert g_rs.shape == (geom.n_regions, real.S)
    assert np.max(np.abs(g_rs.sum(axis=0) - g)) < 1e-12
    shares = g_rs.sum(axis=1) / g.sum()
    assert np.allclose(shares, geom.population_share, atol=1e-12)


def test_sfc_per_region_household_entity(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io)
    led = build_phase2_ledger(cfg, real, n_regions=3)
    post_opening_passthrough(led, real, fin, region=0)
    assert_consistent(led, 0)
    for r in range(3):
        assert household_entity(r) in led.entities
    # national opening lives on HH:0; split a slice of W to HH:1 and HH:2, SFC-balanced
    slice_w = 0.1 * fin.W
    led.post(
        Tx(
            1,
            "transfers",
            [
                Entry("HH:0", "DEP", -slice_w),
                Entry("HH:1", "DEP", 0.5 * slice_w),
                Entry("HH:2", "DEP", 0.5 * slice_w),
            ],
            memo="regional HH split",
        )
    )
    assert_consistent(led, 1)
    assert wealth_from_ledger(led, "HH:1") == pytest.approx(0.05 * fin.W, abs=1e-8)
