from __future__ import annotations

import pytest

from marketsim.ledger.opening import open_passthrough_books
from marketsim.ledger.sfc import assert_consistent
from marketsim.real.government import baseline_deficit, iterate_debt_ratio, post_bond_issue
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_baseline_deficit_is_grow_b(cfg, io, pi_star: float) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=pi_star)
    assert baseline_deficit(real, fin) == pytest.approx(fin.grow * fin.B, abs=1e-4)


def test_bond_issue_sfc(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    led = open_passthrough_books(cfg, real, fin)
    post_bond_issue(led, 10.0, tick=1)
    assert_consistent(led, 1)
    assert led.position("GOVT", "GB_BILL") == pytest.approx(-fin.B * 0.20 - 2.0, abs=1e-8)


def test_debt_ratio_half_life_under_25y(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    fisc = cfg.dynamics.fiscal
    g_nom = float(real.flat(real.G0).sum())
    d_b = 0.10 * 12.0 * fin.gdp0
    ratios = iterate_debt_ratio(
        b0=fin.B + d_b,
        tau0=fin.tau_y,
        gdp0=fin.gdp0,
        pretax=fin.pretax0,
        g_nom=g_nom,
        transfers0=fin.transfers0,
        ctax=float(fin.tax0.sum()),
        r=fin.r0,
        kappa=fisc.kappa_debt,
        debt_to_gdp=fisc.debt_to_gdp,
        bounds=fisc.tax_rate_bounds,
        tau_m=fisc.tau_tax_m,
        months=25 * 12,
    )
    assert ratios[0] == pytest.approx(0.70, abs=0.01)
    # Partial-equilibrium books (no GDP feedback) close a 10pp gap to ≤6pp within 25y
    # and keep falling — the full-economy half-life is checked in T2.17/T2.23.
    assert ratios[-1] <= 0.66
    assert ratios[-1] < ratios[12]


@pytest.mark.slow
def test_no_rule_diverges_when_r_gt_g(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    fisc = cfg.dynamics.fiscal
    g_nom = float(real.flat(real.G0).sum())
    ratios = iterate_debt_ratio(
        b0=fin.B,
        tau0=fin.tau_y,
        gdp0=fin.gdp0,
        pretax=fin.pretax0,
        g_nom=g_nom,
        transfers0=fin.transfers0,
        ctax=float(fin.tax0.sum()),
        r=0.04,
        kappa=0.0,
        debt_to_gdp=fisc.debt_to_gdp,
        bounds=fisc.tax_rate_bounds,
        tau_m=fisc.tau_tax_m,
        months=40 * 12,
    )
    assert ratios[-1] > ratios[0] + 0.05
