"""T6.31 — collect bond-market gates 9–16 and GB_BOND Tier-2."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest

from marketsim.core.config import Config
from marketsim.market.bond_market import DEFAULT_BOND_SIGMA, BondSecondaryMarket
from marketsim.market.metrics import loglog_slope, peak_impact, revert_fraction, round_trip_pnl
from marketsim.pricing.bond_buckets import BUCKET_ORDER, DURATION_REF_YIELD, bucket_duration_years
from marketsim.pricing.curve import KAPPA_DEBT, KAPPA_QE, term_premium_bucket
from marketsim.real.policy.cb_operations import opening_mm_book

SIZES = (0.001, 0.003, 0.01, 0.03, 0.10, 0.30)
HORIZONS = (1, 2, 5, 10, 20)


def _gb_bond_xi_path(cfg: Config, q_over_adv: float, duration_d: int) -> np.ndarray:
    """Execute ``q_over_adv · ADV`` of ``GB_BOND`` over ``duration_d`` days, then 20 quiet days."""
    face = 1_000_000.0
    mkt = BondSecondaryMarket.from_config(
        cfg,
        face={b: face for b in BUCKET_ORDER},
        holdings=opening_mm_book(face),
        fair_yield=DURATION_REF_YIELD,
        sigma=DEFAULT_BOND_SIGMA,
    )
    adv = mkt.adv("GB_BOND")
    days = int(duration_d)
    q_day = float(q_over_adv) * adv / float(days)
    xs: list[float] = []
    for _ in range(days):
        mkt.kernels["GB_BOND"].step(q_day, adv, mkt.sigma["GB_BOND"])
        xs.append(float(mkt.kernels["GB_BOND"].xi))
    for _ in range(20):
        mkt.kernels["GB_BOND"].step(0.0, adv, mkt.sigma["GB_BOND"])
        xs.append(float(mkt.kernels["GB_BOND"].xi))
    return np.asarray(xs, dtype=float)


def _call(module: str, name: str, **kwargs: object) -> None:
    fn = getattr(import_module(module), name)
    fn(**kwargs)


@pytest.mark.validation
def test_gates_9_to_16_collected(cfg: Config, config_dir, io) -> None:
    """Re-run the named gate tests so T6.31 fails if any of 9–16 regresses."""
    del config_dir
    _call("tests.unit.pricing.test_bond_buckets", "test_par_opening_bitwise_phase2", cfg=cfg, io=io)
    _call("tests.unit.pricing.test_term_premium", "test_gate10_plus_100bp_gb_bond_35bp_and_bill_quiet", cfg=cfg)
    _call("tests.market.test_bond_secondary", "test_gate_11_regime_flip_standin", cfg=cfg)
    _call("tests.market.test_auctions", "test_gate12_calm_cover_and_tail")
    _call("tests.market.test_cb_operations", "test_gate13_qe_then_qt_reserves_and_yield", cfg=cfg)
    _call("tests.market.test_cb_operations", "test_gate14_bond_financed_raises_tp_and_lowers_capex", cfg=cfg)
    _call("tests.validation.test_financials_mtm", "test_gate15_insurance_mtm_from_ledger_exceeds_banks", cfg=cfg)
    _call("tests.market.test_corp_pool", "test_gate16_default_raises_all_firm_rates_equally")


@pytest.mark.validation
def test_gate3_repeated_on_gb_bond(cfg: Config) -> None:
    """§6.1 gate 3 on the live ``GB_BOND`` kernel (same size/horizon grid as T6.21)."""
    peaks = [peak_impact(_gb_bond_xi_path(cfg, q, 5), 5) for q in SIZES]
    delta = loglog_slope(np.array(SIZES), np.array(peaks))
    assert 0.4 <= delta <= 0.7
    peak_h = [peak_impact(_gb_bond_xi_path(cfg, 0.05, d), d) for d in HORIZONS]
    slope_h = loglog_slope(np.array(HORIZONS, dtype=float), np.array(peak_h))
    assert 0.0 <= slope_h <= 0.25
    assert peaks[-1] / peaks[0] < (SIZES[-1] / SIZES[0])
    for d in (5, 10, 20):
        path = _gb_bond_xi_path(cfg, 0.10, d)
        assert 0.40 <= revert_fraction(path, d) <= 0.75
        assert round_trip_pnl(path, d) <= 0.0


@pytest.mark.validation
def test_measured_kappa_debt_and_kappa_qe(cfg: Config) -> None:
    """Recover §6.11 κ_debt / κ_qe from ``tp_b`` at GB_BOND duration (T8.04 flags)."""
    assert cfg.bonds is not None
    d = bucket_duration_years(DURATION_REF_YIELD, cfg.bonds.buckets["GB_BOND"].decay)
    base = dict(debt_to_gdp=0.60, debt_to_gdp_star=0.60, h_cb=0.0, h_cb0=0.0, z_risk=0.0)
    tp0 = term_premium_bucket(d, **base)
    tp_debt = term_premium_bucket(d, **{**base, "debt_to_gdp": 0.70})
    tp_qe = term_premium_bucket(d, **{**base, "h_cb": 0.10})
    scale = d / 7.0
    kappa_debt_hat = (tp_debt - tp0) / (0.10 * scale)
    kappa_qe_hat = (tp0 - tp_qe) / (0.10 * scale)
    assert kappa_debt_hat == pytest.approx(KAPPA_DEBT)
    assert kappa_qe_hat == pytest.approx(KAPPA_QE)
    assert (tp_debt - tp0) == pytest.approx(scale * KAPPA_DEBT * 0.10)
    assert (tp_qe - tp0) == pytest.approx(-scale * KAPPA_QE * 0.10)
