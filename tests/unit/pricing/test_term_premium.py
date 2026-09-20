"""T6.25 — bucket fair yields, term premium, guidance, gate 10."""

from __future__ import annotations

import pytest

from marketsim.core.config import Config
from marketsim.pricing.bond_buckets import (
    DURATION_REF_YIELD,
    bucket_duration_years,
    bucket_price,
)
from marketsim.pricing.curve import (
    DURATION_BENCH_Y,
    HORIZON_M,
    KAPPA_DEBT,
    KAPPA_QE,
    blend_expected_path,
    bucket_fair_yield,
    cash_flow_weights,
    expected_policy_path,
    term_premium_bucket,
    y10,
)


def _weights(cfg: Config, name: str) -> tuple:
    spec = cfg.bonds.buckets[name]
    w = cash_flow_weights(spec.decay, DURATION_REF_YIELD)
    return spec, w


def test_gate10_plus_100bp_gb_bond_35bp_and_bill_quiet(cfg: Config) -> None:
    assert cfg.bonds is not None
    r_n, pi_star = DURATION_REF_YIELD, 0.0
    r0 = r_n + pi_star
    r1 = r0 + 0.01
    path0 = expected_policy_path(r0, r_n, pi_star)
    path1 = expected_policy_path(r1, r_n, pi_star)
    spec_b, w_b = _weights(cfg, "GB_BOND")
    spec_i, w_i = _weights(cfg, "GB_BILL")
    yb0 = bucket_fair_yield(path0, w_b)
    yb1 = bucket_fair_yield(path1, w_b)
    yi0 = bucket_fair_yield(path0, w_i)
    yi1 = bucket_fair_yield(path1, w_i)
    dy_bond = yb1 - yb0
    dy_bill = yi1 - yi0
    # Gate 10: GB_BOND +35bp (±10); the 10-year average is also +35bp.
    assert dy_bond == pytest.approx(0.0035, abs=0.0010)
    assert y10(r1, r_n, pi_star) - y10(r0, r_n, pi_star) == pytest.approx(0.0035, abs=0.0005)
    p0 = bucket_price(yb0, DURATION_REF_YIELD, spec_b.decay)
    p1 = bucket_price(yb1, DURATION_REF_YIELD, spec_b.decay)
    assert p0 == pytest.approx(1.0)
    assert p1 / p0 - 1.0 == pytest.approx(-0.024, abs=0.007)
    p_bill0 = bucket_price(yi0, DURATION_REF_YIELD, spec_i.decay)
    p_bill1 = bucket_price(yi1, DURATION_REF_YIELD, spec_i.decay)
    assert abs(p_bill1 / p_bill0 - 1.0) < 0.003
    # Curve flattens: front-end yield rises more than the 10-year.
    assert dy_bill > dy_bond


def test_tp_rises_with_debt_falls_with_qe_scaled_by_duration(cfg: Config) -> None:
    assert cfg.bonds is not None
    d_bond = bucket_duration_years(DURATION_REF_YIELD, cfg.bonds.buckets["GB_BOND"].decay)
    d_bill = bucket_duration_years(DURATION_REF_YIELD, cfg.bonds.buckets["GB_BILL"].decay)
    base = dict(
        debt_to_gdp=0.60,
        debt_to_gdp_star=0.60,
        h_cb=0.0,
        h_cb0=0.0,
        z_risk=0.0,
    )
    tp_debt = term_premium_bucket(d_bond, **{**base, "debt_to_gdp": 0.70})
    tp_qe = term_premium_bucket(d_bond, **{**base, "h_cb": 0.10})
    assert tp_debt == pytest.approx((d_bond / DURATION_BENCH_Y) * KAPPA_DEBT * 0.10)
    assert tp_qe == pytest.approx(-(d_bond / DURATION_BENCH_Y) * KAPPA_QE * 0.10)
    tp_bill_debt = term_premium_bucket(d_bill, **{**base, "debt_to_gdp": 0.70})
    assert tp_bill_debt < tp_debt
    assert tp_bill_debt == pytest.approx(tp_debt * (d_bill / d_bond))


def test_flight_to_quality_lowers_yields() -> None:
    calm = term_premium_bucket(7.0, debt_to_gdp=0.6, debt_to_gdp_star=0.6, h_cb=0.0, h_cb0=0.0, z_risk=0.0)
    stress = term_premium_bucket(7.0, debt_to_gdp=0.6, debt_to_gdp_star=0.6, h_cb=0.0, h_cb0=0.0, z_risk=0.02)
    assert stress < calm


def test_guidance_moves_path_with_credibility() -> None:
    rule = expected_policy_path(0.05, 0.01, 0.02)
    guide = expected_policy_path(0.01, 0.01, 0.02)  # hold at 3 %
    none = blend_expected_path(rule, None, 0.7)
    assert none == pytest.approx(rule)
    half = blend_expected_path(rule, guide, 0.5)
    assert half == pytest.approx(0.5 * rule + 0.5 * guide)
    full = blend_expected_path(rule, guide, 1.0)
    assert full == pytest.approx(guide)
    w = cash_flow_weights(0.10, DURATION_REF_YIELD)
    y_rule = bucket_fair_yield(rule, w)
    y_half = bucket_fair_yield(half, w)
    y_full = bucket_fair_yield(full, w)
    assert y_full < y_half < y_rule


def test_cash_flow_weights_sum_to_one() -> None:
    w = cash_flow_weights(0.10, 0.042, horizon_m=HORIZON_M)
    assert w.shape == (HORIZON_M,)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[-1]
