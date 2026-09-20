"""T8.06 — demand product and asset-return log-sum identities (1e-9)."""

from __future__ import annotations

import math

import pytest

from marketsim.explain.traces import (
    TRACE_ATOL,
    asset_return_trace,
    check_demand_change,
    check_demand_identity,
    check_return_identity,
    default_demand_trace,
    demand_change,
    demand_identity_gap,
    demand_trace,
    return_identity_gap,
)


def _sample_demand():
    return demand_trace(
        baseline=100.0,
        income=1.1,
        relative_price=0.95,
        rate=0.98,
        typed_edge=1.02,
        want_shifter=0.9,
        events=1.2,
        own_price=0.85,
        availability=0.7,
        stickiness=1.05,
        rationing=0.8,
    )


def _sample_return():
    return asset_return_trace(
        ee0=10.0,
        ee1=11.0,
        pe0_0=12.0,
        pe0_1=12.0,
        duration=7.0,
        rho0=0.05,
        rho1=0.06,
        g0=0.02,
        g1=0.025,
        impact0=0.0,
        impact1=0.01,
        sentiment0=0.02,
        sentiment1=0.01,
        noise0=0.0,
        noise1=0.001,
    )


def test_demand_product_identity() -> None:
    trace = _sample_demand()
    cell = (
        100.0 * 1.1 * 0.95 * 0.98 * 1.02 * 0.9 * 1.2
    )
    share = 0.85 * 0.7 * 1.05
    qty = cell * share * 0.8
    assert trace.cell_demand == pytest.approx(cell, abs=TRACE_ATOL)
    assert trace.share == pytest.approx(share, abs=TRACE_ATOL)
    assert trace.qty == pytest.approx(qty, abs=TRACE_ATOL)
    check_demand_identity(trace)
    assert demand_identity_gap(trace) <= TRACE_ATOL


def test_demand_log_change_sums() -> None:
    before = default_demand_trace()
    before = demand_trace(baseline=100.0)
    after = _sample_demand()
    parts = demand_change(before, after)
    keys = (
        "baseline",
        "income",
        "relative_price",
        "rate",
        "typed_edge",
        "want_shifter",
        "events",
        "own_price",
        "availability",
        "stickiness",
        "rationing",
    )
    assert sum(parts[k] for k in keys) == pytest.approx(parts["dlog_qty"], abs=TRACE_ATOL)
    check_demand_change(before, after)
    assert parts["dlog_qty"] == pytest.approx(math.log(after.qty / before.qty), abs=TRACE_ATOL)


def test_return_log_sum_identity() -> None:
    trace = _sample_return()
    earnings = math.log(11.0 / 10.0)
    pe0 = math.log(12.0 / 12.0)
    discount = -7.0 * (0.06 - 0.05)
    growth = 7.0 * (0.025 - 0.02)
    dln_v = earnings + pe0 + discount + growth
    dxi = 0.01 + (0.01 - 0.02) + 0.001
    assert trace.earnings == pytest.approx(earnings, abs=TRACE_ATOL)
    assert trace.pe0 == pytest.approx(pe0, abs=TRACE_ATOL)
    assert trace.discount == pytest.approx(discount, abs=TRACE_ATOL)
    assert trace.growth == pytest.approx(growth, abs=TRACE_ATOL)
    assert trace.dln_v == pytest.approx(dln_v, abs=TRACE_ATOL)
    assert trace.dxi == pytest.approx(dxi, abs=TRACE_ATOL)
    assert trace.dln_p == pytest.approx(dln_v + dxi, abs=TRACE_ATOL)
    check_return_identity(trace)
    assert return_identity_gap(trace) <= TRACE_ATOL
    assert abs(trace.earnings + trace.pe0 + trace.discount + trace.growth + trace.impact + trace.sentiment + trace.noise - trace.dln_p) <= TRACE_ATOL


def test_identity_checkers_reject_broken_parts() -> None:
    good = _sample_demand()
    broken = type(good)(**{**good.to_state(), "qty": good.qty + 1.0})
    with pytest.raises(AssertionError):
        check_demand_identity(broken)
    ret = _sample_return()
    broken_ret = type(ret)(**{**ret.to_state(), "dln_p": ret.dln_p + 1.0})
    with pytest.raises(AssertionError):
        check_return_identity(broken_ret)
