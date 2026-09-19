from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from marketsim.real.orders import order_matrix, supply_and_ration
from marketsim.real.production import goods_output, input_cap, labour_cap, plan_output
from marketsim.real.steady_state import compute_real_baseline


def _fc_ustar(cfg, codes):
    fc = np.array([cfg.sectors.params(c).fixed_cost for c in codes])
    ustar = np.array([cfg.sectors.params(c).util_target for c in codes])
    return fc, ustar


def _baseline_supply(cfg, io):
    real = compute_real_baseline(io, cfg)
    dyn = cfg.dynamics
    se = real.flat(real.s0)
    inv = real.flat(real.inv0)
    k = real.flat(real.K0)
    fc, ustar = _fc_ustar(cfg, real.codes)
    plan = plan_output(
        se,
        inv,
        real.flat(real.cover),
        real.flat(real.leak),
        real.flat(real.tau_inv),
        real.flat(real.backlog0),
        dyn.production.tau_backlog_m,
        real.is_stock,
        k,
    )
    lab = labour_cap(real.flat(real.n0), fc, real.flat(real.ell), k, ustar, dyn.production.overtime_cap, np.zeros(real.S))
    in_c = input_cap(real.S_in0[0], io.A, real.stor_in, real.crit)
    x_goods = goods_output(plan, k, lab, in_c, np.ones(real.S))
    orders = order_matrix(
        io.A, plan, real.S_in0[0], real.stor_in, dyn.production.input_cover_m, dyn.production.tau_input_m
    )
    final = real.flat(real.d0)
    new_orders = orders.sum(axis=1) + final
    res = supply_and_ration(
        x_goods=x_goods,
        new_orders=new_orders,
        final=final,
        orders=orders,
        a=io.A,
        inv=inv,
        backlog=real.flat(real.backlog0),
        s_in=real.S_in0[0],
        leak=real.flat(real.leak),
        tau_ob=real.flat(real.tau_ob),
        cap=k,
        flow_crit=np.ones(real.S),
        is_stock=real.is_stock,
        is_order=real.is_order,
        is_flow=real.is_flow,
        stor_in=real.stor_in,
        backlog_loss=dyn.production.backlog_loss,
    )
    return real, plan, orders, final, new_orders, res


def test_baseline_is_fixed_point(cfg, io) -> None:
    real, _plan, _o, _f, _n, res = _baseline_supply(cfg, io)
    assert np.allclose(res.x, real.flat(real.x0), atol=1e-8)
    assert np.allclose(res.sales, real.flat(real.s0), atol=1e-8)
    assert np.allclose(res.inv, real.flat(real.inv0), atol=1e-8)
    assert np.allclose(res.backlog, real.flat(real.backlog0), atol=1e-8)
    assert np.allclose(res.s_in, real.S_in0[0], atol=1e-8)
    assert np.allclose(res.fill, 1.0, atol=1e-8)


def test_stock_inventory_conservation(cfg, io) -> None:
    real, _p, _o, _f, _new_orders, res = _baseline_supply(cfg, io)
    leak = real.flat(real.leak)
    inv = real.flat(real.inv0)
    stock = real.is_stock
    expect = (1.0 - leak) * inv + res.x - res.sales
    assert np.allclose(res.inv[stock], expect[stock], atol=1e-9)


def test_order_book_conservation(cfg, io) -> None:
    real, _p, _o, _f, new_orders, res = _baseline_supply(cfg, io)
    ob = real.flat(real.backlog0) + new_orders
    order = real.is_order
    assert np.allclose(res.backlog[order], (ob - res.x)[order], atol=1e-9)


def test_input_stocks_nonnegative_under_outage(cfg, io) -> None:
    real, _plan, orders, final, new_orders, _ = _baseline_supply(cfg, io)
    dyn = cfg.dynamics
    energy = real.codes.index("ENERGY")
    x_goods = real.flat(real.x0).copy()
    x_goods[energy] *= 0.5
    cap = real.flat(real.K0).copy()
    cap[energy] *= 0.5
    res = supply_and_ration(
        x_goods=x_goods,
        new_orders=new_orders,
        final=final,
        orders=orders,
        a=io.A,
        inv=real.flat(real.inv0),
        backlog=real.flat(real.backlog0),
        s_in=real.S_in0[0],
        leak=real.flat(real.leak),
        tau_ob=real.flat(real.tau_ob),
        cap=cap,
        flow_crit=np.ones(real.S),
        is_stock=real.is_stock,
        is_order=real.is_order,
        is_flow=real.is_flow,
        stor_in=real.stor_in,
        backlog_loss=dyn.production.backlog_loss,
    )
    assert np.all(res.s_in >= -1e-12)


@given(
    fill=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_deliveries_plus_final_equal_sales(fill: float, scale: float) -> None:
    rng = np.random.default_rng(0)
    s = 5
    a = np.abs(rng.normal(0.05, 0.02, size=(s, s)))
    plan = np.abs(rng.normal(2.0, 0.3, size=s)) * scale
    orders = a * plan[None, :]
    final = np.abs(rng.normal(1.0, 0.2, size=s)) * scale
    new_orders = orders.sum(axis=1) + final
    is_stock = np.array([True, True, False, False, False])
    is_order = np.array([False, False, True, False, False])
    is_flow = np.array([False, False, False, True, True])
    x_goods = new_orders * fill
    cap = np.full(s, 1e9)
    res = supply_and_ration(
        x_goods=x_goods,
        new_orders=new_orders,
        final=final,
        orders=orders,
        a=a,
        inv=np.zeros(s),
        backlog=np.zeros(s),
        s_in=np.zeros((s, s)),
        leak=np.zeros(s),
        tau_ob=np.ones(s),
        cap=cap,
        flow_crit=np.ones(s),
        is_stock=is_stock,
        is_order=is_order,
        is_flow=is_flow,
        stor_in=np.zeros((s, s), dtype=bool),
        backlog_loss=0.0,
    )
    allocated = res.deliveries.sum(axis=1) + res.final_sales
    assert np.allclose(allocated, res.sales, atol=1e-9)
