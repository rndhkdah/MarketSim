"""T6.08 — engine market maker (§6.5)."""

from __future__ import annotations

import inspect

import pytest

from marketsim.core.config import Config
from marketsim.market.clob import CLOB
from marketsim.market.clob import Fill as ClobFill
from marketsim.market.clob import Order as ClobOrder
from marketsim.market.clob import Side as ClobSide
from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import MMCfg
from marketsim.market.mm import (
    MM_ACCOUNT,
    EngineMM,
    inventory_skew,
    liquidity_budget,
    quote_prices,
    quoted_spread,
)
from marketsim.market.venue import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    Venue,
)

SYM = "EQ:NPC:AUTOS"


def _mm(
    cfg: MMCfg | None = None,
    *,
    mid: float = 1.0,
    sigma: float = 0.01,
    cap: float = 1_000_000.0,
    adv: float = 100.0,
    inventory: float = 0.0,
) -> EngineMM:
    mm = EngineMM(cfg)
    mm.add_book(SYM, mid=mid, sigma=sigma, cap=cap, adv=adv, inventory=inventory)
    return mm


def _order(
    agent: str,
    side: Side,
    qty: int,
    *,
    order_type: OrderType = OrderType.MARKET,
    tif: TimeInForce = TimeInForce.GTC,
    price: float | None = None,
    stop_price: float | None = None,
    sequence: int | None = None,
) -> Order:
    return Order(
        agent_id=agent,
        side=side,
        qty=qty,
        symbol=SYM,
        order_type=order_type,
        tif=tif,
        price=price,
        stop_price=stop_price,
        sequence=sequence,
    )


def test_spread_widens_with_volatility_and_inventory() -> None:
    cfg = MMCfg()
    cap = 1_000_000.0
    sigma = 0.04
    inv = 50_000.0
    s0 = quoted_spread(0.0, 0.0, cap, cfg)
    s_vol = quoted_spread(sigma, 0.0, cap, cfg)
    s_inv = quoted_spread(0.0, inv, cap, cfg)
    s_both = quoted_spread(sigma, inv, cap, cfg)
    assert s0 == pytest.approx(cfg.s0)
    assert s_vol == pytest.approx(cfg.s0 + cfg.k_sigma * sigma)
    assert s_inv == pytest.approx(cfg.s0 + cfg.k_inv * abs(inv) / cap)
    assert s_both == pytest.approx(cfg.s0 + cfg.k_sigma * sigma + cfg.k_inv * abs(inv) / cap)
    assert s_vol > s0
    assert s_inv > s0
    assert s_both > s_vol
    assert s_both > s_inv

    mm = _mm(cfg, sigma=0.0, inventory=0.0, cap=cap)
    flat = mm.quote(SYM).spread
    mm.set_mark(SYM, sigma=sigma)
    assert mm.quote(SYM).spread > flat
    mm.set_mark(SYM, sigma=0.0, inventory=inv)
    assert mm.quote(SYM).spread > flat


def test_budget_partial_fills(cfg: Config) -> None:
    assert cfg.markets is not None
    mm_cfg = cfg.markets.mm
    assert mm_cfg.participation_cap == pytest.approx(0.25)
    mid = 1.0
    adv = 80.0
    budget = liquidity_budget(adv, mm_cfg)
    assert budget == pytest.approx(mm_cfg.participation_cap * adv)
    cap_qty = int(budget / mid)
    assert cap_qty == 20

    mm = _mm(mm_cfg, mid=mid, adv=adv)
    parked = mm.submit(_order("alice", Side.BUY, 50, tif=TimeInForce.GTC))
    assert parked.fills == ()
    fills = mm.end_tick()
    assert len(fills) == 1
    assert fills[0].qty == cap_qty
    assert fills[0].maker == MM_ACCOUNT
    assert fills[0].taker == "alice"
    assert parked.order_id in mm.resting_ids(SYM)
    leftover = 50 - cap_qty
    assert leftover == 30

    ioc = _mm(mm_cfg, mid=mid, adv=adv)
    ioc_res = ioc.submit(_order("bob", Side.SELL, 50, tif=TimeInForce.IOC))
    ioc_fills = ioc.end_tick()
    assert ioc_fills[0].qty == cap_qty
    assert ioc_res.order_id not in ioc.resting_ids(SYM)


def test_next_tick_fills() -> None:
    mm = _mm()
    qte = mm.quote(SYM)
    accepted = mm.submit(_order("carol", Side.BUY, 4, sequence=1))
    assert accepted.status is OrderStatus.RESTING
    assert accepted.fills == ()
    assert mm.last_fills == ()
    assert accepted.order_id in mm.resting_ids(SYM)

    fills = mm.end_tick()
    assert len(fills) == 1
    assert fills[0].taker == "carol"
    assert fills[0].maker == MM_ACCOUNT
    assert fills[0].qty == 4
    assert fills[0].tick == 0
    assert accepted.order_id not in mm.resting_ids(SYM)
    assert mm.clock_tick == 1

    kn = ImpactKernel()
    q = 4.0 * qte.mid
    expected = kn.fill(qte.skewed_mid, qte.spread, q, 100.0, 0.01)
    assert fills[0].price == pytest.approx(float(expected))
    assert fills[0].price > qte.ask  # post-impact; nobody trades ahead of their own impact


def test_skew_sign() -> None:
    cfg = MMCfg()
    cap = 1_000_000.0
    inv = 80_000.0
    long_skew = inventory_skew(inv, cap, cfg)
    short_skew = inventory_skew(-inv, cap, cfg)
    assert long_skew == pytest.approx(-cfg.k_skew * inv / cap)
    assert long_skew < 0.0
    assert short_skew > 0.0
    assert short_skew == pytest.approx(-long_skew)

    q_flat = quote_prices(100.0, 0.01, 0.0, cap, cfg)
    q_long = quote_prices(100.0, 0.01, inv, cap, cfg)
    q_short = quote_prices(100.0, 0.01, -inv, cap, cfg)
    assert q_long.skewed_mid < q_flat.skewed_mid < q_short.skewed_mid
    # Same |inv| → same spread; long mid is below short mid.
    assert q_long.spread == pytest.approx(q_short.spread)
    assert q_long.bid < q_short.bid
    assert q_long.ask < q_short.ask

    mm = _mm(cfg, mid=100.0, cap=cap, inventory=inv)
    quoted = mm.quote(SYM)
    assert quoted.skew < 0.0
    assert quoted.skewed_mid < quoted.mid


def test_venue_protocol_shared_with_clob() -> None:
    mm = EngineMM()
    clob = CLOB(tick=0.01, lot=1, halt_band=0.20)
    assert isinstance(mm, Venue)
    assert isinstance(clob, Venue)
    assert Order is ClobOrder
    assert Fill is ClobFill
    assert Side is ClobSide

    for cls in (EngineMM, CLOB):
        cancel = inspect.signature(cls.cancel)
        assert list(cancel.parameters) == ["self", "symbol", "order_id"]
        repl = inspect.signature(cls.replace)
        assert list(repl.parameters) == ["self", "symbol", "order_id", "qty", "price"]
        assert repl.parameters["qty"].kind is inspect.Parameter.KEYWORD_ONLY
        assert "order" in inspect.signature(cls.submit).parameters
        assert "orders" in inspect.signature(cls.ingest).parameters
        end = inspect.signature(cls.end_tick)
        required = [
            p
            for p in end.parameters.values()
            if p.default is inspect.Parameter.empty and p.name != "self"
        ]
        assert required == []
