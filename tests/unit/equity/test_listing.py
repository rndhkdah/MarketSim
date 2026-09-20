"""T6.15 — listing and IPO call auction (§6.7)."""

from __future__ import annotations

import pytest

from marketsim.core.config import Config
from marketsim.equity.captable import DEFAULT_FOUNDER_SHARES, CapTable
from marketsim.equity.listing import list_firm
from marketsim.firms.accounts import agent_entity, firm_entity
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent
from marketsim.market.clob import CLOB, Order, OrderType, Side, TimeInForce
from marketsim.market.instruments import InstrumentRegistry, firm_symbol


def _led(*names: str, cash: dict[str, float] | None = None) -> Ledger:
    led = Ledger.empty(debug_journal=True)
    led.register_entity("BANKSYS")
    for name in names:
        led.register_entity(name)
    if cash:
        entries = [Entry(ent, "DEP", float(amt)) for ent, amt in cash.items()]
        entries.append(Entry("BANKSYS", "DEP", -sum(float(a) for a in cash.values())))
        led.post(Tx(0, "opening", tuple(entries)))
    return led


def _open(
    led: Ledger,
    *,
    firm_id: str = "acme",
    founder: str | None = None,
) -> CapTable:
    founder = founder or agent_entity("alice")
    return CapTable.open(led, firm_id=firm_id, founder=founder)


def _clob(cfg: Config) -> CLOB:
    assert cfg.markets is not None
    return CLOB.from_clob_cfg(cfg.markets.clob)


def _buy(agent: str, qty: int, price: float, symbol: str) -> Order:
    return Order(
        agent_id=agent,
        side=Side.BUY,
        qty=qty,
        symbol=symbol,
        order_type=OrderType.LIMIT,
        tif=TimeInForce.GTC,
        price=price,
    )


def _cash(led: Ledger, entity: str) -> float:
    if entity not in led.entities:
        return 0.0
    return led.position(entity, "DEP")


def test_primary_cash_lands_in_the_firm(cfg: Config) -> None:
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    reserve = 10.00
    offered = 100_000
    notional = offered * reserve
    led = _led(bob, cash={bob: notional})
    table = _open(led, founder=founder)
    issuer = firm_entity(table.firm_id)
    symbol = firm_symbol(table.firm_id)
    clob = _clob(cfg)
    registry = InstrumentRegistry()

    result = list_firm(
        table,
        led,
        clob,
        cfg,
        kind="primary",
        shares=offered,
        reserve=reserve,
        bids=(_buy(bob, offered, reserve, symbol),),
        registry=registry,
        tick=1,
    )

    assert result.cash_to == "firm"
    assert result.price == pytest.approx(reserve)
    assert result.volume == offered
    assert sum(f.qty for f in result.fills) == offered
    assert _cash(led, issuer) == pytest.approx(notional)
    assert _cash(led, founder) == pytest.approx(0.0)
    assert _cash(led, bob) == pytest.approx(0.0)
    assert table.holding(led, founder) == pytest.approx(float(DEFAULT_FOUNDER_SHARES))
    assert table.holding(led, bob) == pytest.approx(float(offered))
    assert table.shares_outstanding(led) == pytest.approx(float(DEFAULT_FOUNDER_SHARES + offered))
    assert table.invariant_holds(led)
    assert symbol in registry
    assert_consistent(led)


def test_secondary_cash_lands_in_the_founder(cfg: Config) -> None:
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    reserve = 10.00
    offered = 100_000
    notional = offered * reserve
    led = _led(bob, cash={bob: notional})
    table = _open(led, founder=founder)
    issuer = firm_entity(table.firm_id)
    symbol = firm_symbol(table.firm_id)
    clob = _clob(cfg)

    result = list_firm(
        table,
        led,
        clob,
        cfg,
        kind="secondary",
        shares=offered,
        reserve=reserve,
        bids=(_buy(bob, offered, reserve, symbol),),
        tick=1,
    )

    assert result.cash_to == "founder"
    assert result.price == pytest.approx(reserve)
    assert result.volume == offered
    assert _cash(led, founder) == pytest.approx(notional)
    assert _cash(led, issuer) == pytest.approx(0.0)
    assert _cash(led, bob) == pytest.approx(0.0)
    assert table.holding(led, founder) == pytest.approx(float(DEFAULT_FOUNDER_SHARES - offered))
    assert table.holding(led, bob) == pytest.approx(float(offered))
    assert table.shares_outstanding(led) == pytest.approx(float(DEFAULT_FOUNDER_SHARES))
    assert table.invariant_holds(led)
    assert_consistent(led)


def test_reserve_price_respected(cfg: Config) -> None:
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    carol = agent_entity("carol")
    reserve = 10.00
    offered = 50_000
    led = _led(bob, carol, cash={bob: 1_000_000.0, carol: 1_000_000.0})
    table = _open(led, founder=founder)
    issuer = firm_entity(table.firm_id)
    symbol = firm_symbol(table.firm_id)
    clob = _clob(cfg)
    before = {
        founder: _cash(led, founder),
        issuer: _cash(led, issuer),
        bob: _cash(led, bob),
        carol: _cash(led, carol),
        "BANKSYS": _cash(led, "BANKSYS"),
    }

    result = list_firm(
        table,
        led,
        clob,
        cfg,
        kind="primary",
        shares=offered,
        reserve=reserve,
        bids=(
            _buy(bob, offered, 9.00, symbol),
            _buy(carol, offered, 9.99, symbol),
        ),
        tick=1,
    )

    assert result.price is None
    assert result.volume == 0
    assert result.fills == ()
    assert table.shares_outstanding(led) == pytest.approx(float(DEFAULT_FOUNDER_SHARES))
    assert table.holding(led, bob) == pytest.approx(0.0)
    assert table.holding(led, carol) == pytest.approx(0.0)
    assert table.holding(led, founder) == pytest.approx(float(DEFAULT_FOUNDER_SHARES))
    for name, amt in before.items():
        assert _cash(led, name) == pytest.approx(amt)
    assert table.invariant_holds(led)
    assert_consistent(led)


def test_49_percent_warning(cfg: Config) -> None:
    # 10 % of pre-issue SO → issued / (SO + issued) ≈ 0.0909, below 0.49 (§6.7).
    ten_pct = DEFAULT_FOUNDER_SHARES // 10
    # 50 % of post-issue SO → issued / (SO + issued) = 0.50 > 0.49.
    over_49 = DEFAULT_FOUNDER_SHARES
    reserve = 10.00
    bob = agent_entity("bob")
    carol = agent_entity("carol")

    led_small = _led(bob, cash={bob: ten_pct * reserve})
    small = _open(led_small, firm_id="smallco", founder=agent_entity("alice"))
    r_small = list_firm(
        small,
        led_small,
        _clob(cfg),
        cfg,
        kind="primary",
        shares=ten_pct,
        reserve=reserve,
        bids=(_buy(bob, ten_pct, reserve, firm_symbol(small.firm_id)),),
        tick=1,
    )
    assert r_small.warning_49 is False
    assert r_small.volume == ten_pct
    assert small.invariant_holds(led_small)
    assert_consistent(led_small)

    led_big = _led(carol, cash={carol: over_49 * reserve})
    big = _open(led_big, firm_id="bigco", founder=agent_entity("dave"))
    r_big = list_firm(
        big,
        led_big,
        _clob(cfg),
        cfg,
        kind="primary",
        shares=over_49,
        reserve=reserve,
        bids=(_buy(carol, over_49, reserve, firm_symbol(big.firm_id)),),
        tick=1,
    )
    assert r_big.warning_49 is True
    assert r_big.volume == over_49
    post = float(DEFAULT_FOUNDER_SHARES + over_49)
    assert over_49 / post > 0.49
    assert big.ownership(led_big, big.founder) == pytest.approx(0.50)
    assert big.invariant_holds(led_big)
    assert_consistent(led_big)
