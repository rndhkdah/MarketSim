"""T6.14 — cap tables and corporate actions (§6.7)."""

from __future__ import annotations

import pytest

from marketsim.equity.captable import DEFAULT_FOUNDER_SHARES, CapTable
from marketsim.equity.corporate_actions import (
    RECORD_LAG_TICKS,
    CorporateActions,
    buyback_shares,
    issue_shares,
)
from marketsim.firms.accounts import agent_entity, firm_entity, firm_equity_instrument
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


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
    shares: float | None = None,
) -> tuple[CapTable, str, str, str]:
    founder = founder or agent_entity("alice")
    kwargs: dict[str, float] = {}
    if shares is not None:
        kwargs["shares"] = shares
    table = CapTable.open(led, firm_id=firm_id, founder=founder, **kwargs)
    return table, founder, table.issuer, table.symbol


def _assert_register(table: CapTable, ledger: Ledger) -> None:
    assert table.invariant_holds(ledger)
    assert table.total_holdings(ledger) == pytest.approx(table.shares_outstanding(ledger))
    assert ledger.position(table.issuer, table.symbol) == pytest.approx(-table.shares_outstanding(ledger))
    inst = sum(ledger.position(name, table.symbol) for name in ledger.entities.names)
    assert inst == pytest.approx(0.0)
    assert_consistent(ledger)


def test_founder_default_one_million_shares() -> None:
    assert DEFAULT_FOUNDER_SHARES == 1_000_000
    led = _led()
    table, founder, issuer, symbol = _open(led)
    assert table.shares_outstanding(led) == pytest.approx(1_000_000.0)
    assert table.holding(led, founder) == pytest.approx(1_000_000.0)
    assert table.ownership(led, founder) == pytest.approx(1.0)
    assert table.total_holdings(led) == pytest.approx(table.shares_outstanding(led))
    assert led.position(issuer, symbol) == pytest.approx(-1_000_000.0)
    assert firm_entity("acme") == issuer
    assert firm_equity_instrument("acme") == symbol
    _assert_register(table, led)


def test_holdings_sum_equals_shares_outstanding_always() -> None:
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    led = _led(bob, cash={bob: 50_000.0})
    table, founder, issuer, _ = _open(led)
    _assert_register(table, led)

    table.transfer(led, src=founder, dst=bob, shares=250_000.0, tick=1)
    assert table.shares_outstanding(led) == pytest.approx(1_000_000.0)
    _assert_register(table, led)

    issue_shares(table, led, subscriber=bob, shares=250_000.0, price=0.2, tick=2)
    assert table.shares_outstanding(led) == pytest.approx(1_250_000.0)
    assert table.total_holdings(led) == pytest.approx(1_250_000.0)
    _assert_register(table, led)

    led.post(
        Tx(
            3,
            "opening",
            (Entry(issuer, "DEP", 10_000.0), Entry("BANKSYS", "DEP", -10_000.0)),
        )
    )
    buyback_shares(table, led, seller=bob, shares=50_000.0, price=0.2, tick=3)
    assert table.shares_outstanding(led) == pytest.approx(1_200_000.0)
    assert table.total_holdings(led) == pytest.approx(1_200_000.0)
    _assert_register(table, led)


def test_dividends_per_record_date() -> None:
    assert RECORD_LAG_TICKS == 2
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    firm = firm_entity("acme")
    led = _led(bob)
    table, founder, _, _ = _open(led)
    led.post(Tx(0, "opening", (Entry(firm, "DEP", 100.0), Entry("BANKSYS", "DEP", -100.0))))

    desk = CorporateActions()
    declared = desk.declare_dividend(table, amount=10.0, tick=0)
    assert declared.record_tick == 0 + RECORD_LAG_TICKS
    assert declared.record_tick == 2

    assert desk.settle(table, led, tick=0) == ()
    assert desk.settle(table, led, tick=1) == ()
    assert led.position(founder, "DEP") == pytest.approx(0.0)
    assert led.position(bob, "DEP") == pytest.approx(0.0)
    assert led.position(firm, "DEP") == pytest.approx(100.0)

    # Sold after declaration, before record date → buyer is on the register.
    table.transfer(led, src=founder, dst=bob, shares=400_000.0, tick=1)
    txs = desk.settle(table, led, tick=2)
    assert len(txs) == 1
    assert txs[0].tag == "dividends"
    assert led.position(founder, "DEP") == pytest.approx(6.0)
    assert led.position(bob, "DEP") == pytest.approx(4.0)
    assert led.position(firm, "DEP") == pytest.approx(90.0)
    _assert_register(table, led)

    # After the record date a later buyer does not collect this dividend.
    carol = agent_entity("carol")
    led.register_entity(carol)
    table.transfer(led, src=bob, dst=carol, shares=400_000.0, tick=3)
    assert desk.settle(table, led, tick=3) == ()
    assert led.position(carol, "DEP") == pytest.approx(0.0)
    assert led.position(bob, "DEP") == pytest.approx(4.0)
    _assert_register(table, led)


def test_buyback_cancels_shares() -> None:
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    led = _led(bob)
    table, founder, issuer, symbol = _open(led)
    table.transfer(led, src=founder, dst=bob, shares=200_000.0, tick=1)
    led.post(
        Tx(
            1,
            "opening",
            (Entry(issuer, "DEP", 400_000.0), Entry("BANKSYS", "DEP", -400_000.0)),
        )
    )

    tx = buyback_shares(table, led, seller=bob, shares=200_000.0, price=2.0, tick=2)
    assert tx.tag == "equity_trade"
    assert table.shares_outstanding(led) == pytest.approx(800_000.0)
    assert table.holding(led, bob) == pytest.approx(0.0)
    assert table.holding(led, founder) == pytest.approx(800_000.0)
    assert table.ownership(led, founder) == pytest.approx(1.0)
    assert led.position(issuer, symbol) == pytest.approx(-800_000.0)
    assert led.position(issuer, "DEP") == pytest.approx(0.0)
    assert led.position(bob, "DEP") == pytest.approx(400_000.0)
    _assert_register(table, led)


def test_dilution_arithmetic() -> None:
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    old = float(DEFAULT_FOUNDER_SHARES)
    issued = 500_000.0
    led = _led(bob, cash={bob: issued * 4.0})
    table, founder, issuer, symbol = _open(led)
    assert table.ownership(led, founder) == pytest.approx(1.0)

    tx = issue_shares(table, led, subscriber=bob, shares=issued, price=4.0, tick=1)
    assert tx.tag == "equity_issue"
    new = old + issued
    assert table.shares_outstanding(led) == pytest.approx(new)
    assert table.holding(led, founder) == pytest.approx(old)
    assert table.holding(led, bob) == pytest.approx(issued)
    assert table.ownership(led, founder) == pytest.approx(old / new)
    assert table.ownership(led, bob) == pytest.approx(issued / new)
    assert table.ownership(led, founder) + table.ownership(led, bob) == pytest.approx(1.0)
    assert led.position(issuer, "DEP") == pytest.approx(issued * 4.0)
    assert led.position(bob, "DEP") == pytest.approx(0.0)
    assert led.position(issuer, symbol) == pytest.approx(-new)
    _assert_register(table, led)


def test_issuer_carries_negative_eq_position() -> None:
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    led = _led(bob, cash={bob: 10.0})
    table, founder, issuer, symbol = _open(led)
    assert led.position(issuer, symbol) == pytest.approx(-DEFAULT_FOUNDER_SHARES)
    assert led.position(founder, symbol) == pytest.approx(DEFAULT_FOUNDER_SHARES)
    assert sum(led.position(n, symbol) for n in led.entities.names) == pytest.approx(0.0)

    table.transfer(led, src=founder, dst=bob, shares=100_000.0, tick=1, price=0.0)
    issue_shares(table, led, subscriber=bob, shares=10.0, price=1.0, tick=2)
    assert led.position(issuer, symbol) == pytest.approx(-table.shares_outstanding(led))
    assert table.total_holdings(led) == pytest.approx(-led.position(issuer, symbol))
    assert sum(led.position(n, symbol) for n in led.entities.names) == pytest.approx(0.0)
    _assert_register(table, led)


def test_captable_state_roundtrip() -> None:
    led = _led()
    table, founder, _, _ = _open(led)
    table.transfer(led, src=founder, dst=agent_entity("bob"), shares=1.0, tick=4)
    restored = CapTable.from_state(table.to_state())
    assert restored.firm_id == table.firm_id
    assert restored.founder == table.founder
    assert restored.holdings_on(4, led) == table.holdings_on(4, led)
    desk = CorporateActions()
    desk.declare_dividend(table, amount=1.0, tick=5)
    desk2 = CorporateActions.from_state(desk.to_state())
    assert desk2.declarations[0].record_tick == 5 + RECORD_LAG_TICKS
    _assert_register(table, led)
