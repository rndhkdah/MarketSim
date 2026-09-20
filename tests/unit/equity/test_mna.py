"""T9.04 — tender, dual-class, poison pill, merger; all SFC-green."""

from __future__ import annotations

import pytest

from marketsim.equity.captable import CapTable
from marketsim.equity.control import CONTROL_THRESHOLD
from marketsim.equity.mna import (
    PILL_TRIGGER,
    Defence,
    DualClassBook,
    TenderOffer,
    apply_tender,
    apply_tender_with_defence,
    consideration,
    control_delay_m,
    crossing_control,
    merge_firms,
    would_trip_pill,
)
from marketsim.firms.accounts import agent_entity
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def _open_two(founder_shares: float = 100.0, cash: float = 600.0) -> tuple[Ledger, CapTable, str, str]:
    led = Ledger.empty()
    led.register_entity("BANKSYS")
    alice = agent_entity("alice")
    bob = agent_entity("bob")
    led.register_entity(alice)
    led.register_entity(bob)
    led.post(Tx(0, "opening", (Entry(bob, "DEP", cash), Entry("BANKSYS", "DEP", -cash))))
    table = CapTable.open(led, firm_id="acme", founder=alice, shares=founder_shares)
    return led, table, alice, bob


def test_tender_cash_and_shares() -> None:
    led, table, alice, bob = _open_two()
    offer = TenderOffer(
        firm_id="acme",
        bidder="bob",
        target_holder="alice",
        shares=60.0,
        price=10.0,
    )
    assert consideration(60.0, 10.0) == pytest.approx(600.0)
    apply_tender(led, table, offer, tick=1)
    assert table.holding(led, bob) == pytest.approx(60.0)
    assert table.holding(led, alice) == pytest.approx(40.0)
    assert led.position(bob, "DEP") == pytest.approx(0.0)
    assert led.position(alice, "DEP") == pytest.approx(600.0)
    assert table.invariant_holds(led)
    assert_consistent(led)
    assert crossing_control(table.ownership(led, bob)) is True
    assert CONTROL_THRESHOLD == 0.50


def test_nonvoting_does_not_confer_control() -> None:
    book = DualClassBook()
    book.grant("alice", 51.0, "voting")
    book.grant("bob", 49.0, "voting")
    book.grant("bob", 200.0, "nonvoting")
    assert book.economic_ownership("bob") > 0.80
    assert book.voting_ownership("bob") == pytest.approx(0.49)
    assert crossing_control(book.voting_ownership("bob")) is False
    assert crossing_control(book.voting_ownership("alice")) is True
    restored = DualClassBook.from_state(book.to_state())
    assert restored.to_state() == book.to_state()


def test_poison_pill_dilutes_bidder() -> None:
    led, table, alice, bob = _open_two(founder_shares=100.0, cash=1_000.0)
    defence = Defence(poison_pill=True, pill_trigger=PILL_TRIGGER, pill_ratio=1.0)
    offer = TenderOffer(firm_id="acme", bidder="bob", target_holder="alice", shares=20.0, price=10.0)
    assert would_trip_pill(0.0, 0.20, defence) is True
    apply_tender_with_defence(led, table, offer, defence, tick=1)
    # Pill issues 100 new shares to alice *before* the 20-share tender, then
    # 20 move to bob → alice 180, bob 20, outstanding 200. Bidder at 10 % < 15 %.
    assert table.shares_outstanding(led) == pytest.approx(200.0)
    assert table.holding(led, alice) == pytest.approx(180.0)
    assert table.holding(led, bob) == pytest.approx(20.0)
    assert table.ownership(led, bob) == pytest.approx(0.10)
    assert crossing_control(table.ownership(led, bob)) is False
    assert table.invariant_holds(led)
    assert_consistent(led)


def test_staggered_board_delays_control_only() -> None:
    assert control_delay_m(Defence()) == 0
    assert control_delay_m(Defence(staggered_board=True)) == 1


def test_merger_exchanges_shares_sfc() -> None:
    led = Ledger.empty()
    led.register_entity("BANKSYS")
    alice = agent_entity("alice")
    carol = agent_entity("carol")
    led.register_entity(alice)
    led.register_entity(carol)
    survivor = CapTable.open(led, firm_id="big", founder=alice, shares=100.0)
    target = CapTable.open(led, firm_id="small", founder=carol, shares=50.0)
    issued = merge_firms(led, survivor, target, exchange_ratio=2.0, tick=1)
    assert issued[carol] == pytest.approx(100.0)
    assert survivor.holding(led, alice) == pytest.approx(100.0)
    assert survivor.holding(led, carol) == pytest.approx(100.0)
    assert survivor.shares_outstanding(led) == pytest.approx(200.0)
    assert target.shares_outstanding(led) == pytest.approx(0.0)
    assert target.holding(led, carol) == pytest.approx(0.0)
    assert survivor.invariant_holds(led)
    assert target.invariant_holds(led)
    assert_consistent(led)
