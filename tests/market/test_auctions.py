"""T6.26 — uniform-price auctions, DMO split, gate 12."""

from __future__ import annotations

import math

import pytest

from marketsim.core.clock import EventQueue
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.opening import GOVT_MIX
from marketsim.market.auction import (
    ANNOUNCE_LEAD_TICKS,
    AUCTION_DAY,
    ETA_A,
    Bid,
    announce_tick,
    auction_tick,
    buyback,
    clear_uniform,
    npc_schedule,
    settle_auction,
)
from marketsim.real.policy.debt_management import (
    autopilot_calendar,
    financing_need,
    schedule_month,
    split_issuance,
)


def test_hand_computed_npc_only_clearing() -> None:
    y_fair, q0, size = 0.042, 100.0, 150.0
    y = y_fair + math.log(size / q0) / (ETA_A * 100.0)
    result = clear_uniform(size, (), y_fair=y_fair, q0=q0)
    assert result.stop_out == pytest.approx(y)
    assert result.npc_fill == pytest.approx(size)
    assert result.filled == pytest.approx(size)
    assert result.agent_fill == {}
    assert npc_schedule(result.stop_out, y_fair, q0) == pytest.approx(size)


def test_losing_bids_pay_nothing() -> None:
    bids = (
        Bid("alice", 0.045, 8.0),  # wins
        Bid("bob", 0.080, 10.0),  # asks too much yield; loses
    )
    result = clear_uniform(10.0, bids, y_fair=0.042, q0=5.0)
    assert "alice" in result.agent_fill
    assert "bob" not in result.agent_fill
    assert result.agent_fill["alice"] == pytest.approx(8.0)


def test_announcement_precedes_auction_by_5_ticks() -> None:
    cal = autopilot_calendar(0)
    assert cal["lead"] == ANNOUNCE_LEAD_TICKS == 5
    assert cal["auction"] - cal["announce"] == 5
    assert auction_tick(0) == AUCTION_DAY - 1
    assert announce_tick(2) == auction_tick(2) - 5


def test_gate12_calm_cover_and_tail() -> None:
    y_fair = 0.042
    result = clear_uniform(99.0, (), y_fair=y_fair, q0=100.0, secondary_yield=y_fair)
    assert result.bid_to_cover > 1.0
    assert abs(result.tail) < 0.0005  # 5 bp


def test_doubling_size_raises_stop_out() -> None:
    y_fair, q0 = 0.042, 80.0
    small = clear_uniform(40.0, (), y_fair=y_fair, q0=q0)
    big = clear_uniform(80.0, (), y_fair=y_fair, q0=q0)
    assert big.stop_out > small.stop_out


def test_winning_bids_settle_and_govt_dep_nonneg() -> None:
    led = Ledger.empty()
    for name in ("GOVT", "alice", "MM", "BANKSYS"):
        led.register_entity(name)
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry("alice", "DEP", 50.0),
                Entry("MM", "DEP", 200.0),
                Entry("BANKSYS", "DEP", -250.0),
            ),
        )
    )
    result = clear_uniform(30.0, (Bid("alice", 0.044, 10.0),), y_fair=0.042, q0=20.0)
    settle_auction(led, result, instrument="GB_NOTE", tick=1)
    assert led.position("alice", "GB_NOTE") == pytest.approx(result.agent_fill["alice"])
    assert led.position("MM", "GB_NOTE") == pytest.approx(result.npc_fill)
    assert led.position("GOVT", "GB_NOTE") == pytest.approx(-result.size)
    assert led.position("GOVT", "DEP") == pytest.approx(result.size)
    assert led.position("GOVT", "DEP") >= 0.0
    assert led.position("alice", "GB_NOTE") + led.position("MM", "GB_NOTE") + led.position(
        "GOVT", "GB_NOTE"
    ) == pytest.approx(0.0)


def test_buyback_retires_face_and_keeps_sfc() -> None:
    led = Ledger.empty()
    for name in ("GOVT", "MM", "BANKSYS"):
        led.register_entity(name)
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry("MM", "GB_BOND", 20.0),
                Entry("GOVT", "GB_BOND", -20.0),
                Entry("GOVT", "DEP", 20.0),
                Entry("BANKSYS", "DEP", -20.0),
            ),
        )
    )
    buyback(led, instrument="GB_BOND", face=5.0, tick=1, holder="MM")
    assert led.position("MM", "GB_BOND") == pytest.approx(15.0)
    assert led.position("GOVT", "GB_BOND") == pytest.approx(-15.0)
    assert led.position("GOVT", "DEP") == pytest.approx(15.0)
    assert led.position("GOVT", "DEP") >= 0.0
    assert led.position("MM", "GB_BOND") + led.position("GOVT", "GB_BOND") == pytest.approx(0.0)


def test_schedule_month_announces_five_ticks_ahead() -> None:
    q = EventQueue()
    schedule_month(q, 0, sizes={"GB_NOTE": 10.0})
    due_announce = q.pop_due(announce_tick(0))
    assert due_announce and due_announce[0]["kind"] == "auction_announce"
    assert due_announce[0]["sizes"]["GB_NOTE"] == pytest.approx(10.0)
    assert q.pop_due(auction_tick(0) - 1) == []
    due_auction = q.pop_due(auction_tick(0))
    assert due_auction and due_auction[0]["kind"] == "auction"


def test_financing_need_and_204040_split() -> None:
    need = financing_need(deficit=10.0, redemptions=4.0, deposit_buffer_topup=1.0)
    assert need == pytest.approx(15.0)
    sizes = split_issuance(need)
    assert sizes["GB_BILL"] == pytest.approx(3.0)
    assert sizes["GB_NOTE"] == pytest.approx(6.0)
    assert sizes["GB_BOND"] == pytest.approx(6.0)
    assert tuple(sizes) == tuple(n for n, _ in GOVT_MIX)
