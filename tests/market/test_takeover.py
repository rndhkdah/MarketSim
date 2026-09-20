"""T6.16 — control transfer and the takeover test (§6.7 / gate 5)."""

from __future__ import annotations

import math
import re

import pytest

from marketsim.core.calendar import Calendar
from marketsim.core.config import Config
from marketsim.core.errors import ConfigError
from marketsim.equity.captable import DEFAULT_FOUNDER_SHARES, CapTable
from marketsim.equity.control import (
    CONTROL_THRESHOLD,
    DISCLOSURE_THRESHOLD,
    ControlDesk,
    disclosures,
    next_month_end,
)
from marketsim.firms.accounts import agent_entity
from marketsim.firms.firm import Firm
from marketsim.firms.levers import FirmDecision, validate_decision
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent, net_financial_assets


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


def _firm(firm_id: str = "acme", operator: str = "alice") -> Firm:
    return Firm(id=firm_id, operator=operator)


def _decide(firm: Firm, who: str, cfg) -> None:
    """Submit ``who``'s decision; authority is the live ``firm.operator``."""
    validate_decision(
        FirmDecision(firm.id, who, dividend=0.0),
        firm,
        cfg,
        controller=firm.operator,
    )


def _marked_nw(ledger: Ledger, entity: str, symbol: str, price: float) -> float:
    """Cash (cr) + long shares marked at ``price`` (cr/share)."""
    cash = ledger.position(entity, "DEP") if entity in ledger.entities else 0.0
    shares = ledger.position(entity, symbol) if entity in ledger.entities else 0.0
    if shares < 0.0:
        shares = 0.0
    return cash + shares * price


def _block() -> int:
    """Shares that just cross the §6.7 control line: 50 % + 1 share."""
    return DEFAULT_FOUNDER_SHARES // 2 + 1


def test_gate5_control_passes_at_next_month_boundary(cfg: Config) -> None:
    assert cfg.firms is not None
    cal = Calendar()
    desk = ControlDesk(calendar=cal)
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    price = 1.0  # cr/share; fees 0
    qty = _block()
    led = _led(bob, cash={bob: qty * price})
    table = CapTable.open(led, firm_id="acme", founder=founder)
    firm = _firm()
    assert_consistent(led)

    mid = 7
    assert not cal.is_month_end(mid)
    month_end = next_month_end(cal, mid)
    assert cal.is_month_end(month_end)

    nw0 = _marked_nw(led, founder, table.symbol, price) + _marked_nw(led, bob, table.symbol, price)
    nfa0 = net_financial_assets(led).copy()

    desk.transfer(table, led, firm, src=founder, dst=bob, shares=qty, tick=mid, price=price)
    assert table.ownership(led, bob) > CONTROL_THRESHOLD
    assert firm.operator == "alice"
    assert desk.pending_operator(firm.id) == "bob"
    with pytest.raises(ConfigError):
        _decide(firm, "bob", cfg.firms)
    _decide(firm, firm.operator, cfg.firms)

    nw1 = _marked_nw(led, founder, table.symbol, price) + _marked_nw(led, bob, table.symbol, price)
    assert nw1 == pytest.approx(nw0)
    assert net_financial_assets(led).sum() == pytest.approx(nfa0.sum())
    assert "fees" not in led._tags
    assert "transaction_tax" not in led._tags
    assert_consistent(led)

    news = desk.apply_month_boundary(firm, month_end)
    assert firm.operator == "bob"
    assert desk.pending_operator(firm.id) is None
    assert news
    _decide(firm, firm.operator, cfg.firms)
    with pytest.raises(ConfigError):
        _decide(firm, "alice", cfg.firms)
    # Previous operator keeps the residual shares and any later dividends.
    assert table.holding(led, founder) == pytest.approx(DEFAULT_FOUNDER_SHARES - qty)
    assert table.holding(led, bob) == pytest.approx(qty)
    nw2 = _marked_nw(led, founder, table.symbol, price) + _marked_nw(led, bob, table.symbol, price)
    assert nw2 == pytest.approx(nw0)
    assert_consistent(led)


def test_lever_authority_switches_exactly_at_month_boundary(cfg: Config) -> None:
    assert cfg.firms is not None
    cal = Calendar()
    desk = ControlDesk(calendar=cal)
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    qty = _block()
    led = _led(bob)
    table = CapTable.open(led, firm_id="acme", founder=founder)
    firm = _firm()

    mid = 7
    month_end = next_month_end(cal, mid)
    desk.transfer(table, led, firm, src=founder, dst=bob, shares=qty, tick=mid)
    assert desk.pending_operator(firm.id) == "bob"
    assert next_month_end(cal, month_end) > month_end

    assert desk.apply_month_boundary(firm, mid) == []
    assert firm.operator == "alice"
    assert desk.apply_month_boundary(firm, month_end - 1) == []
    assert firm.operator == "alice"
    assert not cal.is_month_end(month_end - 1)
    _decide(firm, "alice", cfg.firms)
    with pytest.raises(ConfigError):
        _decide(firm, "bob", cfg.firms)

    desk.apply_month_boundary(firm, month_end)
    assert cal.is_month_end(month_end)
    assert firm.operator == "bob"
    _decide(firm, firm.operator, cfg.firms)
    with pytest.raises(ConfigError):
        _decide(firm, "alice", cfg.firms)

    # Crossing on a month-end waits for the *following* month-end (documented).
    other = _firm("beta", "alice")
    table2 = CapTable.open(led, firm_id="beta", founder=founder)
    desk.transfer(table2, led, other, src=founder, dst=bob, shares=qty, tick=month_end)
    assert desk.pending_operator(other.id) == "bob"
    assert desk.apply_month_boundary(other, month_end) == []
    assert other.operator == "alice"
    later = next_month_end(cal, month_end)
    desk.apply_month_boundary(other, later)
    assert other.operator == "bob"


def test_professional_mode_discloses_five_percent(cfg: Config) -> None:
    del cfg
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    carol = agent_entity("carol")
    led = _led(bob, carol)
    table = CapTable.open(led, firm_id="acme", founder=founder)
    five = math.ceil(DISCLOSURE_THRESHOLD * DEFAULT_FOUNDER_SHARES)
    table.transfer(led, src=founder, dst=bob, shares=float(five), tick=1)
    table.transfer(led, src=founder, dst=carol, shares=float(five - 1), tick=1)
    assert table.ownership(led, bob) >= DISCLOSURE_THRESHOLD
    assert table.ownership(led, carol) < DISCLOSURE_THRESHOLD

    pro = disclosures(table, led, mode="professional")
    assert pro[bob] == pytest.approx(table.ownership(led, bob))
    assert pro[founder] == pytest.approx(table.ownership(led, founder))
    assert carol not in pro
    assert disclosures(table, led, mode="game") == {}


def test_news_emitted_on_control_change() -> None:
    cal = Calendar()
    desk = ControlDesk(calendar=cal)
    founder = agent_entity("alice")
    bob = agent_entity("bob")
    led = _led(bob)
    table = CapTable.open(led, firm_id="acme", founder=founder)
    firm = _firm()
    mid = 7
    desk.transfer(table, led, firm, src=founder, dst=bob, shares=_block(), tick=mid)
    assert desk.news == []
    month_end = next_month_end(cal, mid)
    items = desk.apply_month_boundary(firm, month_end)
    assert len(items) == 1
    item = items[0]
    assert item is desk.news[0]
    assert item.tick == month_end
    assert item.category == "corporate"
    assert item.is_rumour is False
    assert re.search(r"\d", item.headline) is None
    assert "%" not in item.headline
    restored = ControlDesk.from_state(desk.to_state())
    assert restored.pending_operator(firm.id) is None
    assert len(restored.news) == 1
    assert restored.news[0].headline == item.headline
    assert restored.news[0].category == "corporate"
