"""T5.13 — lagged books and professional blackout."""

from __future__ import annotations

from marketsim.firms.accounts import FirmBooks
from marketsim.firms.firm import FirmsFile
from marketsim.firms.reports import ReportDesk


def test_non_operator_never_sees_live() -> None:
    desk = ReportDesk(FirmsFile(), "professional")
    books = FirmBooks("acme")
    desk.publish_month("acme", books, tick=5)
    live = desk.observe("acme", viewer="alice", operator="alice", tick=5)
    assert live["live"] is True
    other = desk.observe("acme", viewer="bob", operator="alice", tick=5)
    assert other["live"] is False
    assert other["books"] is None  # 10-day lag
    later = desk.observe("acme", viewer="bob", operator="alice", tick=5 + 10)
    assert later["live"] is False
    assert later["books"] is not None


def test_blackout_professional_only() -> None:
    desk = ReportDesk(FirmsFile(), "professional")
    desk.quarter_end_tick["acme"] = 62
    assert desk.can_trade_own_shares("acme", tick=70) is False
    assert desk.can_trade_own_shares("acme", tick=62 + 21) is True
    game = ReportDesk(FirmsFile(), "game")
    game.quarter_end_tick["acme"] = 62
    assert game.can_trade_own_shares("acme", tick=63) is True
