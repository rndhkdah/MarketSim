"""T5.02 — firm ledger accounts, founding capital, gate-6 books."""

from __future__ import annotations

import pytest

from marketsim.firms.accounts import (
    BalanceSheet,
    FirmBooks,
    agent_entity,
    firm_entity,
    firm_equity_instrument,
    post_founding_capital,
    register_firm,
)
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def _world() -> Ledger:
    led = Ledger.empty()
    for name in ("HH:0", "BANKSYS", "GOVT"):
        led.register_entity(name)
    led.post(
        Tx(
            0,
            "opening",
            (Entry("HH:0", "DEP", 400.0), Entry("BANKSYS", "DEP", -400.0)),
        )
    )
    return led


def test_founding_equity_issue_sfc() -> None:
    led = _world()
    register_firm(led, "acme", operator="alice")
    led.post(
        Tx(
            1,
            "capital_transfer",
            (Entry("HH:0", "DEP", -80.0), Entry(agent_entity("alice"), "DEP", 80.0)),
        )
    )
    post_founding_capital(led, firm_id="acme", amount=80.0, source=agent_entity("alice"), tick=1)
    assert_consistent(led)
    assert led.position(firm_entity("acme"), "DEP") == pytest.approx(80.0)
    assert led.position(firm_entity("acme"), firm_equity_instrument("acme")) == pytest.approx(-80.0)
    assert led.position(agent_entity("alice"), firm_equity_instrument("acme")) == pytest.approx(80.0)


def test_founding_capital_transfer_from_hh() -> None:
    led = _world()
    register_firm(led, "acme")
    post_founding_capital(
        led, firm_id="acme", amount=50.0, source="HH:0", tick=1, tag="capital_transfer"
    )
    assert_consistent(led)
    books = FirmBooks("acme")
    sheet = books.balance_sheet(led)
    assert sheet.identity_holds()
    assert sheet.cash == pytest.approx(50.0)
    assert sheet.equity == pytest.approx(50.0)


def test_gate6_scripted_24_month_firm() -> None:
    led = _world()
    register_firm(led, "acme")
    post_founding_capital(led, firm_id="acme", amount=100.0, source="HH:0", tick=0, tag="capital_transfer")
    books = FirmBooks("acme")
    open_stmt = books.close_month(led, owner="HH:0", tick=0)
    assert open_stmt.capital == pytest.approx(100.0)
    assert open_stmt.net_income == pytest.approx(0.0)
    firm = firm_entity("acme")
    for month in range(1, 25):
        led.post(Tx(month, "consumption", (Entry("HH:0", "DEP", -10.0), Entry(firm, "DEP", 10.0))))
        led.post(Tx(month, "wages", (Entry(firm, "DEP", -4.0), Entry("HH:0", "DEP", 4.0))))
        stmt = books.close_month(led, owner="HH:0", tick=month)
        assert stmt.revenue == pytest.approx(10.0)
        assert stmt.wages == pytest.approx(-4.0)
        assert stmt.net_income == pytest.approx(6.0)
        assert books.sheets[-1].identity_holds()
        assert_consistent(led)

    sheet = books.sheets[-1]
    assert sheet.cash == pytest.approx(100.0 + 24 * 6.0)
    assert sheet.equity == pytest.approx(100.0 + 24 * 6.0)
    assert sheet.identity_holds()


def test_balance_sheet_identity_helper() -> None:
    sheet = BalanceSheet(cash=10.0, real_assets=5.0, debt=3.0, equity=12.0)
    assert sheet.identity_holds()
    sheet.equity = 11.0
    assert not sheet.identity_holds()
