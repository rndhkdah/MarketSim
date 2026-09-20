"""T5.12 — waterfall, loss cap, SFC, workers, large-bankruptcy event."""

from __future__ import annotations

import pytest

from marketsim.events.firm_hooks import FirmHookBus
from marketsim.firms.accounts import register_firm
from marketsim.firms.bankruptcy import Claim, resolve, should_resolve
from marketsim.firms.firm import Firm, FirmsFile, Plant
from marketsim.firms.founding import CellBook, NpcCell, npc_entity
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def _led() -> Ledger:
    led = Ledger.empty()
    for n in ("HH:0", "BANKSYS", "GOVT"):
        led.register_entity(n)
    led.post(Tx(0, "opening", (Entry("HH:0", "DEP", 200.0), Entry("BANKSYS", "DEP", -200.0))))
    return led


def test_waterfall_by_hand() -> None:
    led = _led()
    register_firm(led, "acme")
    firm = Firm(
        id="acme",
        operator="a",
        plants=(Plant("INDUSTRIAL", "AUTOS", 10.0),),
        employees=4.0,
        cash=0.0,
        credit_drawn=30.0,
        equity_negative_months=3,
    )
    book = CellBook({("INDUSTRIAL", "AUTOS"): NpcCell("INDUSTRIAL", "AUTOS", 5.0, 50.0)})
    # unit replacement 10; plant recovery 0.70 → 70
    seller = npc_entity("INDUSTRIAL", "AUTOS")
    led.register_entity(seller)
    hooks = FirmHookBus()
    claims = [
        Claim("wages", "HH:0", 10.0),
        Claim("bank", "BANKSYS", 40.0),
        Claim("suppliers", "HH:0", 5.0),
    ]
    assert should_resolve(firm, obligations_due=100.0)
    res = resolve(
        firm, book, led, FirmsFile(), claims=claims, monthly_gdp=10.0, labour_pool={"INDUSTRIAL": 1.0}, hooks=hooks, tick=1
    )
    assert res.steps == 1
    assert res.recoveries["wages"] == pytest.approx(10.0)
    assert res.recoveries["bank"] + res.losses["bank"] == pytest.approx(40.0)
    assert res.losses["bank"] <= 40.0
    assert firm.employees == 0.0
    assert firm.status == "liquidated"
    assert res.event_emitted  # 55 > 0.01*10
    assert hooks.calls[0].event_id == "large_bankruptcy"
    assert_consistent(led)


def test_small_default_no_event() -> None:
    led = _led()
    register_firm(led, "tiny")
    firm = Firm(id="tiny", operator="a", plants=(Plant("INDUSTRIAL", "AUTOS", 0.1),), equity_negative_months=3)
    book = CellBook({("INDUSTRIAL", "AUTOS"): NpcCell("INDUSTRIAL", "AUTOS", 1.0, 1.0)})
    led.register_entity(npc_entity("INDUSTRIAL", "AUTOS"))
    res = resolve(
        firm,
        book,
        led,
        FirmsFile(),
        claims=[Claim("bank", "BANKSYS", 0.01)],
        monthly_gdp=1000.0,
        labour_pool={},
        tick=1,
    )
    assert res.event_emitted is False
    assert_consistent(led)
