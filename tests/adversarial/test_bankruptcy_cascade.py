"""T5.17 gate 4 — large bankruptcy is one step and emits an event."""

from __future__ import annotations

from marketsim.events.firm_hooks import FirmHookBus
from marketsim.firms.accounts import register_firm
from marketsim.firms.bankruptcy import Claim, resolve
from marketsim.firms.firm import Firm, FirmsFile, Plant
from marketsim.firms.founding import CellBook, NpcCell, npc_entity
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def test_large_bankruptcy_one_step_event() -> None:
    led = Ledger.empty()
    for n in ("HH:0", "BANKSYS", "GOVT"):
        led.register_entity(n)
    led.post(Tx(0, "opening", (Entry("HH:0", "DEP", 50.0), Entry("BANKSYS", "DEP", -50.0))))
    register_firm(led, "boom")
    firm = Firm(id="boom", operator="x", plants=(Plant("INDUSTRIAL", "AUTOS", 20.0),), employees=3.0)
    book = CellBook({("INDUSTRIAL", "AUTOS"): NpcCell("INDUSTRIAL", "AUTOS", 1.0, 20.0)})
    led.register_entity(npc_entity("INDUSTRIAL", "AUTOS"))
    hooks = FirmHookBus()
    res = resolve(
        firm,
        book,
        led,
        FirmsFile(),
        claims=[Claim("bank", "BANKSYS", 80.0)],
        monthly_gdp=5.0,
        labour_pool={},
        hooks=hooks,
        tick=1,
    )
    assert res.steps == 1
    assert res.event_emitted
    assert_consistent(led)
