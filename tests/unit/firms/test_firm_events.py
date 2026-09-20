"""T5.15 — plant accident, strike, recall through firm hooks + ledger."""

from __future__ import annotations

import pytest

from marketsim.events.firm_hooks import FirmHookPayload, apply_firm_hook
from marketsim.events.schema import load_catalog
from marketsim.firms.accounts import firm_entity, register_firm
from marketsim.firms.firm import Firm, FirmRegistry, Plant, cells_from_lists
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def test_catalog_has_firm_templates() -> None:
    cat = load_catalog("config/events")
    assert "firm_plant_accident" in cat
    assert "firm_strike" in cat
    assert "firm_recall" in cat


def test_accident_strike_recall() -> None:
    led = Ledger.empty()
    led.register_entity("HH:0")
    led.register_entity("BANKSYS")
    led.post(Tx(0, "opening", (Entry("HH:0", "DEP", 50.0), Entry("BANKSYS", "DEP", -50.0))))
    register_firm(led, "acme")
    led.post(Tx(1, "capital_transfer", (Entry("HH:0", "DEP", -20.0), Entry(firm_entity("acme"), "DEP", 20.0))))
    firm = Firm(
        id="acme",
        operator="a",
        plants=(Plant("INDUSTRIAL", "AUTOS", 10.0),),
        employees=5.0,
        finished_inventories={("INDUSTRIAL", "AUTOS"): 8.0},
    )
    reg = FirmRegistry([firm], cells=cells_from_lists(("INDUSTRIAL",), ("AUTOS",)))
    apply_firm_hook(FirmHookPayload("firm_plant_accident", 2, ("acme",), ("AUTOS",), (), 0.2), reg, led)
    assert reg["acme"].plants[0].capacity == pytest.approx(8.0)
    apply_firm_hook(FirmHookPayload("firm_strike", 3, ("acme",), (), (), None), reg, led)
    assert reg["acme"].employees == 0.0
    apply_firm_hook(FirmHookPayload("firm_recall", 4, ("acme",), (), (), 0.5), reg, led)
    assert_consistent(led)
    assert led.position(firm_entity("acme"), "DEP") < 20.0
