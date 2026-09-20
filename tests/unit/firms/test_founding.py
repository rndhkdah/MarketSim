"""T5.03 — founding, NPC purchase, greenfield lag."""

from __future__ import annotations

import pytest

from marketsim.firms.accounts import firm_entity
from marketsim.firms.firm import FirmRegistry, cells_from_lists
from marketsim.firms.founding import (
    CellBook,
    NpcCell,
    advance_construction,
    buy_from_npc,
    found_firm,
    months_from_build_lag_q,
    npc_entity,
    split_npc_share,
    start_greenfield,
)
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def _led() -> Ledger:
    led = Ledger.empty()
    led.register_entity("HH:0")
    led.register_entity("BANKSYS")
    led.post(Tx(0, "opening", (Entry("HH:0", "DEP", 500.0), Entry("BANKSYS", "DEP", -500.0))))
    return led


def _reg() -> FirmRegistry:
    return FirmRegistry(cells=cells_from_lists(("INDUSTRIAL",), ("AUTOS", "ENERGY")))


def test_npc_share_split() -> None:
    kept, released = split_npc_share(100.0, 0.7)
    assert kept == pytest.approx(70.0)
    assert released == pytest.approx(30.0)


def test_purchase_conserves_capacity_and_cash() -> None:
    led = _led()
    reg = _reg()
    book = CellBook(
        cells={("INDUSTRIAL", "AUTOS"): NpcCell("INDUSTRIAL", "AUTOS", capacity=100.0, capital_stock=200.0)}
    )
    found_firm(reg, led, firm_id="acme", operator="alice", capital=80.0, source="HH:0", tag="capital_transfer")
    seller = npc_entity("INDUSTRIAL", "AUTOS")
    led.register_entity(seller)
    led.post(Tx(0, "opening", (Entry(seller, "CAPITAL", 200.0),)))
    total_before = book.total_capacity("INDUSTRIAL", "AUTOS", reg)
    cash = buy_from_npc(
        book, reg, led, firm_id="acme", region="INDUSTRIAL", sector="AUTOS", capacity=20.0
    )
    assert cash == pytest.approx(40.0)  # unit replacement 2
    assert book.total_capacity("INDUSTRIAL", "AUTOS", reg) == pytest.approx(total_before)
    assert book.get("INDUSTRIAL", "AUTOS").capacity == pytest.approx(80.0)
    assert reg["acme"].plants[0].capacity == pytest.approx(20.0)
    assert led.position(firm_entity("acme"), "DEP") == pytest.approx(40.0)
    assert led.position(npc_entity("INDUSTRIAL", "AUTOS"), "DEP") == pytest.approx(40.0)
    assert led.position(firm_entity("acme"), "CAPITAL") == pytest.approx(40.0)
    assert led.position(seller, "CAPITAL") == pytest.approx(160.0)
    assert_consistent(led)


def test_greenfield_arrives_after_lag() -> None:
    led = _led()
    reg = _reg()
    found_firm(reg, led, firm_id="acme", operator="alice", capital=90.0, source="HH:0", tag="capital_transfer")
    seller = npc_entity("INDUSTRIAL", "AUTOS")
    led.register_entity(seller)
    lag = months_from_build_lag_q(1.0)  # BIZSVC-like 1q → 3 months
    paid = start_greenfield(
        reg,
        led,
        firm_id="acme",
        region="INDUSTRIAL",
        sector="AUTOS",
        capacity=10.0,
        lag_m=lag,
        unit_cost=3.0,
        seller=seller,
        new_cell=True,
        setup_premium=0.10,
    )
    assert paid == pytest.approx(33.0)
    assert led.position(firm_entity("acme"), "DEP") == pytest.approx(57.0)
    assert led.position(seller, "DEP") == pytest.approx(33.0)
    assert reg["acme"].plants[0].capacity == pytest.approx(0.0)
    advance_construction(reg, months=lag - 1)
    assert reg["acme"].plants[0].capacity == pytest.approx(0.0)
    advance_construction(reg, months=1)
    assert reg["acme"].plants[0].capacity == pytest.approx(10.0)
    assert_consistent(led)


def test_energy_lag_longer_than_one_quarter() -> None:
    assert months_from_build_lag_q(12.0) == 36
    assert months_from_build_lag_q(1.0) == 3
