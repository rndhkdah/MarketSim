"""T8.08 — fines are ledger postings; share caps and merger control bind."""

from __future__ import annotations

from pathlib import Path

import pytest

from marketsim.events.schema import PRIMITIVES, load_catalog, load_event
from marketsim.firms.controls import apply_fine, check_firm, fine_govt, merger_control
from marketsim.firms.firm import ControlsCfg, Firm, FirmsFile, Plant
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def test_fine_is_ledger_posting() -> None:
    led = Ledger.empty()
    led.register_entity("FIRM:a")
    led.register_entity("GOVT")
    led.register_entity("BANKSYS")
    led.post(Tx(0, "opening", (Entry("FIRM:a", "DEP", 8.0), Entry("BANKSYS", "DEP", -8.0))))
    fine_govt(led, "FIRM:a", 1.0, tick=1)
    apply_fine(led, "FIRM:a", 2.0, tick=2)
    assert led.position("GOVT", "DEP") == pytest.approx(3.0)
    assert led.position("FIRM:a", "DEP") == pytest.approx(5.0)
    assert led.period_flows[("FIRM:a", "GOVT", "fees")] == pytest.approx(3.0)
    assert_consistent(led)


def test_cell_share_cap_binds() -> None:
    cfg = FirmsFile(controls=ControlsCfg(cell_share_cap=0.5))
    firm = Firm(id="a", operator="x", plants=(Plant("INDUSTRIAL", "AUTOS", 1.0),), employees=1.0)
    hits = check_firm(
        firm,
        cfg,
        cell_share={("INDUSTRIAL", "AUTOS"): 0.6},
        decision_size=0.0,
        cash=1.0,
        undrawn=1.0,
    )
    assert any(h.control == "cell_share_cap" for h in hits)


def test_merger_control_blocks_over_cap() -> None:
    assert merger_control(0.30, 0.15, 0.50) == "allow"
    assert merger_control(0.40, 0.20, 0.50) == "block"
    assert merger_control(0.20, 0.20, 0.50) == "allow"


def test_regulator_events_load(config_dir: Path) -> None:
    events = config_dir / "events"
    anti = load_event(events / "regulator_antitrust.yaml")
    merge = load_event(events / "regulator_merger.yaml")
    assert anti.id == "regulator_antitrust"
    assert merge.id == "regulator_merger"
    catalog = load_catalog(events)
    assert "regulator_antitrust" in catalog
    assert "regulator_merger" in catalog
    for spec in (anti, merge):
        assert spec.composition
        for shock in spec.composition:
            assert shock.shock in PRIMITIVES
