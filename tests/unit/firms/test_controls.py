"""T5.14 — each control fires; regulator fine posts to GOVT."""

from __future__ import annotations

import pytest

from marketsim.firms.controls import check_firm, check_leverage, fine_govt
from marketsim.firms.firm import Firm, FirmsFile, Plant
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def test_each_control() -> None:
    cfg = FirmsFile()
    firm = Firm(id="a", operator="x", plants=(Plant("INDUSTRIAL", "AUTOS", 1.0),), employees=1.0)
    firm.plants[0].capacity = -1.0
    firm.employees = -1.0
    hits = check_firm(firm, cfg, cell_share={("INDUSTRIAL", "AUTOS"): 1.5}, decision_size=2.0, cash=-20.0, undrawn=5.0)
    names = {h.control for h in hits}
    assert "hard_budget" in names
    assert "decision_size" in names
    assert "cell_share_cap" in names
    assert "nonneg" in names
    assert check_leverage(70.0, 10.0, cfg) is not None  # 7 > 6
    assert check_leverage(10.0, 10.0, cfg) is None


def test_regulator_fine_to_govt() -> None:
    led = Ledger.empty()
    led.register_entity("FIRM:a")
    led.register_entity("GOVT")
    led.register_entity("BANKSYS")
    led.post(Tx(0, "opening", (Entry("FIRM:a", "DEP", 8.0), Entry("BANKSYS", "DEP", -8.0))))
    fine_govt(led, "FIRM:a", 3.0, tick=1)
    assert led.position("GOVT", "DEP") == pytest.approx(3.0)
    assert_consistent(led)
