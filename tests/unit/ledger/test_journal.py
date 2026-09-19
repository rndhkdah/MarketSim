from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.errors import LedgerError
from marketsim.ledger.journal import Entry, Ledger, Tx


def _seed_hh_bank() -> Ledger:
    led = Ledger.empty(debug_journal=True)
    led.register_entity("HH:0")
    led.register_entity("BANKSYS")
    led.register_entity("NPC:0:AUTOS")
    # Opening: household holds 100 DEP; bank owes 100 DEP.
    led.post(
        Tx(
            0,
            "opening",
            [Entry("HH:0", "DEP", 100.0), Entry("BANKSYS", "DEP", -100.0)],
        )
    )
    return led


def test_unbalanced_tx_rejected() -> None:
    led = Ledger.empty()
    led.register_entity("HH:0")
    with pytest.raises(LedgerError, match="unbalanced"):
        led.post(Tx(0, "consumption", [Entry("HH:0", "DEP", -10.0)]))


def test_payment_to_banksys_reduces_dep_liability() -> None:
    led = _seed_hh_bank()
    assert led.position("BANKSYS", "DEP") == pytest.approx(-100.0)
    led.post(
        Tx(
            1,
            "fees",
            [Entry("HH:0", "DEP", -10.0), Entry("BANKSYS", "DEP", 10.0)],
        )
    )
    assert led.position("BANKSYS", "DEP") == pytest.approx(-90.0)
    assert led.position("HH:0", "DEP") == pytest.approx(90.0)


def test_new_loan_creates_deposit() -> None:
    led = _seed_hh_bank()
    firm = "NPC:0:AUTOS"
    x = 25.0
    led.post(
        Tx(
            1,
            "loan_new",
            [
                Entry(firm, "DEP", x),
                Entry("BANKSYS", "DEP", -x),
                Entry("BANKSYS", "LOAN", x),
                Entry(firm, "LOAN", -x),
            ],
        )
    )
    assert led.position(firm, "DEP") == pytest.approx(x)
    assert led.position(firm, "LOAN") == pytest.approx(-x)
    assert led.position("BANKSYS", "LOAN") == pytest.approx(x)
    assert led.position("BANKSYS", "DEP") == pytest.approx(-125.0)


def test_registration_after_start_keeps_ids_stable() -> None:
    led = _seed_hh_bank()
    hh = led.entities.id("HH:0")
    dep = led.instruments.id("DEP")
    led.register_entity("NPC:0:SOFTWARE")
    led.register_instrument("EQ:SOFTWARE", financial=True)
    assert led.entities.id("HH:0") == hh
    assert led.instruments.id("DEP") == dep
    assert led.entities.id("NPC:0:SOFTWARE") > hh
    assert led.pos.shape == (len(led.entities), len(led.instruments))


def test_state_round_trip() -> None:
    led = _seed_hh_bank()
    led.post(
        Tx(1, "wages", [Entry("BANKSYS", "DEP", -4.0), Entry("HH:0", "DEP", 4.0)]),
    )
    restored = Ledger.from_state(led.to_state())
    assert restored.entities.names == led.entities.names
    assert restored.instruments.names == led.instruments.names
    assert np.allclose(restored.pos, led.pos)
    assert restored.position("HH:0", "DEP") == pytest.approx(104.0)
