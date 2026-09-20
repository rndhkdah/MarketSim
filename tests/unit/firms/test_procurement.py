"""T5.09 — shortage allocation, premiums, order-size cap."""

from __future__ import annotations

import pytest

from marketsim.core.errors import LedgerError
from marketsim.firms.firm import FirmsFile
from marketsim.firms.procurement import Bid, allocate_shortage, cornering_capped, post_premiums
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent


def _cfg() -> FirmsFile:
    return FirmsFile()


def test_no_shortage_prorata_premiums_unused() -> None:
    cfg = _cfg()
    bids = [Bid("a", 10.0, premium=0.5), Bid("b", 10.0, premium=0.0)]
    alloc = allocate_shortage(bids, 30.0, cfg)
    assert alloc.fill["a"] == pytest.approx(10.0)
    assert alloc.fill["b"] == pytest.approx(10.0)
    assert alloc.premium_paid["a"] == 0.0


def test_shortage_bid_and_relationship() -> None:
    cfg = _cfg()
    bids = [Bid("plain", 10.0, 0.0, 0.0), Bid("rich", 10.0, 0.4, 0.5)]
    alloc = allocate_shortage(bids, 10.0, cfg)
    assert alloc.fill["rich"] > alloc.fill["plain"]
    assert alloc.fill["rich"] + alloc.fill["plain"] == pytest.approx(10.0)
    assert alloc.fill["rich"] <= 10.0
    assert alloc.premium_paid["rich"] > 0


def test_cornering_capped() -> None:
    cfg = _cfg()
    assert cornering_capped(0.5, 10.0, cfg) == pytest.approx(0.5)
    with pytest.raises(LedgerError, match="cell cap"):
        cornering_capped(11.0, 10.0, cfg)


def test_premium_posted_on_ledger() -> None:
    cfg = _cfg()
    led = Ledger.empty()
    led.register_entity("FIRM:a")
    led.register_entity("NPC:INDUSTRIAL:AUTOS")
    led.register_entity("BANKSYS")
    led.post(Tx(0, "opening", (Entry("FIRM:a", "DEP", 5.0), Entry("BANKSYS", "DEP", -5.0))))
    alloc = allocate_shortage([Bid("FIRM:a", 8.0, 0.5)], 4.0, cfg)
    post_premiums(led, alloc, seller="NPC:INDUSTRIAL:AUTOS", tick=1)
    assert_consistent(led)
    assert led.position("NPC:INDUSTRIAL:AUTOS", "DEP") == pytest.approx(alloc.premium_paid["FIRM:a"])
