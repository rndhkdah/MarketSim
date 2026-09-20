"""T5.17 gate 2 — scripted attacker policies stay bounded."""

from __future__ import annotations

import pytest

from marketsim.core.errors import LedgerError
from marketsim.firms.financing import Payment, scale_payments
from marketsim.firms.firm import Firm, FirmsFile, Plant
from marketsim.firms.goods_market import SellerQuote, allocate
from marketsim.firms.levers import FirmDecision, validate_decision
from marketsim.firms.procurement import Bid, allocate_shortage, cornering_capped


def test_price_at_floor_does_not_take_all_share() -> None:
    cfg = FirmsFile()
    quotes = [
        SellerQuote("npc", 80.0, 1.0, 80.0, 0.8),
        SellerQuote("floor", 20.0, 0.01, 20.0, 0.2),
    ]
    first = None
    for i in range(24):
        for q in quotes:
            q.avail = q.capacity
        res = allocate(quotes, 80.0, cfg)
        if i == 0:
            first = quotes[1].share
            assert res.sales["npc"] > 0
            assert quotes[1].share < 0.45  # loyalty; not a one-tick grab
    assert first is not None
    assert quotes[1].share < 1.0


def test_input_cornering_hits_cap() -> None:
    cfg = FirmsFile()
    with pytest.raises(LedgerError):
        cornering_capped(50.0, 10.0, cfg)
    alloc = allocate_shortage([Bid("attacker", 50.0)], 10.0, cfg)
    assert alloc.fill["attacker"] <= 10.0


def test_max_leverage_and_unbounded_hiring_clipped() -> None:
    cfg = FirmsFile()
    firm = Firm(
        id="atk",
        operator="x",
        plants=(Plant("INDUSTRIAL", "AUTOS", 5.0),),
        employees=2.0,
        credit_limit=10.0,
        credit_drawn=9.0,
        posted_price={("INDUSTRIAL", "AUTOS"): 1.0},
    )
    out = validate_decision(
        FirmDecision("atk", "x", borrow=1e6, vacancies=1e9),
        firm,
        cfg,
        unemployed=20.0,
    )
    assert out.decision.borrow == pytest.approx(1.0)
    assert out.decision.vacancies == pytest.approx(2.0)


def test_strategic_default_scales_and_stays_finite() -> None:
    res = scale_payments(0.0, 1.0, [Payment("wages", 5.0), Payment("dividends", 20.0)])
    assert res.cash >= -1.0
    assert res.distressed
    assert res.paid["dividends"] == 0.0
