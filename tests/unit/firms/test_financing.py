"""T5.10 — limits, uniform rate, seniority, tax carry, cash floor."""

from __future__ import annotations

import random

import pytest

from marketsim.firms.financing import (
    Payment,
    apply_tax,
    borrowing_rate,
    credit_limit,
    fuzz_never_below_line,
    scale_payments,
)
from marketsim.firms.firm import FirmsFile
from marketsim.firms.rating import score_rating


def _cfg() -> FirmsFile:
    return FirmsFile()


def test_limit_shrinks_with_gate_and_rating() -> None:
    cfg = _cfg()
    base = credit_limit(ebitda_12m=10.0, replacement=100.0, lam=1.0, letter="A", cfg=cfg)
    closed = credit_limit(ebitda_12m=10.0, replacement=100.0, lam=0.5, letter="A", cfg=cfg)
    worse = credit_limit(ebitda_12m=10.0, replacement=100.0, lam=1.0, letter="CCC", cfg=cfg)
    assert closed == pytest.approx(0.5 * base)
    assert worse < base
    assert score_rating(nd_ebitda=5.5, icr=0.5, size=0.2, vol=0.5) == "CCC"


def test_rate_identical_across_firms() -> None:
    r = borrowing_rate(0.04, 0.015)
    assert borrowing_rate(0.04, 0.015) == r
    # rating / leverage do not enter
    assert r == pytest.approx(0.055)


def test_seniority_scaling() -> None:
    bills = [
        Payment("wages", 10.0),
        Payment("taxes", 5.0),
        Payment("dividends", 8.0),
    ]
    res = scale_payments(12.0, 0.0, bills)
    assert res.paid["wages"] == pytest.approx(10.0)
    assert res.paid["taxes"] == pytest.approx(2.0)
    assert res.paid["dividends"] == pytest.approx(0.0)
    assert res.distressed
    assert res.cash == pytest.approx(0.0)


def test_loss_carryforward() -> None:
    tax, carry = apply_tax(-4.0, 0.0, 0.25, years=5)
    assert tax == 0.0
    assert carry == pytest.approx(4.0)
    tax, carry = apply_tax(10.0, carry, 0.25, years=5)
    assert tax == pytest.approx(0.25 * 6.0)
    assert carry == pytest.approx(0.0)


def test_fuzz_cash_floor() -> None:
    rng = random.Random(0)
    for _ in range(200):
        cash = rng.uniform(-5, 20)
        undrawn = rng.uniform(0, 15)
        bills = [Payment(n, rng.uniform(0, 12)) for n in ("wages", "taxes", "capex", "dividends")]
        assert fuzz_never_below_line(cash, undrawn, bills)
        res = scale_payments(cash, undrawn, bills)
        assert res.cash >= -undrawn - 1e-9
