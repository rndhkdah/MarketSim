"""T8.01 — term loans, covenants, committed lines, refinancing wall, SFC."""

from __future__ import annotations

import pytest

from marketsim.firms.financing import (
    ICR_COVENANT,
    MONTHS_PER_YEAR,
    LineBook,
    Payment,
    TermLoan,
    borrowing_rate,
    covenant_icr_breach,
    covenant_nd_breach,
    drawable,
    monthly_interest,
    post_loan_draw,
    post_loan_interest,
    principal_due,
    refinance_wall,
)
from marketsim.ledger.journal import Ledger
from marketsim.ledger.sfc import assert_consistent
from marketsim.pricing.bonds import POOL_TENORS_Y, interpolate_govt_yield, pool_funding_at_tenor


def test_rate_still_common() -> None:
    assert borrowing_rate(0.04, 0.015) == borrowing_rate(0.04, 0.015)
    assert borrowing_rate(0.04, 0.015) == pytest.approx(0.055)


def test_refinancing_wall_closed_gate_distresses() -> None:
    loan = TermLoan(face=50.0, remaining_m=0, rate=0.06)
    closed = LineBook(committed_cr=0.0, uncommitted_cr=50.0, lam=0.0)
    open_ = LineBook(committed_cr=0.0, uncommitted_cr=50.0, lam=1.0)
    wall = refinance_wall(loan, cash=5.0, book=closed)
    refinanced = refinance_wall(loan, cash=5.0, book=open_)
    assert principal_due(loan) == pytest.approx(50.0)
    assert wall.distressed is True
    assert wall.unpaid["principal"] > 0.0
    assert refinanced.distressed is False


def test_committed_line_survives_closed_gate() -> None:
    c, u, tot = drawable(LineBook(committed_cr=10.0, uncommitted_cr=40.0, lam=0.0))
    assert c == pytest.approx(10.0)
    assert u == pytest.approx(0.0)
    assert tot == pytest.approx(10.0)


def test_covenants_are_quantity_not_price() -> None:
    assert covenant_nd_breach(70.0, 10.0, 6.0) is True
    assert covenant_nd_breach(10.0, 10.0, 6.0) is False
    assert covenant_icr_breach(0.5, 1.0, ICR_COVENANT) is True
    assert monthly_interest(TermLoan(face=120.0, remaining_m=6, rate=0.12)) == pytest.approx(1.2)
    assert MONTHS_PER_YEAR == 12


def test_loan_postings_sfc() -> None:
    led = Ledger.empty()
    led.register_entity("FIRM:a")
    led.register_entity("BANKSYS")
    post_loan_draw(led, "FIRM:a", 20.0, tick=1)
    post_loan_interest(led, "FIRM:a", 1.0, tick=2)
    assert led.position("FIRM:a", "DEP") == pytest.approx(19.0)
    assert led.position("FIRM:a", "LOAN") == pytest.approx(-20.0)
    assert_consistent(led)


def test_extra_pool_tenors_common_spread() -> None:
    assert 4.2 in POOL_TENORS_Y
    y2 = interpolate_govt_yield(0.03, 0.05, 2.0)
    y7 = interpolate_govt_yield(0.03, 0.05, 7.0)
    assert y2 < y7
    a = pool_funding_at_tenor(0.03, 0.05, 0.01, 4.2)
    b = pool_funding_at_tenor(0.03, 0.05, 0.01, 4.2)
    assert a == b


def test_other_senior_bills_still_apply() -> None:
    loan = TermLoan(face=10.0, remaining_m=0, rate=0.05)
    book = LineBook(committed_cr=0.0, uncommitted_cr=0.0, lam=0.0)
    out = refinance_wall(loan, cash=4.0, book=book, other_bills=[Payment("wages", 3.0)])
    assert out.distressed is True
    assert out.paid["wages"] == pytest.approx(3.0)
