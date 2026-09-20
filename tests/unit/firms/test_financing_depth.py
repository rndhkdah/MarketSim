"""T8.01 — term loans, covenants, committed lines, extra pool tenors, SFC."""

from __future__ import annotations

import pytest

from marketsim.firms.accounts import register_firm
from marketsim.firms.financing import (
    BANK_LENDER,
    ICR_COVENANT_MIN,
    MONTHS_PER_YEAR,
    POOL_INSTRUMENT,
    POOL_LENDER,
    CreditCapacity,
    Payment,
    TermLoan,
    borrowing_rate,
    check_covenants,
    committed_undrawn,
    credit_capacity,
    credit_limit,
    monthly_interest,
    pool_funding_rate,
    post_loan_draw,
    post_loan_interest,
    principal_due,
    service_term_loan,
    total_drawable,
    uncommitted_undrawn,
)
from marketsim.firms.firm import FirmsFile
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent
from marketsim.market.corp_pool import BOND_DURATION_Y, NOTE_DURATION_Y, POOL_DURATION_Y, y_match
from marketsim.pricing.bonds import (
    EXTRA_POOL_TENORS_Y,
    POOL_TENORS_Y,
    interpolate_govt_yield,
    pool_funding_at_tenor,
    pool_maturity_curve,
)


def _cfg() -> FirmsFile:
    return FirmsFile()


def _healthy_capacity(
    *,
    lam: float,
    committed_cr: float = 0.0,
    letter: str = "A",
) -> CreditCapacity:
    """Covenants slack so the credit gate / committed line are the only quantity limits."""
    return credit_capacity(
        ebitda_12m=100.0,
        replacement=1_000.0,
        lam=lam,
        letter=letter,
        cfg=_cfg(),
        committed_cr=committed_cr,
        net_debt=50.0,
        interest_12m=1.0,
    )


def _open_term_loan(*, face: float, cash: float, instrument: str = "LOAN") -> tuple[Ledger, str]:
    led = Ledger.empty()
    register_firm(led, "acme")
    firm = "FIRM:acme"
    lender = POOL_LENDER if instrument == POOL_INSTRUMENT else BANK_LENDER
    led.register_entity(lender)
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry(firm, "DEP", cash),
                Entry(lender, "DEP", -cash),
                Entry(lender, instrument, face),
                Entry(firm, instrument, -face),
            ),
            memo="term loan opening",
        )
    )
    assert_consistent(led)
    return led, firm


def test_rate_identical_across_firms_still() -> None:
    r_a = borrowing_rate(0.04, 0.015)
    r_b = borrowing_rate(0.04, 0.015)
    assert r_a == r_b
    assert r_a == pytest.approx(0.055)
    loan_a = TermLoan(face=80.0, remaining_m=6, rate=r_a)
    loan_b = TermLoan(face=20.0, remaining_m=3, rate=r_b)
    assert loan_a.rate == loan_b.rate


def test_monthly_interest_is_face_times_rate_over_12() -> None:
    assert MONTHS_PER_YEAR == 12
    assert monthly_interest(120.0, 0.12) == pytest.approx(1.2)


def test_refinancing_wall_closed_gate_produces_distress() -> None:
    """Closed uncommitted gate and no committed line → unpaid principal → distressed."""
    rate = borrowing_rate(0.04, 0.015)
    face = 50.0
    coupon = monthly_interest(face, rate)
    led, firm = _open_term_loan(face=face, cash=coupon)
    loan = TermLoan(face=face, remaining_m=1, rate=rate, tenor_m=12)
    closed = _healthy_capacity(lam=0.0, committed_cr=0.0)
    assert closed.total == pytest.approx(0.0)
    assert not closed.covenant_breach

    month = service_term_loan(led, loan, firm=firm, tick=1, capacity=closed)

    assert month.loan.remaining_m == 0
    assert principal_due(month.loan) == pytest.approx(face)
    assert month.refinanced is False
    assert month.budget.distressed is True
    assert month.budget.unpaid["principal"] == pytest.approx(face)
    assert month.interest_paid == pytest.approx(coupon)
    assert month.principal_paid == pytest.approx(0.0)
    # Residual face stays on the books — never clipped to hide the unpaid wall.
    assert led.position(firm, "LOAN") == pytest.approx(-face)
    assert_consistent(led)


def test_open_gate_refinances_the_wall_sfc() -> None:
    rate = borrowing_rate(0.04, 0.015)
    face = 50.0
    coupon = monthly_interest(face, rate)
    led, firm = _open_term_loan(face=face, cash=coupon)
    loan = TermLoan(face=face, remaining_m=1, rate=rate, tenor_m=12)
    open_ = _healthy_capacity(lam=1.0, committed_cr=0.0)
    assert open_.total >= face

    month = service_term_loan(led, loan, firm=firm, tick=1, capacity=open_)

    assert month.refinanced is True
    assert month.budget.distressed is False
    assert month.loan.remaining_m == 12
    assert month.loan.face == pytest.approx(face)
    assert month.loan.rate == pytest.approx(rate)
    assert led.position(firm, "LOAN") == pytest.approx(-face)
    assert led.position(firm, "DEP") == pytest.approx(0.0)
    assert_consistent(led)


def test_committed_line_refinances_when_gate_closes() -> None:
    rate = borrowing_rate(0.04, 0.015)
    face = 40.0
    coupon = monthly_interest(face, rate)
    led, firm = _open_term_loan(face=face, cash=coupon)
    loan = TermLoan(face=face, remaining_m=1, rate=rate, tenor_m=12)
    backstop = _healthy_capacity(lam=0.0, committed_cr=face)

    assert backstop.uncommitted == pytest.approx(0.0)
    assert backstop.committed == pytest.approx(face)
    assert backstop.total == pytest.approx(face)

    month = service_term_loan(led, loan, firm=firm, tick=1, capacity=backstop)
    assert month.refinanced is True
    assert month.budget.distressed is False
    assert month.committed_used == pytest.approx(face)
    assert_consistent(led)


def test_committed_available_when_lam_zero() -> None:
    cfg = _cfg()
    closed = uncommitted_undrawn(
        ebitda_12m=10.0, replacement=100.0, lam=0.0, letter="A", cfg=cfg
    )
    open_ = uncommitted_undrawn(
        ebitda_12m=10.0, replacement=100.0, lam=1.0, letter="A", cfg=cfg
    )
    half = uncommitted_undrawn(
        ebitda_12m=10.0, replacement=100.0, lam=0.5, letter="A", cfg=cfg
    )
    assert closed == pytest.approx(0.0)
    assert open_ == pytest.approx(
        credit_limit(ebitda_12m=10.0, replacement=100.0, lam=1.0, letter="A", cfg=cfg)
    )
    assert half == pytest.approx(0.5 * open_)
    assert committed_undrawn(12.0, drawn=3.0) == pytest.approx(9.0)
    assert total_drawable(12.0, closed, covenant_breach=False) == pytest.approx(12.0)


def test_covenants_restrict_drawing_not_the_rate() -> None:
    cfg = _cfg()
    nd = check_covenants(net_debt=50.0, ebitda_12m=10.0, interest_12m=2.0, cfg=cfg)
    assert nd.nd_ebitda == pytest.approx(5.0)
    assert nd.nd_breach is True
    assert nd.icr_breach is False
    assert nd.covenant_breach is True

    icr = check_covenants(net_debt=10.0, ebitda_12m=10.0, interest_12m=10.0, cfg=cfg)
    assert icr.icr == pytest.approx(1.0)
    assert icr.icr < ICR_COVENANT_MIN
    assert icr.icr_breach is True
    assert icr.covenant_breach is True

    ok = check_covenants(net_debt=10.0, ebitda_12m=10.0, interest_12m=2.0, cfg=cfg)
    assert ok.covenant_breach is False

    blocked = credit_capacity(
        ebitda_12m=10.0,
        replacement=100.0,
        lam=1.0,
        letter="A",
        cfg=cfg,
        committed_cr=30.0,
        net_debt=50.0,
        interest_12m=2.0,
    )
    assert blocked.covenant_breach is True
    assert blocked.committed == pytest.approx(30.0)
    assert blocked.total == pytest.approx(0.0)
    assert borrowing_rate(0.04, 0.015) == pytest.approx(0.055)


def test_draw_interest_refinance_postings_sfc() -> None:
    led = Ledger.empty()
    register_firm(led, "a")
    led.register_entity(BANK_LENDER)
    post_loan_draw(led, firm="FIRM:a", amount=20.0, tick=1)
    post_loan_interest(led, firm="FIRM:a", amount=1.0, tick=2, lender=BANK_LENDER)
    assert led.position("FIRM:a", "DEP") == pytest.approx(19.0)
    assert led.position("FIRM:a", "LOAN") == pytest.approx(-20.0)
    assert_consistent(led)


def test_pool_cloan_wall_closed_gate_distresses() -> None:
    rate = pool_funding_rate(y_match(0.03, 0.05), 0.015)
    face = 25.0
    coupon = monthly_interest(face, rate)
    led, firm = _open_term_loan(face=face, cash=coupon, instrument=POOL_INSTRUMENT)
    loan = TermLoan(face=face, remaining_m=1, rate=rate, instrument=POOL_INSTRUMENT)
    closed = _healthy_capacity(lam=0.0, committed_cr=0.0)
    month = service_term_loan(led, loan, firm=firm, tick=1, capacity=closed)
    assert month.budget.distressed is True
    assert led.position(firm, POOL_INSTRUMENT) == pytest.approx(-face)
    assert_consistent(led)


def test_extra_pool_tenors_common_spread() -> None:
    assert POOL_DURATION_Y in POOL_TENORS_Y
    assert EXTRA_POOL_TENORS_Y == (NOTE_DURATION_Y, BOND_DURATION_Y)
    y_note, y_bond, spread = 0.03, 0.05, 0.01
    assert interpolate_govt_yield(y_note, y_bond, POOL_DURATION_Y) == pytest.approx(
        y_match(y_note, y_bond)
    )
    short = pool_funding_at_tenor(y_note, y_bond, spread, NOTE_DURATION_Y)
    long_ = pool_funding_at_tenor(y_note, y_bond, spread, BOND_DURATION_Y)
    mid = pool_funding_at_tenor(y_note, y_bond, spread, POOL_DURATION_Y)
    assert short == pytest.approx(y_note + spread)
    assert long_ == pytest.approx(y_bond + spread)
    assert short < mid < long_
    # Same tenor, same global spread — two names, identical rate (D14).
    assert pool_funding_at_tenor(y_note, y_bond, spread, POOL_DURATION_Y) == mid
    curve = pool_maturity_curve(y_note, y_bond, spread)
    assert tuple(t for t, _ in curve) == POOL_TENORS_Y


def test_term_loan_state_roundtrip() -> None:
    loan = TermLoan(face=15.0, remaining_m=4, rate=0.055, tenor_m=12, instrument="LOAN")
    assert TermLoan.from_state(loan.to_state()) == loan


def test_seniority_still_pays_wages_before_unpaid_principal() -> None:
    """Closed gate: cash covers wages then interest; principal unpaid → distress."""
    from marketsim.firms.financing import scale_payments

    bills = [
        Payment("wages", 3.0),
        Payment("interest", 1.0),
        Payment("principal", 10.0),
    ]
    out = scale_payments(4.0, 0.0, bills)
    assert out.distressed is True
    assert out.paid["wages"] == pytest.approx(3.0)
    assert out.paid["interest"] == pytest.approx(1.0)
    assert out.unpaid["principal"] == pytest.approx(10.0)
