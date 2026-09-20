"""Uniform corporate rate, rating-scaled limits, tax, hard budget (T5.10 / §5.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marketsim.firms.firm import FirmsFile
from marketsim.firms.rating import limit_multiplier

SENIORITY: tuple[str, ...] = (
    "wages",
    "taxes",
    "suppliers",
    "interest",
    "principal",
    "capex",
    "dividends",
)


def borrowing_rate(policy_rate: float, spread: float) -> float:
    """Common corporate rate (D14). Annual decimal; no firm/sector/rating term."""
    return float(policy_rate) + float(spread)


def pool_funding_rate(y_match: float, spread: float) -> float:
    """Fixed pool rate ``y_match + s_t`` (§6.11 / D14). Annual decimal; no name term."""
    return float(y_match) + float(spread)


@dataclass(frozen=True)
class FundingMenu:
    """Same menu for every firm: floating bank loan or fixed pool (§6.11).

    Rates and ``spread`` / ``y_match`` are annual decimals.
    """

    bank_rate: float
    pool_rate: float
    spread: float
    y_match: float


def funding_menu(*, policy_rate: float, y_match: float, spread: float) -> FundingMenu:
    """Bank ``r + s_t`` and pool ``y_match + s_t``. Rating / sector / region do not enter."""
    s = float(spread)
    ym = float(y_match)
    return FundingMenu(
        bank_rate=borrowing_rate(policy_rate, s),
        pool_rate=pool_funding_rate(ym, s),
        spread=s,
        y_match=ym,
    )


def funding_split(need: float, pool_share: float) -> tuple[float, float]:
    """``(pool, bank)`` faces (cr). ``pool_share`` is NPC ``corp_funding_mix`` or an agent's choice."""
    need_ = float(need)
    share = float(pool_share)
    return need_ * share, need_ * (1.0 - share)


def credit_limit(
    *,
    ebitda_12m: float,
    replacement: float,
    lam: float,
    letter: str,
    cfg: FirmsFile,
) -> float:
    """Bank line in cr. ``lam`` is the lagged credit gate (dimensionless)."""
    fin = cfg.financing
    by_lev = fin.nd_ebitda_hard * max(ebitda_12m, 0.0)
    by_coll = fin.credit_line_of_replacement * max(replacement, 0.0)
    return min(by_lev, by_coll) * max(lam, 0.0) * limit_multiplier(letter, cfg)


def apply_tax(ebt: float, carry: float, rate: float, *, years: int) -> tuple[float, float]:
    """Corporate tax on positive EBT with loss carry-forward. Returns ``(tax, new_carry)`` in cr."""
    del years
    if ebt < 0:
        return 0.0, carry - ebt
    used = min(carry, ebt)
    return rate * (ebt - used), carry - used


@dataclass
class Payment:
    name: str
    amount: float  # cr due


@dataclass
class BudgetResult:
    paid: dict[str, float]
    unpaid: dict[str, float]
    cash: float
    distressed: bool


def scale_payments(cash: float, undrawn: float, bills: list[Payment]) -> BudgetResult:
    """Pay in seniority order. Never take cash below ``-undrawn``."""
    line = max(undrawn, 0.0)
    pocket = cash + line
    paid: dict[str, float] = {}
    unpaid: dict[str, float] = {}
    for bill in bills:
        due = max(bill.amount, 0.0)
        take = min(due, pocket)
        paid[bill.name] = take
        unpaid[bill.name] = due - take
        pocket -= take
    new_cash = pocket - line
    distressed = any(v > 1e-12 for v in unpaid.values())
    return BudgetResult(paid, unpaid, new_cash, distressed)


def fuzz_never_below_line(cash: float, undrawn: float, bills: list[Payment]) -> bool:
    return scale_payments(cash, undrawn, bills).cash >= -undrawn - 1e-9


# T8.01: monthly interest = face * rate / 12. ``rate`` is annual decimal.
MONTHS_PER_YEAR = 12
# T8.01 / §6.9 (c): interest-cover covenant floor (EBITDA / interest, dimensionless).
ICR_COVENANT_MIN = 1.5
# Skip no-op ledger posts (cr). Same floor as corp_pool on-lend.
POST_EPS = 1e-15
# Dimensionless; avoid false covenant breaches on float noise.
RATIO_TOL = 1e-12
# T8.01 / §6.11: bank term loans vs pooled CLOAN. Strings match corp_pool.ISSUER / CLOAN.
BANK_LENDER = "BANKSYS"
BANK_INSTRUMENT = "LOAN"
POOL_LENDER = "CORPPOOL"
POOL_INSTRUMENT = "CLOAN"


def loan_lender(instrument: str) -> str:
    """Counterparty for ``instrument``: ``LOAN`` → BANKSYS, ``CLOAN`` → CORPPOOL."""
    if instrument == POOL_INSTRUMENT:
        return POOL_LENDER
    return BANK_LENDER


def monthly_interest(face: float, rate: float) -> float:
    """Interest due this month (cr). ``rate`` is annual decimal; T8.01: ``face * rate / 12``."""
    return float(face) * float(rate) / MONTHS_PER_YEAR


@dataclass(frozen=True)
class TermLoan:
    """Bullet term loan. Face in cr; ``remaining_m`` / ``tenor_m`` in months.

    Contractual ``rate`` is the common corporate rate ``r + s_t`` (annual decimal).
    No firm / sector / region / rating term (D14 / T8.01).
    """

    face: float
    remaining_m: int
    rate: float
    tenor_m: int | None = None
    instrument: str = BANK_INSTRUMENT

    @property
    def lender(self) -> str:
        """Ledger entity that holds the matching asset (BANKSYS or CORPPOOL)."""
        return loan_lender(self.instrument)

    @property
    def tenor(self) -> int:
        """Contractual maturity (months). Defaults to the origination ``remaining_m``."""
        return int(self.remaining_m if self.tenor_m is None else self.tenor_m)

    def to_state(self) -> dict[str, Any]:
        return {
            "face": float(self.face),
            "remaining_m": int(self.remaining_m),
            "rate": float(self.rate),
            "tenor_m": None if self.tenor_m is None else int(self.tenor_m),
            "instrument": str(self.instrument),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TermLoan:
        tenor = state.get("tenor_m")
        return cls(
            face=float(state["face"]),
            remaining_m=int(state["remaining_m"]),
            rate=float(state["rate"]),
            tenor_m=None if tenor is None else int(tenor),
            instrument=str(state.get("instrument", BANK_INSTRUMENT)),
        )


def age_term_loan(loan: TermLoan) -> TermLoan:
    """Count down remaining months by one. Principal is due at 0 (T8.01)."""
    remaining = int(loan.remaining_m) - 1
    if remaining < 0:
        remaining = 0
    return replace(loan, remaining_m=remaining)


@dataclass(frozen=True)
class CovenantResult:
    """ND/EBITDA and interest-cover checks (T8.01 / §5.5). Ratios are dimensionless."""

    nd_ebitda: float
    icr: float
    nd_breach: bool
    icr_breach: bool
    covenant_breach: bool


def _nd_ratio(net_debt: float, ebitda_12m: float) -> float:
    """ND / EBITDA (dimensionless). Infinite when EBITDA ≤ 0 and ND > 0."""
    ebitda = float(ebitda_12m)
    debt = float(net_debt)
    if ebitda > 0.0:
        return debt / ebitda
    if debt > 0.0:
        return math.inf
    return 0.0


def _icr(ebitda_12m: float, interest_12m: float) -> float:
    """Interest cover EBITDA / interest (dimensionless). Infinite when interest ≤ 0."""
    interest = float(interest_12m)
    if interest > 0.0:
        return float(ebitda_12m) / interest
    return math.inf


def check_covenants(
    *,
    net_debt: float,
    ebitda_12m: float,
    interest_12m: float,
    cfg: FirmsFile,
) -> CovenantResult:
    """ND/EBITDA (soft cap) and ICR (1.5) checks. Breach is a quantity flag, never a spread.

    ``net_debt``, ``ebitda_12m`` and ``interest_12m`` are cr (12-month sums for the
    flows). Thresholds: ``financing.nd_ebitda_soft`` (§5.5 leverage norm) and
    ``ICR_COVENANT_MIN`` = 1.5 (§6.9 (c) / T8.01).
    """
    nd_ratio = _nd_ratio(net_debt, ebitda_12m)
    icr = _icr(ebitda_12m, interest_12m)
    nd_breach = nd_ratio > float(cfg.financing.nd_ebitda_soft) + RATIO_TOL
    icr_breach = icr < ICR_COVENANT_MIN - RATIO_TOL
    return CovenantResult(
        nd_ebitda=nd_ratio,
        icr=icr,
        nd_breach=nd_breach,
        icr_breach=icr_breach,
        covenant_breach=nd_breach or icr_breach,
    )


def uncommitted_undrawn(
    *,
    ebitda_12m: float,
    replacement: float,
    lam: float,
    letter: str,
    cfg: FirmsFile,
) -> float:
    """Uncommitted undrawn line (cr). T8.01: ``credit_limit(...) * lam``.

    The ungated quantity is ``credit_limit`` at ``lam=1``; the credit gate then
    scales it. Equivalent to T5.10 ``credit_limit`` at the live ``lam``.
    """
    ungated = credit_limit(
        ebitda_12m=ebitda_12m,
        replacement=replacement,
        lam=1.0,
        letter=letter,
        cfg=cfg,
    )
    return ungated * max(float(lam), 0.0)


def committed_undrawn(committed_cr: float, *, drawn: float = 0.0) -> float:
    """Undrawn committed line (cr). Available even when ``lam → 0`` (T8.01)."""
    return max(float(committed_cr) - max(float(drawn), 0.0), 0.0)


def total_drawable(
    committed: float,
    uncommitted: float,
    *,
    covenant_breach: bool = False,
) -> float:
    """Total drawable (cr) = committed + uncommitted, or 0 on covenant breach (T8.01)."""
    if covenant_breach:
        return 0.0
    return max(float(committed), 0.0) + max(float(uncommitted), 0.0)


@dataclass(frozen=True)
class CreditCapacity:
    """Drawable lines (cr). ``committed`` / ``uncommitted`` are undrawn; ``total`` is drawable.

    On ``covenant_breach``, ``total`` is 0 (further drawing restricted) while the
    two line sizes remain visible (T8.01).
    """

    committed: float
    uncommitted: float
    total: float
    covenant_breach: bool
    covenants: CovenantResult


def credit_capacity(
    *,
    ebitda_12m: float,
    replacement: float,
    lam: float,
    letter: str,
    cfg: FirmsFile,
    committed_cr: float,
    committed_drawn: float = 0.0,
    net_debt: float,
    interest_12m: float,
) -> CreditCapacity:
    """Committed + uncommitted capacity and covenant flag (T8.01). Amounts in cr."""
    covenants = check_covenants(
        net_debt=net_debt,
        ebitda_12m=ebitda_12m,
        interest_12m=interest_12m,
        cfg=cfg,
    )
    committed = committed_undrawn(committed_cr, drawn=committed_drawn)
    uncommitted = uncommitted_undrawn(
        ebitda_12m=ebitda_12m,
        replacement=replacement,
        lam=lam,
        letter=letter,
        cfg=cfg,
    )
    total = total_drawable(committed, uncommitted, covenant_breach=covenants.covenant_breach)
    return CreditCapacity(
        committed=committed,
        uncommitted=uncommitted,
        total=total,
        covenant_breach=covenants.covenant_breach,
        covenants=covenants,
    )


def allocate_draw(amount: float, capacity: CreditCapacity) -> tuple[float, float]:
    """Split ``amount`` cr across uncommitted then committed. ``(uncommitted, committed)``.

    Uncommitted is used first so the committed line is the closed-gate backstop
    (T8.01). Covenant breach → ``(0, 0)``.
    """
    if capacity.covenant_breach:
        return 0.0, 0.0
    want = max(float(amount), 0.0)
    uncommitted = min(want, max(capacity.uncommitted, 0.0))
    committed = min(want - uncommitted, max(capacity.committed, 0.0))
    return uncommitted, committed


def can_refinance(face: float, drawable: float) -> bool:
    """True iff ``drawable`` covers the maturing face (cr). T8.01 refinancing wall."""
    return float(drawable) + POST_EPS >= max(float(face), 0.0) and float(face) > POST_EPS


def _cash(ledger: Ledger, firm: str) -> float:
    """Firm deposit balance (cr). Missing books read as 0."""
    if firm not in ledger.entities or "DEP" not in ledger.instruments:
        return 0.0
    return float(ledger.position(firm, "DEP"))


def post_loan_draw(
    ledger: Ledger,
    *,
    firm: str,
    amount: float,
    tick: int,
    instrument: str = BANK_INSTRUMENT,
    lender: str | None = None,
) -> None:
    """Draw ``amount`` cr: DEP/LOAN or DEP/CLOAN. Tag ``loan_new``. SFC-checked."""
    qty = float(amount)
    if qty <= POST_EPS:
        return
    counterparty = lender or loan_lender(instrument)
    ledger.post(
        Tx(
            tick,
            "loan_new",
            (
                Entry(firm, "DEP", qty),
                Entry(counterparty, "DEP", -qty),
                Entry(counterparty, instrument, qty),
                Entry(firm, instrument, -qty),
            ),
            memo="term draw",
        )
    )
    assert_consistent(ledger)


def post_loan_repay(
    ledger: Ledger,
    *,
    firm: str,
    amount: float,
    tick: int,
    instrument: str = BANK_INSTRUMENT,
    lender: str | None = None,
) -> None:
    """Repay ``amount`` cr of principal. Tag ``loan_repay``. SFC-checked."""
    qty = float(amount)
    if qty <= POST_EPS:
        return
    counterparty = lender or loan_lender(instrument)
    ledger.post(
        Tx(
            tick,
            "loan_repay",
            (
                Entry(firm, "DEP", -qty),
                Entry(counterparty, "DEP", qty),
                Entry(counterparty, instrument, -qty),
                Entry(firm, instrument, qty),
            ),
            memo="term repay",
        )
    )
    assert_consistent(ledger)


def post_loan_interest(
    ledger: Ledger,
    *,
    firm: str,
    amount: float,
    tick: int,
    lender: str,
) -> None:
    """Pay ``amount`` cr of interest. Tag ``interest_loans``. DEP only. SFC-checked."""
    qty = float(amount)
    if qty <= POST_EPS:
        return
    ledger.post(
        Tx(
            tick,
            "interest_loans",
            (Entry(firm, "DEP", -qty), Entry(lender, "DEP", qty)),
            memo="term interest",
        )
    )
    assert_consistent(ledger)


def _line_used(cash0: float, cash1: float) -> float:
    """Additional overdraft (cr) implied by ``scale_payments`` cash going more negative."""
    return max(0.0, -float(cash1)) - max(0.0, -float(cash0))


@dataclass
class TermLoanMonth:
    """One monthly service of a term loan. Money amounts in cr."""

    loan: TermLoan
    budget: BudgetResult
    refinanced: bool
    interest_paid: float
    principal_paid: float
    line_drawn: float
    committed_used: float
    uncommitted_used: float


def service_term_loan(
    ledger: Ledger,
    loan: TermLoan,
    *,
    firm: str,
    tick: int,
    capacity: CreditCapacity,
) -> TermLoanMonth:
    """Age the loan one month, post interest, and roll or bill maturing principal.

    If the loan matures and ``capacity.total`` covers the face, the wall is
    refinanced (draw then repay at the same common rate). If the credit gate is
    closed (``lam=0`` and committed undrawn = 0) the face becomes a principal
    bill; ``scale_payments`` leaves it unpaid → ``budget.distressed`` (T8.01).
    Never clips cash or residual face to hide a leak.
    """
    interest_due = monthly_interest(loan.face, loan.rate)
    aged = age_term_loan(loan)
    matured = aged.remaining_m <= 0 and float(loan.face) > POST_EPS

    refinanced = False
    committed_used = 0.0
    uncommitted_used = 0.0
    line_drawn = 0.0
    leftover = max(capacity.total, 0.0)
    working = aged

    if matured:
        uncommitted_used, committed_used = allocate_draw(loan.face, capacity)
        funded = uncommitted_used + committed_used
        if can_refinance(loan.face, funded):
            post_loan_draw(
                ledger,
                firm=firm,
                amount=loan.face,
                tick=tick,
                instrument=loan.instrument,
                lender=loan.lender,
            )
            post_loan_repay(
                ledger,
                firm=firm,
                amount=loan.face,
                tick=tick,
                instrument=loan.instrument,
                lender=loan.lender,
            )
            refinanced = True
            leftover = max(capacity.total - loan.face, 0.0)
            working = replace(aged, remaining_m=aged.tenor, face=float(loan.face))
            bills = [Payment("interest", interest_due)]
        else:
            uncommitted_used = 0.0
            committed_used = 0.0
            bills = [Payment("interest", interest_due), Payment("principal", float(loan.face))]
    else:
        bills = [Payment("interest", interest_due)]

    cash0 = _cash(ledger, firm)
    budget = scale_payments(cash0, leftover, bills)
    overdraft = _line_used(cash0, budget.cash)
    if overdraft > POST_EPS:
        extra_u, extra_c = allocate_draw(overdraft, capacity)
        # Residual capacity after a roll; if the wall was not rolled, leftover == total.
        extra_u = min(extra_u, leftover)
        extra_c = min(extra_c, max(leftover - extra_u, 0.0))
        drawn = extra_u + extra_c
        if drawn + POST_EPS < overdraft:
            drawn = min(overdraft, leftover)
        post_loan_draw(
            ledger,
            firm=firm,
            amount=drawn,
            tick=tick,
            instrument=loan.instrument,
            lender=loan.lender,
        )
        line_drawn += drawn
        uncommitted_used += extra_u
        committed_used += extra_c

    interest_paid = float(budget.paid.get("interest", 0.0))
    principal_paid = float(budget.paid.get("principal", 0.0))
    post_loan_interest(ledger, firm=firm, amount=interest_paid, tick=tick, lender=loan.lender)
    if principal_paid > POST_EPS:
        post_loan_repay(
            ledger,
            firm=firm,
            amount=principal_paid,
            tick=tick,
            instrument=loan.instrument,
            lender=loan.lender,
        )
        working = replace(working, face=float(budget.unpaid.get("principal", 0.0)))

    return TermLoanMonth(
        loan=working,
        budget=budget,
        refinanced=refinanced,
        interest_paid=interest_paid,
        principal_paid=principal_paid,
        line_drawn=line_drawn,
        committed_used=committed_used,
        uncommitted_used=uncommitted_used,
    )


# Master plan §6: 12 months / year. T8.01 term-loan coupon.
MONTHS_PER_YEAR = 12
# T8.01 covenant ICR floor (EBIT / interest). Quantity gate, not a spread.
ICR_COVENANT = 1.0


@dataclass(frozen=True)
class TermLoan:
    """One term loan. ``face`` is cr; ``remaining_m`` is months; ``rate`` is annual decimal."""

    face: float
    remaining_m: int
    rate: float
    committed: bool = False


@dataclass(frozen=True)
class LineBook:
    """Committed + uncommitted capacity (cr). Uncommitted scales with ``lam``."""

    committed_cr: float
    uncommitted_cr: float
    lam: float


def monthly_interest(loan: TermLoan) -> float:
    """Coupon this month (cr). ``rate`` is annual; divide by ``MONTHS_PER_YEAR``."""
    return float(loan.face) * float(loan.rate) / float(MONTHS_PER_YEAR)


def age_loan(loan: TermLoan) -> TermLoan:
    """Advance one month. ``remaining_m`` is months."""
    return TermLoan(
        face=loan.face,
        remaining_m=max(int(loan.remaining_m) - 1, 0),
        rate=loan.rate,
        committed=loan.committed,
    )


def principal_due(loan: TermLoan) -> float:
    """Face (cr) due when ``remaining_m`` is 0; else 0."""
    return float(loan.face) if int(loan.remaining_m) <= 0 else 0.0


def covenant_nd_breach(nd: float, ebitda: float, hard: float) -> bool:
    """True when ND/EBITDA exceeds ``hard``. Quantity covenant; rate unchanged."""
    if ebitda <= 0.0:
        return nd > 0.0
    return float(nd) / float(ebitda) > float(hard) + 1e-12


def covenant_icr_breach(ebit: float, interest: float, floor: float = ICR_COVENANT) -> bool:
    """True when EBIT / interest is below ``floor``. ``interest`` is cr / year."""
    if interest <= 0.0:
        return False
    return float(ebit) / float(interest) + 1e-12 < float(floor)


def drawable(book: LineBook) -> tuple[float, float, float]:
    """``(committed, uncommitted, total)`` still available (cr)."""
    committed = max(float(book.committed_cr), 0.0)
    uncommitted = max(float(book.uncommitted_cr), 0.0) * max(float(book.lam), 0.0)
    return committed, uncommitted, committed + uncommitted


def refinance_wall(
    loan: TermLoan,
    *,
    cash: float,
    book: LineBook,
    other_bills: list[Payment] | None = None,
) -> BudgetResult:
    """Mature the loan and try to refinance. Closed uncommitted gate → distress.

    Principal is a ``principal`` bill. Drawable cash is ``cash + committed +
    uncommitted*lam``. When ``lam = 0`` only the committed line remains.
    """
    due = principal_due(loan)
    _c, _u, undrawn = drawable(book)
    bills = list(other_bills or ())
    if due > 0.0:
        bills.append(Payment("principal", due))
    return scale_payments(float(cash), undrawn, bills)


def post_loan_draw(ledger: Any, borrower: str, amount: float, *, tick: int) -> None:
    """Bank draws ``amount`` cr: DEP + / LOAN + on the borrower vs BANKSYS."""
    from marketsim.ledger.journal import Entry, Ledger, Tx
    from marketsim.ledger.sfc import assert_consistent

    if amount == 0.0:
        return
    assert isinstance(ledger, Ledger)
    if "BANKSYS" not in ledger.entities:
        ledger.register_entity("BANKSYS")
    if borrower not in ledger.entities:
        ledger.register_entity(borrower)
    amt = float(amount)
    ledger.post(
        Tx(
            int(tick),
            "loan_new",
            (
                Entry(borrower, "DEP", amt),
                Entry(borrower, "LOAN", -amt),
                Entry("BANKSYS", "DEP", -amt),
                Entry("BANKSYS", "LOAN", amt),
            ),
            memo="term draw",
        )
    )
    assert_consistent(ledger)


def post_loan_interest(ledger: Any, borrower: str, amount: float, *, tick: int) -> None:
    """Interest (cr) DEP borrower → BANKSYS. Tag ``interest_loans``."""
    from marketsim.ledger.journal import Entry, Ledger, Tx
    from marketsim.ledger.sfc import assert_consistent

    if amount == 0.0:
        return
    assert isinstance(ledger, Ledger)
    ledger.post(
        Tx(
            int(tick),
            "interest_loans",
            (Entry(borrower, "DEP", -float(amount)), Entry("BANKSYS", "DEP", float(amount))),
            memo="term coupon",
        )
    )
    assert_consistent(ledger)
