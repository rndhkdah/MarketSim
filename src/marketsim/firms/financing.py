"""Uniform corporate rate, rating-scaled limits, tax, hard budget (T5.10 / §5.5)."""

from __future__ import annotations

from dataclasses import dataclass

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
