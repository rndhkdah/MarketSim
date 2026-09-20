"""Abuse controls and optional regulator (T5.14 / T8.08 / §5.7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from marketsim.firms.firm import Firm, FirmsFile
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent

MergerVerdict = Literal["allow", "block"]


@dataclass
class Violation:
    control: str
    detail: str


def check_firm(
    firm: Firm,
    cfg: FirmsFile,
    *,
    cell_share: dict[tuple[str, str], float],
    decision_size: float,
    cash: float,
    undrawn: float,
) -> list[Violation]:
    """Return every control that fires on this constructed state.

    ``cell_share`` values are dimensionless cell shares; ``decision_size`` is a
    share of cell; ``cash`` and ``undrawn`` are cr.
    """
    hits: list[Violation] = []
    if cash < -undrawn - 1e-12:
        hits.append(Violation("hard_budget", f"cash {cash} < -undrawn {undrawn}"))
    if decision_size > cfg.controls.decision_size_cap + 1e-12:
        hits.append(Violation("decision_size", f"{decision_size}"))
    for cell, share in cell_share.items():
        if share > cfg.controls.cell_share_cap + 1e-12:
            hits.append(Violation("cell_share_cap", f"{cell} {share}"))
    if firm.employees < 0 or any(p.capacity < 0 for p in firm.plants):
        hits.append(Violation("nonneg", "negative stock"))
    return hits


def check_leverage(drawn: float, ebitda: float, cfg: FirmsFile) -> Violation | None:
    if ebitda <= 0:
        return Violation("leverage_cap", "non-positive ebitda with debt") if drawn > 0 else None
    if drawn / ebitda > cfg.financing.nd_ebitda_hard + 1e-12:
        return Violation("leverage_cap", f"ND/EBITDA {drawn / ebitda}")
    return None


def fine_govt(ledger: Ledger, firm_entity: str, amount: float, *, tick: int) -> None:
    """Regulator fine is a balanced DEP posting FIRM→GOVT.

    ``amount`` is cr; ``tick`` is days. Tag ``fees``. SFC-checked.
    """
    if "GOVT" not in ledger.entities:
        ledger.register_entity("GOVT")
    ledger.post(
        Tx(tick, "fees", (Entry(firm_entity, "DEP", -amount), Entry("GOVT", "DEP", amount)), memo="regulator")
    )
    assert_consistent(ledger)


def apply_fine(ledger: Ledger, firm_entity: str, amount: float, *, tick: int) -> None:
    """Apply a regulator fine (cr) via :func:`fine_govt` (SFC ledger posting).

    ``amount`` is cr; ``tick`` is days. Does not invent a fiscal shock type.
    """
    fine_govt(ledger, firm_entity, amount, tick=tick)


def merger_control(acquirer_share: float, target_share: float, cap: float) -> MergerVerdict:
    """Allow a merger iff combined cell share does not exceed ``cap``.

    ``acquirer_share``, ``target_share`` and ``cap`` are dimensionless cell
    shares. Quantity limit, not a price (D14). Does not move operating control
    — that remains :mod:`marketsim.equity.control` (ownership > 0.50 at the
    next month-end).
    """
    combined = acquirer_share + target_share
    if combined > cap + 1e-12:
        return "block"
    return "allow"
