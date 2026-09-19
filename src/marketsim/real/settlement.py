"""R9 income settlement: netted ledger postings in passthrough mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.errors import SFCError
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.opening import GOVT_MIX
from marketsim.ledger.sfc import assert_consistent, net_financial_assets
from marketsim.real.government import post_bond_issue
from marketsim.real.steady_state import RealBaseline


@dataclass
class MonthFlows:
    """Nominal cr/month flows used for R9 postings (passthrough)."""

    p: np.ndarray
    sales: np.ndarray
    c_nom: np.ndarray
    g_nom: np.ndarray
    i_nom: np.ndarray
    res_nom: float
    ex_nom: np.ndarray
    imp_nom: np.ndarray
    deliv_nom: np.ndarray
    wages: np.ndarray
    transfers: float
    income_tax: float
    corp_tax: np.ndarray
    interest_loans: np.ndarray
    interest_bonds: float
    dividends: np.ndarray
    d_debt: np.ndarray
    deficit: float
    vat: float = 0.0


def _pay(tick: int, tag: str, payer: str, payee: str, amount: float, inst: str = "DEP") -> Tx | None:
    if abs(amount) < 1e-14:
        return None
    return Tx(tick, tag, (Entry(payer, inst, -amount), Entry(payee, inst, amount)))


def _npc(code: str, region: int = 0) -> str:
    return f"NPC:{region}:{code}"


def settle_month(
    ledger: Ledger,
    real: RealBaseline,
    flows: MonthFlows,
    *,
    tick: int,
    clip_debt: bool = False,
    region: int = 0,
) -> None:
    """Post one month of passthrough flows. ``clip_debt`` reproduces the prototype bug."""
    hh = f"HH:{region}"
    codes = real.codes

    def post(tx: Tx | None) -> None:
        if tx is not None:
            ledger.post(tx)

    for i, code in enumerate(codes):
        firm = _npc(code, region)
        post(_pay(tick, "wages", firm, hh, float(flows.wages[i])))
        post(_pay(tick, "dividends", firm, hh, float(flows.dividends[i])))
        post(_pay(tick, "interest_loans", firm, hh, float(flows.interest_loans[i])))
        post(_pay(tick, "corp_tax", firm, "GOVT", float(flows.corp_tax[i])))
        post(_pay(tick, "consumption", hh, firm, float(flows.c_nom[i])))
        post(_pay(tick, "govt_purchases", "GOVT", firm, float(flows.g_nom[i])))
        post(_pay(tick, "exports", "ROW", firm, float(flows.ex_nom[i])))
        post(_pay(tick, "imports", firm, "ROW", float(flows.imp_nom[i])))
    post(_pay(tick, "transfers", "GOVT", hh, float(flows.transfers)))
    post(_pay(tick, "income_tax", hh, "GOVT", float(flows.income_tax)))
    post(_pay(tick, "vat", hh, "GOVT", float(flows.vat)))
    post(_pay(tick, "interest_bonds", "GOVT", hh, float(flows.interest_bonds)))
    post(_pay(tick, "residential", hh, _npc("CONSTRUCT", region), float(flows.res_nom)))

    # Intermediate: buyer j pays seller i
    for j, buy in enumerate(codes):
        for i, sell in enumerate(codes):
            amt = float(flows.deliv_nom[i, j])
            if abs(amt) < 1e-14:
                continue
            post(_pay(tick, "intermediate", _npc(buy, region), _npc(sell, region), amt))

    # Business investment: each firm pays its capex to investment-good firms via a pool on HH? 
    # Payer = firm (uses retained funds / new loans); payee = sellers in i_nom.
    # Firms' capex is financed by d_debt + retained earnings; the *purchase* is posted seller-side as i_nom.
    # Charge a single "investment" payment from each seller's customers in proportion: sellers receive i_nom.
    # Pay for that by debiting firms in proportion to max(d_debt, 0) + residual on HH (passthrough owner).
    i_total = float(np.maximum(flows.i_nom, 0.0).sum())
    if i_total > 0:
        # sellers receive
        for i, code in enumerate(codes):
            post(_pay(tick, "investment", hh, _npc(code, region), float(flows.i_nom[i])))
        # HH is reimbursed by firms (they "buy" the goods HH warehoused) — net HH investment = 0
        # except residential, already posted. Debit firms for i_total.
        weights = np.maximum(flows.d_debt, 0.0)
        if weights.sum() <= 1e-14:
            weights = np.ones(len(codes))
        weights = weights / weights.sum()
        for j, code in enumerate(codes):
            post(_pay(tick, "investment", _npc(code, region), hh, float(i_total * weights[j])))

    # Debt: HH holds firm loans
    for i, code in enumerate(codes):
        d = float(flows.d_debt[i])
        posted = max(d, 0.0) if clip_debt else d
        if clip_debt and d < 0:
            posted = 0.0
        firm = _npc(code, region)
        if posted > 1e-14:
            ledger.post(
                Tx(
                    tick,
                    "loan_new",
                    (
                        Entry(hh, "LOAN", posted),
                        Entry(firm, "LOAN", -posted),
                    ),
                )
            )
        elif posted < -1e-14:
            amt = -posted
            ledger.post(
                Tx(
                    tick,
                    "loan_repay",
                    (
                        Entry(hh, "LOAN", -amt),
                        Entry(firm, "LOAN", amt),
                    ),
                )
            )
        if clip_debt and d < -1e-14:
            # Deposit side of the repayment still happens in the investment/profit
            # posts; skipping the loan write makes Σ LOAN ≠ 0 after a later mutation.
            pass

    post_bond_issue(ledger, float(flows.deficit), tick, mix=GOVT_MIX)


def household_saving(flows: MonthFlows) -> float:
    pretax = (
        float(flows.wages.sum())
        + float(flows.dividends.sum())
        + float(flows.interest_loans.sum())
        + float(flows.interest_bonds)
        + float(flows.transfers)
    )
    yd = pretax - float(flows.income_tax)
    return yd - float(flows.c_nom.sum()) - float(flows.res_nom) - float(flows.vat)


def government_deficit(flows: MonthFlows) -> float:
    return (
        float(flows.g_nom.sum())
        + float(flows.transfers)
        + float(flows.interest_bonds)
        - float(flows.income_tax)
        - float(flows.corp_tax.sum())
        - float(flows.vat)
    )


def firm_net_borrowing(flows: MonthFlows) -> float:
    return float(flows.d_debt.sum())


def trade_balance_nom(flows: MonthFlows) -> float:
    return float(flows.ex_nom.sum() - flows.imp_nom.sum())


def nfa_map(ledger: Ledger) -> dict[str, float]:
    nfa = net_financial_assets(ledger)
    return {ledger.entities.name(i): float(nfa[i]) for i in range(len(ledger.entities))}


def _delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def assert_sector_balances(
    ledger: Ledger,
    before: dict[str, float],
    *,
    region: int = 0,
    atol: float = 1e-9,
) -> None:
    """Net-lending identity: ΔNFA_HH = −ΔNFA_GOVT − ΔNFA_firms − ΔNFA_ROW."""
    after = nfa_map(ledger)
    hh = _delta(before, after, f"HH:{region}")
    govt = _delta(before, after, "GOVT")
    row = _delta(before, after, "ROW")
    firms = 0.0
    for name in after:
        if name.startswith(f"NPC:{region}:"):
            firms += _delta(before, after, name)
    # HH saving = deficit + firm net borrowing + TB, with
    # deficit = −ΔNFA_GOVT, firm borrow = −ΔNFA_NPC, TB = −ΔNFA_ROW.
    rhs = -govt + (-firms) + (-row)
    if abs(hh - rhs) > atol:
        raise SFCError(
            f"HH saving {hh} != deficit {-govt} + borrow {-firms} + TB {-row}",
            amount=hh - rhs,
        )


def clip_firm_loans_bug(ledger: Ledger, real: RealBaseline, region: int = 0) -> None:
    """The prototype bug: ``max(debt, 0)`` wipes a firm liability with no counter-entry."""
    ii = ledger.instruments.id("LOAN")
    for code in real.codes:
        ei = ledger.entities.id(_npc(code, region))
        if ledger.pos[ei, ii] < 0.0:
            ledger.pos[ei, ii] = 0.0
            break


def settle_and_check(
    ledger: Ledger,
    real: RealBaseline,
    flows: MonthFlows,
    *,
    tick: int,
    clip_debt: bool = False,
) -> None:
    before = nfa_map(ledger)
    settle_month(ledger, real, flows, tick=tick, clip_debt=clip_debt)
    assert_sector_balances(ledger, before)
    assert_consistent(ledger, tick)
