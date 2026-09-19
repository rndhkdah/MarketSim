"""Banking system for ``banks.mode: full`` (§2.7, §2.2 opening)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.config import Config
from marketsim.core.gates import logistic_gate
from marketsim.layer1.io import IOTable
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.opening import GOVT_MIX, build_phase2_ledger
from marketsim.ledger.sfc import assert_consistent, net_financial_assets
from marketsim.real.steady_state import FinancialBaseline, RealBaseline

PAYOUT_FLOOR = 0.085


@dataclass
class BankSheet:
    """Opening / SS bank stocks. Units: cr (stocks)."""

    loans: float
    deposits: float
    equity: float
    reserves: float
    gb: float
    cb_gb: float
    hh_gb: float
    leverage: float


def bank_sheet(loans: float, govt_debt: float, cfg: Config) -> BankSheet:
    """§2.2: equity = 0.125·loans; assets = asset_leverage·equity; RES = 8 % of deposits."""
    assert cfg.dynamics is not None
    target = cfg.dynamics.banks.capital_target
    lev = float(cfg.sectors.params("BANKS").asset_leverage)
    equity = target * loans
    assets = lev * equity
    deposits = assets - equity
    reserves = cfg.dynamics.banks.reserves_to_deposits * deposits
    gb = assets - loans - reserves
    cb_gb = reserves
    hh_gb = govt_debt - gb - cb_gb
    if hh_gb < -1e-8:
        raise ValueError(f"government debt {govt_debt} too small for bank/CB bond holdings {gb + cb_gb}")
    return BankSheet(
        loans=float(loans),
        deposits=float(deposits),
        equity=float(equity),
        reserves=float(reserves),
        gb=float(gb),
        cb_gb=float(cb_gb),
        hh_gb=float(hh_gb),
        leverage=lev,
    )


def loss_rate(nd: np.ndarray, icr: np.ndarray, icr0: np.ndarray, ll0: float, kappa_ll: float, cap_mult: float) -> np.ndarray:
    """Annual expected-loss rate ``ll0·(nd/2.5)·exp(κ·(ICR0/ICR−1))``, capped at ``cap_mult``× base."""
    base = ll0 * (np.asarray(nd, dtype=float) / 2.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.asarray(icr) > 1e-12, np.asarray(icr0) / np.asarray(icr), 20.0)
        ratio = np.clip(ratio, 0.05, 20.0)
    raw = base * np.exp(kappa_ll * (ratio - 1.0))
    return np.minimum(raw, cap_mult * base)


def expected_loss(
    nd: np.ndarray,
    icr: np.ndarray,
    icr0: np.ndarray,
    *,
    ll0: float,
    kappa_ll: float,
    cap_mult: float,
) -> np.ndarray:
    """Annual expected-loss *rate* by firm (1/year)."""
    return loss_rate(nd, icr, icr0, ll0, kappa_ll, cap_mult)


def monthly_writeoff(rate: np.ndarray, debt: np.ndarray, g: float = 1.0) -> np.ndarray:
    """Write-off this month: ``(ll/12)·debt`` (same units as stepper interest, no ÷G)."""
    del g
    return (np.asarray(rate) / 12.0) * np.asarray(debt)


def ss_writeoff(rate: np.ndarray, debt: np.ndarray, g: float) -> np.ndarray:
    """Baseline write-off in the /G units of ``int0`` so payout + G·step match."""
    return monthly_writeoff(rate, debt) / g


def bank_payout_frac(capital: float, floor: float = PAYOUT_FLOOR, target: float = 0.125) -> float:
    """``clip((c − 0.085)/(0.125 − 0.085), 0, 1)``."""
    return float(np.clip((capital - floor) / (target - floor), 0.0, 1.0))


def capital_ratio(ledger: Ledger) -> float:
    """``equity / RWA`` with RWA = bank loans."""
    loans = float(ledger.position("BANKSYS", "LOAN"))
    eq = float(net_financial_assets(ledger)[ledger.entities.id("BANKSYS")])
    return eq / max(loans, 1e-12)


def deposit_rate(r: float, margin: float) -> float:
    """``max(r − dep_margin, 0)`` — NIM compresses at the ELB."""
    return max(r - margin, 0.0)


def steady_bank_profit(cfg: Config, io: IOTable, *, r: float) -> float:
    """Closed-form monthly bank profit at a given policy rate (SS ICR, write-offs)."""
    from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline

    real = compute_real_baseline(io, cfg)
    # Sheet and write-offs do not depend on r through W; rebuild at r via a local copy of the identities.
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    sheet = bank_sheet(float(fin.debt.sum()), fin.B, cfg)
    dyn = cfg.dynamics
    assert dyn is not None
    nd = np.array([cfg.sectors.params(c).nd_ebitda for c in real.codes])
    ll = expected_loss(nd, fin.icr0, fin.icr0, ll0=dyn.banks.ll0, kappa_ll=dyn.banks.kappa_ll, cap_mult=dyn.banks.ll_cap_mult)
    wo = float(monthly_writeoff(ll, fin.debt, 1.0).sum())
    s = float(fin.spread.mean())
    d_r = deposit_rate(r, dyn.banks.dep_margin)
    int_loans = (r + s) * sheet.loans / 12.0
    int_deps = d_r * sheet.deposits / 12.0
    int_res = r * sheet.reserves / 12.0
    int_gb = r * sheet.gb / 12.0
    return float(int_loans + int_res + int_gb - int_deps - wo)


def apply_full_initialiser(fin: FinancialBaseline, real: RealBaseline, cfg: Config) -> FinancialBaseline:
    """Rewrite W, payout, α2, τ_y so the full-bank opening is an exact real SS."""
    assert cfg.dynamics is not None
    dyn = cfg.dynamics
    nd = np.array([cfg.sectors.params(c).nd_ebitda for c in real.codes], dtype=float)
    icr0 = np.where(fin.int0 > 1e-12, fin.ebitda0 / fin.int0, 1e6)
    ll = expected_loss(nd, icr0, icr0, ll0=dyn.banks.ll0, kappa_ll=dyn.banks.kappa_ll, cap_mult=dyn.banks.ll_cap_mult)
    wo0 = ss_writeoff(ll, fin.debt, fin.G)
    prof = fin.ebitda0 - fin.int0 - fin.tax0
    payout = (prof - fin.dep0 + fin.grow * fin.debt + wo0) / prof
    sheet = bank_sheet(float(fin.debt.sum()), fin.B, cfg)
    w_hh = sheet.deposits + sheet.hh_gb
    vat = fin.vat
    c0 = float(real.flat(real.C0).sum())
    yd0 = c0 * (1.0 + vat) + real.res0 + fin.grow * w_hh
    alpha2 = (c0 * (1.0 + vat) - dyn.households.alpha1 * yd0) / w_hh
    d_r = deposit_rate(fin.r0, dyn.banks.dep_margin)
    int_deps = d_r * sheet.deposits / 12.0 / fin.G
    int_hh_b = fin.r0 * sheet.hh_gb / 12.0 / fin.G
    int_loans = float(fin.int0.sum())
    int_res = fin.r0 * sheet.reserves / 12.0 / fin.G
    int_gb = fin.r0 * sheet.gb / 12.0 / fin.G
    profit0 = int_loans + int_res + int_gb - int_deps - float(wo0.sum())
    # Retain ``grow·E`` so equity/loans stays 0.125 when stocks grow at G.
    bank_div0 = profit0 - fin.grow * sheet.equity
    firm_div = float((prof - fin.dep0 + fin.grow * fin.debt + wo0).sum())
    pretax0 = fin.wages0 + firm_div + bank_div0 + int_deps + int_hh_b + fin.transfers0
    tau_y = 1.0 - yd0 / pretax0
    fin.payout = payout
    fin.W = float(w_hh)
    fin.YD0 = float(yd0)
    fin.alpha2 = float(alpha2)
    fin.tau_y = float(tau_y)
    fin.pretax0 = float(pretax0)
    fin.div0 = firm_div
    fin.wo0 = wo0
    fin.icr0 = icr0
    fin.bank_loans = sheet.loans
    fin.bank_deposits = sheet.deposits
    fin.bank_equity = sheet.equity
    fin.bank_reserves = sheet.reserves
    fin.bank_gb = sheet.gb
    fin.cb_gb = sheet.cb_gb
    fin.hh_gb = sheet.hh_gb
    fin.bank_profit0 = float(profit0)
    fin.dep_rate0 = d_r
    return fin


def post_opening_full(ledger: Ledger, real: RealBaseline, fin: FinancialBaseline, *, region: int = 0) -> None:
    """BANKSYS holds firm loans; HH holds deposits and residual bonds; CB matches RES with GB."""
    hh = f"HH:{region}"
    entries: list[Entry] = []
    for i, code in enumerate(real.codes):
        amt = float(fin.debt[i])
        if abs(amt) < 1e-15:
            continue
        entries.append(Entry("BANKSYS", "LOAN", amt))
        entries.append(Entry(f"NPC:{region}:{code}", "LOAN", -amt))
    d = float(fin.bank_deposits)
    entries.append(Entry(hh, "DEP", d))
    entries.append(Entry("BANKSYS", "DEP", -d))
    res = float(fin.bank_reserves)
    entries.append(Entry("BANKSYS", "RES", res))
    entries.append(Entry("CB", "RES", -res))
    for inst, share in GOVT_MIX:
        entries.append(Entry("BANKSYS", inst, float(fin.bank_gb * share)))
        entries.append(Entry("CB", inst, float(fin.cb_gb * share)))
        entries.append(Entry(hh, inst, float(fin.hh_gb * share)))
        entries.append(Entry("GOVT", inst, -float(fin.B * share)))
    ledger.post(Tx(0, "opening", entries, memo="full-bank opening"))


def open_full_books(cfg: Config, real: RealBaseline, fin: FinancialBaseline) -> Ledger:
    led = build_phase2_ledger(cfg, real)
    post_opening_full(led, real, fin)
    assert_consistent(led, 0)
    return led


def bank_month_flows(
    *,
    r: float,
    debt: np.ndarray,
    nd: np.ndarray,
    ebitda: np.ndarray,
    interest: np.ndarray,
    deposits: float,
    reserves: float,
    gb: float,
    equity: float,
    grow: float,
    g: float,
    fin: FinancialBaseline,
    cfg: Config,
    gate: object | None = None,
    payout_target: float | None = None,
) -> tuple[np.ndarray, float, float, float, float]:
    """Write-offs, deposit interest, bank dividends, profit, capital ratio."""
    del gate, grow
    assert cfg.dynamics is not None
    dyn = cfg.dynamics
    icr = np.where(interest > 1e-12, ebitda / interest, 1e6)
    icr0 = fin.icr0 if fin.icr0.size == debt.size else icr
    ll = expected_loss(nd, icr, icr0, ll0=dyn.banks.ll0, kappa_ll=dyn.banks.kappa_ll, cap_mult=dyn.banks.ll_cap_mult)
    wo = monthly_writeoff(ll, debt, g)
    d_r = deposit_rate(r, dyn.banks.dep_margin)
    dep_int = d_r * deposits / 12.0
    profit = float(interest.sum() + r * reserves / 12.0 + r * gb / 12.0 - dep_int - float(wo.sum()))
    loans = float(debt.sum())
    c = equity / max(loans, 1e-12)
    target = dyn.banks.capital_target if payout_target is None else float(payout_target)
    frac = bank_payout_frac(c, target=target)
    # Retain ``(G−1)·E`` so equity grows with prices the same way loans do after G·d_debt.
    div = max(0.0, frac * profit - (g - 1.0) * equity)
    _ = logistic_gate(c)  # used by T2.20; keep the ratio live
    return wo, float(dep_int), float(div), float(profit), float(c)
