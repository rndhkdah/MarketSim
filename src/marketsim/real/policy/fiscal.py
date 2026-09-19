"""Fiscal instruments: purchases, tax wedges, transfers, rescue (D13 / §2.13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.real.government import post_bond_issue


@dataclass
class FiscalLevers:
    """Resolved GOVT levers for this month. ``None`` means autopilot default."""

    purchases_level: float | None = None
    purchases_mix: dict[str, float] | None = None
    tau_y: float | None = None
    tau_c: float | None = None
    vat: float = 0.0
    tariff: float = 0.0
    excise: dict[str, float] = field(default_factory=dict)
    benefit_replacement: float | None = None
    transfer_oneoff: float = 0.0
    capex_subsidy: dict[str, float] = field(default_factory=dict)
    rescue_banksys: float = 0.0
    debt_target: float | None = None
    kappa_debt: float | None = None
    fiscal_rule_on: bool | None = None


def levers_from_merged(merged: dict[str, Any]) -> FiscalLevers:
    """Map ``PolicyAuthority.merged()`` onto typed fiscal levers."""
    mix = merged.get("purchases_mix")
    excise = merged.get("excise") or {}
    sub = merged.get("capex_subsidy") or {}
    return FiscalLevers(
        purchases_level=merged.get("purchases_level"),
        purchases_mix=dict(mix) if mix else None,
        tau_y=merged.get("tau_y"),
        tau_c=merged.get("tau_c"),
        vat=float(merged.get("vat") or 0.0),
        tariff=float(merged.get("tariff") or 0.0),
        excise=dict(excise),
        benefit_replacement=merged.get("benefit_replacement"),
        transfer_oneoff=float(merged.get("transfer_oneoff") or 0.0),
        capex_subsidy=dict(sub),
        rescue_banksys=float(merged.get("rescue_banksys") or 0.0),
        debt_target=merged.get("debt_target"),
        kappa_debt=merged.get("kappa_debt"),
        fiscal_rule_on=merged.get("fiscal_rule_on"),
    )


def consume_oneoffs(authority: Any) -> None:
    """One-shot levers fire once, then return to autopilot."""
    for name in ("rescue_banksys", "transfer_oneoff"):
        authority.effective.pop(name, None)
        authority.source_of.pop(name, None)


def excise_array(excise: dict[str, float], codes: tuple[str, ...]) -> np.ndarray:
    return np.array([float(excise.get(c, 0.0)) for c in codes], dtype=float)


def consumer_prices(p: np.ndarray, vat: float, excise: np.ndarray) -> np.ndarray:
    """Consumer unit prices. Producer ``p`` is not modified."""
    return np.asarray(p, dtype=float) * (1.0 + float(vat) + np.asarray(excise, dtype=float))


def vat_revenue(producer_spend: float, vat: float) -> float:
    return float(vat) * float(producer_spend)


def excise_revenue(producer_spend: np.ndarray, excise: np.ndarray) -> float:
    return float(np.asarray(producer_spend, dtype=float) @ np.asarray(excise, dtype=float))


def tariff_revenue(import_value: float, rate: float) -> float:
    """``rate × import value`` (cif, before the tariff)."""
    return float(rate) * float(import_value)


def apply_purchases(
    g0: np.ndarray,
    z_fisc: float,
    level: float | None,
    mix: dict[str, float] | None,
    codes: tuple[str, ...],
) -> np.ndarray:
    """Real G by sector. ``level`` is total real G (cr/month) when set."""
    g = np.asarray(g0, dtype=float) * np.exp(z_fisc)
    if mix:
        weights = np.array([float(mix.get(c, 0.0)) for c in codes], dtype=float)
        if float(weights.sum()) > 1e-15:
            weights = weights / weights.sum()
            g = weights * float(g.sum())
    if level is not None:
        total = float(level)
        s = float(g.sum())
        g = (g * (total / s)) if s > 1e-15 else np.zeros_like(g)
    return g


def subsidised_v(v: np.ndarray, subsidy: dict[str, float], codes: tuple[str, ...]) -> np.ndarray:
    """Effective capacity cost ``v_s · (1 − subsidy_s)``."""
    rate = excise_array(subsidy, codes)
    return np.asarray(v, dtype=float) * (1.0 - np.clip(rate, 0.0, 0.95))


def _pay(tick: int, tag: str, payer: str, payee: str, amount: float) -> Tx | None:
    if abs(amount) < 1e-14:
        return None
    return Tx(tick, tag, (Entry(payer, "DEP", -amount), Entry(payee, "DEP", amount)))


def post_tariff(ledger: Ledger, amount: float, *, tick: int, firm: str) -> None:
    tx = _pay(tick, "tariff", firm, "GOVT", amount)
    if tx is not None:
        ledger.post(tx)


def post_excise(ledger: Ledger, amount: float, *, tick: int, hh: str = "HH:0") -> None:
    tx = _pay(tick, "excise", hh, "GOVT", amount)
    if tx is not None:
        ledger.post(tx)


def post_subsidy(ledger: Ledger, amounts: dict[str, float], *, tick: int, region: int = 0) -> None:
    for code, amt in amounts.items():
        tx = _pay(tick, "subsidy", "GOVT", f"NPC:{region}:{code}", float(amt))
        if tx is not None:
            ledger.post(tx)


def post_rescue_banksys(ledger: Ledger, amount: float, *, tick: int) -> None:
    """Capital injection: GOVT DEP → BANKSYS (raises bank NFA / the gate)."""
    tx = _pay(tick, "equity_issue", "GOVT", "BANKSYS", amount)
    if tx is not None:
        ledger.post(tx)


def cover_govt_shortfall(ledger: Ledger, *, tick: int) -> float:
    """If GOVT DEP is negative, issue bonds so the deposit is never left negative."""
    dep = ledger.position("GOVT", "DEP")
    if dep >= -1e-12:
        return 0.0
    issued = -float(dep)
    post_bond_issue(ledger, issued, tick)
    return issued
