"""Government purchases, transfers, taxes and the debt rule (§2.5, §2.13)."""

from __future__ import annotations

import numpy as np

from marketsim.core.config import Config
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.opening import GOVT_MIX
from marketsim.real.steady_state import FinancialBaseline, RealBaseline


def real_purchases(g0: np.ndarray, z_fisc: float = 0.0) -> np.ndarray:
    """Real government demand by sector (cr/month)."""
    return np.asarray(g0, dtype=float) * np.exp(z_fisc)


def transfers(benefit: float, w: float, lf: float, n: np.ndarray) -> float:
    """Unemployment benefits (cr/month nominal)."""
    return float(benefit * w * (lf - float(n.sum())))


def debt_ratio(b: float, gdp_nom_annual: float, g: float, real_mode: bool = False, pc: float = 1.0, pn: float = 1.0) -> float:
    """B/GDP using the prototype's nominal-drift convention ``B·g / GDP_nom_a``."""
    if real_mode:
        return (b / pc) / (gdp_nom_annual / pn)
    return b * g / gdp_nom_annual


def tax_rate_target(tau0: float, kappa: float, ratio: float, debt_to_gdp: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(np.clip(tau0 + kappa * (ratio - debt_to_gdp), lo, hi))


def step_tax_rate(tau_eff: float, tau_target: float, tau_m: float) -> float:
    return tau_eff + (tau_target - tau_eff) / tau_m


def budget_deficit(
    g_nom: float,
    transfers_nom: float,
    interest_bonds: float,
    income_tax: float,
    corp_tax: float,
    vat: float = 0.0,
) -> float:
    """GOVT financing need (cr/month). Positive = deficit."""
    return float(g_nom + transfers_nom + interest_bonds - income_tax - corp_tax - vat)


def baseline_deficit(real: RealBaseline, fin: FinancialBaseline) -> float:
    """Closed-form baseline deficit using last-month interest (÷G)."""
    g_nom = float(real.flat(real.G0).sum())
    interest = fin.r0 * fin.B / 12.0 / fin.G
    income_tax = fin.tau_y * fin.pretax0
    return budget_deficit(g_nom, fin.transfers0, interest, income_tax, float(fin.tax0.sum()), vat=0.0)


def cover_deposit_shortfall(ledger: Ledger, *, tick: int) -> float:
    """Bond-finance a negative GOVT deposit. See ``policy.fiscal.cover_govt_shortfall``."""
    from marketsim.real.policy.fiscal import cover_govt_shortfall

    return cover_govt_shortfall(ledger, tick=tick)


def post_bond_issue(ledger: Ledger, amount: float, tick: int, mix: tuple[tuple[str, float], ...] = GOVT_MIX) -> None:
    """Households absorb a government issue at par: DEP for bonds."""
    if abs(amount) < 1e-15:
        return
    entries: list[Entry] = []
    for inst, share in mix:
        amt = float(amount * share)
        entries.extend(
            [
                Entry("HH:0", inst, amt),
                Entry("GOVT", inst, -amt),
                Entry("HH:0", "DEP", -amt),
                Entry("GOVT", "DEP", amt),
            ]
        )
    ledger.post(Tx(tick, "bond_issue", entries, memo="deficit finance"))


def iterate_debt_ratio(
    *,
    b0: float,
    tau0: float,
    gdp0: float,
    pretax: float,
    g_nom: float,
    transfers0: float,
    ctax: float,
    r: float,
    kappa: float,
    debt_to_gdp: float,
    bounds: tuple[float, float],
    tau_m: float,
    months: int,
    g: float = 1.0,
) -> list[float]:
    """Toy closed economy: constant real activity, optional inflation ``g``, debt rule on B."""
    b = b0
    tau = tau0
    gdp_a = 12.0 * gdp0
    ratios = []
    for _ in range(months):
        ratio = debt_ratio(b, gdp_a, g)
        target = tax_rate_target(tau0, kappa, ratio, debt_to_gdp, bounds)
        tau = step_tax_rate(tau, target, tau_m)
        interest = r * b / 12.0
        deficit = budget_deficit(g_nom, transfers0, interest, tau * pretax, ctax)
        b += deficit
        gdp_a *= g
        ratios.append(b / gdp_a if g != 1.0 else b / (12.0 * gdp0))
    return ratios


def fiscal_step(
    cfg: Config,
    fin: FinancialBaseline,
    *,
    b: float,
    tau_eff: float,
    g_nom: float,
    transfers_nom: float,
    interest_bonds: float,
    pretax: float,
    corp_tax: float,
    vat: float,
    gdp_nom_annual: float,
    g: float,
) -> tuple[float, float, float]:
    """Update τ_y and return ``(tau_eff, deficit, tau_target)``."""
    assert cfg.dynamics is not None
    fisc = cfg.dynamics.fiscal
    ratio = debt_ratio(b, gdp_nom_annual, g)
    target = tax_rate_target(fin.tau_y, fisc.kappa_debt, ratio, fisc.debt_to_gdp, fisc.tax_rate_bounds)
    tau_new = step_tax_rate(tau_eff, target, fisc.tau_tax_m)
    deficit = budget_deficit(g_nom, transfers_nom, interest_bonds, tau_new * pretax, corp_tax, vat)
    return tau_new, deficit, target
