"""Discount rates ``ρ_j = y10_real + ERP_t + adj_j`` (T6.04 / §6.3)."""

from __future__ import annotations

from marketsim.pricing.curve import y10

ERP_0 = 0.0  # §6.3 level is absorbed in PE0; ERP_t = ERP_0 + z_risk + sentiment


def y10_real(
    r_t: float,
    r_n: float,
    pi_star: float,
    tp: float = 0.0,
) -> float:
    """Real 10-year yield ``y10 − π*`` (annual decimal)."""
    return y10(r_t, r_n, pi_star, tp) - float(pi_star)


def equity_risk_premium(z_risk: float = 0.0, sentiment: float = 0.0, *, erp0: float = ERP_0) -> float:
    """``ERP_t = ERP_0 + z_risk + sentiment``. Annual decimal."""
    return float(erp0) + float(z_risk) + float(sentiment)


def discount_rate(y10_real_t: float, erp_t: float, adj: float = 0.0) -> float:
    """``ρ = y10_real + ERP + adj``. Annual decimal."""
    return float(y10_real_t) + float(erp_t) + float(adj)


def delta_rho(
    r: float,
    r_n: float,
    pi_star: float,
    *,
    z_risk: float = 0.0,
    sentiment: float = 0.0,
    tp: float = 0.0,
    r0: float | None = None,
    adj: float = 0.0,
    adj0: float = 0.0,
) -> float:
    """``ρ_t − ρ_0`` with ``r0 = r_n + π*`` at baseline. Annual decimal."""
    r_base = float(r_n + pi_star) if r0 is None else float(r0)
    rho_t = discount_rate(
        y10_real(r, r_n, pi_star, tp),
        equity_risk_premium(z_risk, sentiment),
        adj,
    )
    rho_0 = discount_rate(
        y10_real(r_base, r_n, pi_star, tp),
        equity_risk_premium(0.0, 0.0),
        adj0,
    )
    return rho_t - rho_0
