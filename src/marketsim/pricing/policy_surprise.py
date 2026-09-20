"""Policy surprise and announcement effects (T6.32 / §2.14.5, §6.3).

The market-implied path is the §6.3 decay toward neutral, blended with any
published guidance × credibility. Surprise is the announcement-day gap
``decision − implied[0]``. A path jump reprices buckets through cash-flow
weights (front moves most) and equities through the 10-year mean (the same
average the curve uses for ``y10``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from marketsim.pricing.curve import (
    HORIZON_M,
    LAMBDA_M,
    blend_expected_path,
    bucket_fair_yield,
    expected_policy_path,
)

# §6.3: index duration used when a caller does not pass a market-cap mix (years).
DEFAULT_INDEX_DURATION_Y = 10.0


def market_implied_path(
    r_published: float,
    r_n: float,
    pi_star: float,
    *,
    guidance_path: np.ndarray | None = None,
    credibility: float = 0.0,
    lambda_m: float = LAMBDA_M,
    horizon_m: int = HORIZON_M,
) -> np.ndarray:
    """Public expected policy path (annual decimal), length ``horizon_m``.

    ``r_published`` is the last announced policy rate. Guidance, when present,
    is mixed in with ``credibility`` ∈ [0, 1] (§2.14.5).
    """
    rule = expected_policy_path(
        r_published, r_n, pi_star, lambda_m=lambda_m, horizon_m=horizon_m
    )
    return blend_expected_path(rule, guidance_path, credibility)


def policy_surprise(decision: float, implied_path: np.ndarray) -> float:
    """``decision − E[r_t | public]``. Annual decimal."""
    if implied_path.size < 1:
        raise ValueError("implied_path must be non-empty")
    return float(decision) - float(implied_path[0])


def path_given_rate(
    r_t: float,
    r_n: float,
    pi_star: float,
    *,
    lambda_m: float = LAMBDA_M,
    horizon_m: int = HORIZON_M,
) -> np.ndarray:
    """Rule path if the short rate is ``r_t`` (annual decimal)."""
    return expected_policy_path(r_t, r_n, pi_star, lambda_m=lambda_m, horizon_m=horizon_m)


def bucket_yield_changes(
    path_before: np.ndarray,
    path_after: np.ndarray,
    weights: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Δy_b from a path revision. Yields are annual decimals."""
    return {
        name: bucket_fair_yield(path_after, w) - bucket_fair_yield(path_before, w)
        for name, w in weights.items()
    }


def equity_dln(
    path_before: np.ndarray,
    path_after: np.ndarray,
    duration_y: float = DEFAULT_INDEX_DURATION_Y,
) -> float:
    """Index log-price from the 10-year mean of the path. ``duration_y`` is years."""
    dy = float(np.mean(path_after) - np.mean(path_before))
    return -float(duration_y) * dy


def xi_from_path(
    path_before: np.ndarray,
    path_after: np.ndarray,
    *,
    duration_eq_y: float = DEFAULT_INDEX_DURATION_Y,
    duration_bond_y: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Log-mispricing impulse ``ξ`` for equities and (optional) bond buckets.

    Bonds: ``ξ_b = −D_b · Δy10_path`` so a hike lowers the price residual in
    the same direction as the fair-yield move. Equities: ``ξ = −D · Δȳ``.
    """
    dy10 = float(np.mean(path_after) - np.mean(path_before))
    xi_eq = -float(duration_eq_y) * dy10
    xi_b = {}
    if duration_bond_y:
        xi_b = {k: -float(d) * dy10 for k, d in duration_bond_y.items()}
    return xi_eq, xi_b


@dataclass(frozen=True)
class AnnouncementEffect:
    """Units: surprise and Δy are annual decimals; ``dln_equity`` is log-price."""

    surprise: float
    dy: dict[str, float]
    dln_equity: float
    xi_equity: float


def announcement_effect(
    *,
    r_published: float,
    decision: float,
    r_n: float,
    pi_star: float,
    weights: Mapping[str, np.ndarray],
    guidance_path: np.ndarray | None = None,
    credibility: float = 0.0,
    duration_eq_y: float = DEFAULT_INDEX_DURATION_Y,
) -> AnnouncementEffect:
    """Reprice after a rate decision. Anticipated moves leave the path unchanged."""
    implied = market_implied_path(
        r_published, r_n, pi_star, guidance_path=guidance_path, credibility=credibility
    )
    surp = policy_surprise(decision, implied)
    after = path_given_rate(decision, r_n, pi_star)
    # Fully anticipated: implied already equals the post-decision path.
    if abs(surp) < 1e-15:
        after = implied
    dy = bucket_yield_changes(implied, after, weights)
    dln = equity_dln(implied, after, duration_eq_y)
    xi_eq, _ = xi_from_path(implied, after, duration_eq_y=duration_eq_y)
    return AnnouncementEffect(surprise=surp, dy=dy, dln_equity=dln, xi_equity=xi_eq)


def ols_r_squared(y: np.ndarray, x: np.ndarray) -> float:
    """In-sample R² of ``y`` on columns of ``x`` plus a constant. Dimensionless."""
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = y.size
    if n < 2:
        return 0.0
    design = np.column_stack([np.ones(n), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot
