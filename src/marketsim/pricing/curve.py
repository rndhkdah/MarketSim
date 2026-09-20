"""Expected policy path and the 10-year yield (§6.3 curve).

The monthly decay ``λ`` is such that a policy-rate impulse averages into ``y10`` at
Layer 1's ``DISC_PASS`` (``BetasParams.disc_pass = 0.35`` in ``layer1/betas.py``):
``mean(λ^h) = (1 − λ^120) / (120 · (1 − λ)) ≈ 0.3524``. The curve *reproduces*
that pass-through; it does not import or assert the Layer-1 constant.

T6.25 adds bucket cash-flow weights, the §6.11 ``tp_b`` formula, and guidance blending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# §6.3 monthly decay of E[r_{t+h}] toward r_n + π*.
LAMBDA_M = 0.978
# §6.3 10-year average: h = 0 .. 119 (120 months).
HORIZON_M = 120
# §6.11 indicative; flagged for T8.04 calibration.
KAPPA_DEBT = 0.03  # annual decimal per unit of (b − b*) on the 10-year
KAPPA_QE = 0.05  # annual decimal per unit of (h_cb − h_cb0) on the 10-year
LAMBDA_FQ = 0.01  # flight-to-quality loading on z_risk (annual decimal)
DURATION_BENCH_Y = 7.0  # years; ``D_b / 7`` in §6.11; sectors.yaml bond_index.duration


def expected_policy_path(
    r_t: float,
    r_n: float,
    pi_star: float,
    *,
    lambda_m: float = LAMBDA_M,
    horizon_m: int = HORIZON_M,
) -> np.ndarray:
    """Expected policy-rate path at monthly horizons.

    ``E[r_{t+h}] = r_n + π* + (r_t − r_n − π*) · λ^h`` for ``h = 0 .. horizon_m-1``.

    Units: ``r_t``, ``r_n``, ``π*``, and each path entry are annual decimals.
    ``lambda_m`` is the monthly decay (dimensionless). ``horizon_m`` is months.
    Returns a length-``horizon_m`` array (annual decimal).
    """
    h = np.arange(horizon_m, dtype=float)
    neutral = r_n + pi_star
    return neutral + (r_t - neutral) * (lambda_m**h)


def y10(
    r_t: float,
    r_n: float,
    pi_star: float,
    tp: float = 0.0,
    *,
    lambda_m: float = LAMBDA_M,
    horizon_m: int = HORIZON_M,
) -> float:
    """Ten-year yield: mean of the expected policy path plus term premium.

    ``y10 = mean_h E[r_{t+h}] + tp_t``. ``tp`` is added one-for-one.

    Units: ``r_t``, ``r_n``, ``π*``, ``tp``, and the return value are annual decimals.
    """
    path = expected_policy_path(r_t, r_n, pi_star, lambda_m=lambda_m, horizon_m=horizon_m)
    return float(path.mean() + tp)


@dataclass
class TermPremium:
    """AR(1) term-premium state. Units: annual decimal.

    ``tp ← φ · tp + loading · z_risk + shock``. Every coefficient is an argument
    to :meth:`step` (no implicit φ / loading). Does not implement T6.25's
    debt/QE term-premium formula.
    """

    tp: float = 0.0

    def step(self, z_risk: float, *, phi: float, loading: float, shock: float = 0.0) -> float:
        """Advance one month. ``z_risk``, ``shock``, and the return are annual decimals."""
        self.tp = phi * self.tp + loading * z_risk + shock
        return float(self.tp)

    def to_state(self) -> dict[str, Any]:
        return {"tp": float(self.tp)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TermPremium:
        return cls(tp=float(state["tp"]))


def cash_flow_weights(
    delta_annual: float,
    kappa_annual: float,
    horizon_m: int = HORIZON_M,
) -> np.ndarray:
    """Normalised decaying-coupon cash-flow weights ``ω_k`` (§6.11).

    Remaining face after ``k`` months is ``(1 − δ_m)^k``; the month-``k`` coupon
    plus redemption is ``(κ_m + δ_m)`` times remaining face. ``ω`` sums to 1.
    ``delta_annual`` / ``kappa_annual`` are 1/year and annual decimal.
    """
    if horizon_m < 1:
        raise ValueError("horizon_m must be >= 1")
    d_m = float(delta_annual) / 12.0
    k_m = float(kappa_annual) / 12.0
    face = (1.0 - d_m) ** np.arange(horizon_m, dtype=float)
    cf = (k_m + d_m) * face
    total = float(cf.sum())
    if total <= 0.0:
        raise ValueError("cash-flow weights need κ+δ > 0")
    return cf / total


def blend_expected_path(
    rule_path: np.ndarray,
    guidance_path: np.ndarray | None,
    credibility: float,
) -> np.ndarray:
    """``(1 − c) · rule + c · guidance`` (§2.14.5 / §6.11). ``c`` in ``[0, 1]``.

    ``guidance_path`` shorter than the rule is padded with the last guidance
    print; longer is truncated. ``None`` or ``c=0`` returns the rule path.
    Units: annual decimal.
    """
    rule = np.asarray(rule_path, dtype=float)
    c = min(1.0, max(0.0, float(credibility)))
    if guidance_path is None or c == 0.0:
        return rule.copy()
    g = np.asarray(guidance_path, dtype=float).reshape(-1)
    if g.size == 0:
        return rule.copy()
    if g.size < rule.size:
        g = np.pad(g, (0, rule.size - g.size), mode="edge")
    else:
        g = g[: rule.size]
    return (1.0 - c) * rule + c * g


def bucket_fair_yield(path: np.ndarray, weights: np.ndarray, tp: float = 0.0) -> float:
    """``y_b = Σ_k ω_bk · E[r_{t+k}] + tp_b``. Annual decimal."""
    p = np.asarray(path, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = min(p.size, w.size)
    return float(np.dot(w[:n], p[:n]) + tp)


def term_premium_bucket(
    duration_y: float,
    *,
    debt_to_gdp: float,
    debt_to_gdp_star: float,
    h_cb: float,
    h_cb0: float,
    z_risk: float,
    tp0: float = 0.0,
    noise: float = 0.0,
    kappa_debt: float = KAPPA_DEBT,
    kappa_qe: float = KAPPA_QE,
    lambda_fq: float = LAMBDA_FQ,
    duration_bench_y: float = DURATION_BENCH_Y,
) -> float:
    """``tp_b = tp0 + (D_b/7)·[κ_debt(b−b*) − κ_qe(h_cb−h_cb0)] − λ_fq·z_risk + noise``.

    ``duration_y`` is years; ``b`` / ``h_cb`` are face/GDP (dimensionless);
    the return is an annual decimal. Coefficients are §6.11 / T8.04 flags.
    """
    scale = float(duration_y) / float(duration_bench_y)
    fiscal = kappa_debt * (float(debt_to_gdp) - float(debt_to_gdp_star))
    qe = kappa_qe * (float(h_cb) - float(h_cb0))
    return float(tp0 + scale * (fiscal - qe) - lambda_fq * float(z_risk) + noise)

