"""Expected policy path and the 10-year yield (§6.3 curve).

The monthly decay ``λ`` is such that a policy-rate impulse averages into ``y10`` at
Layer 1's ``DISC_PASS`` (``BetasParams.disc_pass = 0.35`` in ``layer1/betas.py``):
``mean(λ^h) = (1 − λ^120) / (120 · (1 − λ)) ≈ 0.3524``. The curve *reproduces*
that pass-through; it does not import or assert the Layer-1 constant.

T6.25's debt/QE term-premium formula is out of scope here: ``tp`` is exogenous
or a coefficient-only AR(1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# §6.3 monthly decay of E[r_{t+h}] toward r_n + π*.
LAMBDA_M = 0.978
# §6.3 10-year average: h = 0 .. 119 (120 months).
HORIZON_M = 120


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
