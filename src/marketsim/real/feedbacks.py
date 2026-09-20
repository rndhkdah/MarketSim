"""Feedback edges 4 → 1/2 (T6.19 / §6.9). Three independent switches.

(a) cost of capital → capex via ``Δerp`` and live ``ln Q``;
(b) wealth → consumption: MPC 0.03 / year on 12-month smoothed equity wealth;
(c) credit spreads → hiring: ``n*`` scaled by ``(ICR/1.5)^0.2`` when ICR < 1.5.

Each accumulator leaks (rule 15). World wiring needs a config switch — see
``QUESTIONS.md`` T6.19 (this card's Files list does not include ``economy.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# §6.9 (b): MPC on equity wealth, annual decimal → month.
MPC_WEALTH_ANN = 0.03
MPC_WEALTH_M = MPC_WEALTH_ANN / 12.0
# §6.9 (b): 12-month smoother (months).
WEALTH_TAU_M = 12.0
# §6.9 (c): interest-cover threshold and exponent.
ICR_REF = 1.5
ICR_EXP = 0.2
# Rule 15: leak of the wealth smoother toward the live mark (1/month).
WEALTH_LEAK = 1.0 / WEALTH_TAU_M


def ln_q(q: float | np.ndarray) -> np.ndarray:
    """``ln Q`` (dimensionless). ``Q`` is Tobin's Q."""
    return np.log(np.maximum(np.asarray(q, dtype=float), 1e-12))


def capex_q_term(
    q: float | np.ndarray,
    *,
    q_scale: float,
    q_tobin_coef: float,
    q_clip: float,
) -> np.ndarray:
    """§2.6 / §6.9 (a) Q add-on to the start-rate, already in ``start_rate`` units."""
    return float(q_scale) * float(q_tobin_coef) * 100.0 * np.clip(ln_q(q), -float(q_clip), float(q_clip))


def derp_to_cc_gap(derp: float) -> float:
    """``Δerp`` in annual decimal → ``cc_gap`` in 100bp (§2.6 / §6.9 (a))."""
    return float(derp) * 100.0


def smooth_wealth(level: float, mark: float, *, tau_m: float = WEALTH_TAU_M) -> float:
    """Drift-free smoother ``s ← s + (x − s)/τ`` (π*=0). ``level`` / ``mark`` in cr."""
    return float(level) + (float(mark) - float(level)) / float(tau_m)


def wealth_consumption(smoothed_equity: float, *, mpc_m: float = MPC_WEALTH_M) -> float:
    """Extra consumption (cr / month) from smoothed equity wealth."""
    return float(mpc_m) * float(smoothed_equity)


def icr_hire_scale(icr: float | np.ndarray, *, ref: float = ICR_REF, exp: float = ICR_EXP) -> np.ndarray:
    """``(ICR/1.5)^0.2`` when ICR < 1.5, else 1. Dimensionless."""
    x = np.asarray(icr, dtype=float)
    scale = np.ones_like(x, dtype=float)
    tight = x < float(ref)
    safe = np.maximum(x, 1e-12)
    scale = np.where(tight, np.power(safe / float(ref), float(exp)), scale)
    return np.clip(scale, 0.0, 1.0)


def scale_employment_target(n_star: np.ndarray, icr: float | np.ndarray) -> np.ndarray:
    """§6.9 (c): shrink ``n*`` when interest cover is below 1.5."""
    return np.asarray(n_star, dtype=float) * icr_hire_scale(icr)


@dataclass
class FeedbackState:
    """Leaking wealth smoother. Units: cr."""

    wealth_s: float = 0.0

    def step_wealth(self, mark: float) -> float:
        self.wealth_s = smooth_wealth(self.wealth_s, mark)
        return self.wealth_s

    def to_state(self) -> dict[str, Any]:
        return {"wealth_s": float(self.wealth_s)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FeedbackState:
        return cls(wealth_s=float(state.get("wealth_s", 0.0)))
