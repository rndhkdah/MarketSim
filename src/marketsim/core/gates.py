"""State-dependent edge gates (T0.14 / B3).

``logistic_gate`` is the bank-capital gate. At a capital ratio of 1/11 ≈ 0.0909
(the old ``1/asset_leverage`` definition) the *raw* logistic with midpoint 0.085
and steepness 90 evaluates to ≈ 0.63 — that is the B3 bug. The ratio is now
equity / RWA (baseline 0.125) and the gate is normalised to 1 at baseline.
"""

from __future__ import annotations

import math


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logistic_gate(
    c: float,
    *,
    midpoint: float = 0.085,
    steepness: float = 90.0,
    baseline: float = 0.125,
    normalise_at_baseline: bool = True,
) -> float:
    """``min(1, σ(k(c−mid)) / σ(k(c0−mid)))`` when normalised.

    Units: `c` is a capital ratio (equity / RWA). At the published defaults
    ``gate(0.125) = 1`` and ``gate(0.085) = 0.5 / 0.9734``.
    """
    raw = _sigmoid(steepness * (c - midpoint))
    if not normalise_at_baseline:
        return raw
    denom = _sigmoid(steepness * (baseline - midpoint))
    return min(1.0, raw / denom)


def asymmetric_gate(gap: float, elasticity: float, up: float, down: float) -> float:
    """Level-based (no ratchet): ``exp(e · (up·max(gap,0) + down·min(gap,0)))``."""
    signed = up * max(gap, 0.0) + down * min(gap, 0.0)
    return math.exp(elasticity * signed)
