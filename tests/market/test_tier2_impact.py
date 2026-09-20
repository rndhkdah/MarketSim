"""T6.21 — Tier-2 meta-order impact (gate 3)."""

from __future__ import annotations

import numpy as np

from marketsim.market.metrics import (
    loglog_slope,
    meta_order_path,
    peak_impact,
    revert_fraction,
    round_trip_pnl,
)

# Gate 3 size grid (fraction of ADV) and horizons (days).
SIZES = (0.001, 0.003, 0.01, 0.03, 0.10, 0.30)
HORIZONS = (1, 2, 5, 10, 20)


def test_gate3_concave_sqrt_law_and_reversion() -> None:
    peaks = [peak_impact(meta_order_path(q, 5), 5) for q in SIZES]
    delta = loglog_slope(np.array(SIZES), np.array(peaks))
    assert 0.4 <= delta <= 0.7
    # Weak horizon dependence at 5 % of ADV.
    q_fixed = 0.05
    peak_h = [peak_impact(meta_order_path(q_fixed, d), d) for d in HORIZONS]
    slope_h = loglog_slope(np.array(HORIZONS, dtype=float), np.array(peak_h))
    assert 0.0 <= slope_h <= 0.25
    # Concave: doubling size less than doubles peak.
    assert peaks[-1] / peaks[0] < (SIZES[-1] / SIZES[0])
    for d in (5, 10, 20):
        path = meta_order_path(0.10, d)
        rev = revert_fraction(path, d)
        assert 0.40 <= rev <= 0.75
        assert round_trip_pnl(path, d) <= 0.0
    # Response function positive and decaying after a one-day pulse.
    pulse = meta_order_path(0.05, 1)
    assert pulse[0] > 0.0
    assert pulse[5] < pulse[0]
    assert pulse[20] < pulse[5]
