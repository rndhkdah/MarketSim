from __future__ import annotations

import math

from marketsim.core.gates import asymmetric_gate, logistic_gate


def test_gate_at_baseline_is_one() -> None:
    assert logistic_gate(0.125) == 1.0


def test_gate_at_midpoint_is_half_over_baseline_raw() -> None:
    raw_mid = 0.5
    raw_base = 1.0 / (1.0 + math.exp(-90.0 * (0.125 - 0.085)))
    assert raw_base == pytest_approx_9734()
    assert logistic_gate(0.085) == raw_mid / raw_base


def pytest_approx_9734() -> float:
    # σ(3.6) ≈ 0.9734 — documented in T0.14.
    return 1.0 / (1.0 + math.exp(-3.6))


def test_legacy_one_over_eleven_raw_is_about_0_63() -> None:
    # B3: defining the ratio as 1/asset_leverage = 1/11 ≈ 0.0909 gives raw ≈ 0.63.
    raw = 1.0 / (1.0 + math.exp(-90.0 * (1.0 / 11.0 - 0.085)))
    assert 0.60 < raw < 0.66


def test_logistic_monotone() -> None:
    xs = [0.04, 0.07, 0.085, 0.10, 0.125, 0.20]
    ys = [logistic_gate(x) for x in xs]
    assert ys == sorted(ys)


def test_asymmetric_gate_round_trip_and_down_heavier() -> None:
    up = asymmetric_gate(0.2, 0.55, 0.6, 1.8)
    down = asymmetric_gate(-0.2, 0.55, 0.6, 1.8)
    assert asymmetric_gate(0.0, 0.55, 0.6, 1.8) == 1.0
    assert down < 1.0 < up
    assert (1.0 - down) > (up - 1.0)
