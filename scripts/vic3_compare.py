#!/usr/bin/env python3
"""Victoria 3 goods-price / stock-loop mechanics — self-contained, no Vic3 install.

Checks A–D from T0.08:
  A. price rule stays inside 25–175 % of base
  B. saturation at 2× the target stock
  C. availability substitution
  D. undamped integrating stock loop diverges at every gain; 10 % decay stabilises
"""

from __future__ import annotations

import argparse

import numpy as np

PRICE_FLOOR = 0.25
PRICE_CEILING = 1.75
SATURATION_MULT = 2.0


def vic3_price(stock: float, target: float) -> float:
    """Vic3-style goods price index. 1.0 at target, 1.75 at 0, 0.25 at 2× target."""
    if target <= 0:
        return PRICE_CEILING
    fill = stock / target
    raw = 1.75 - 0.75 * fill
    return float(np.clip(raw, PRICE_FLOOR, PRICE_CEILING))


def allocate(demands: np.ndarray, availability: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Shift a demand basket toward goods that are actually on the shelf."""
    d = np.asarray(demands, dtype=float)
    a = np.asarray(availability, dtype=float)
    w = d * np.maximum(a, 1e-12) ** sigma
    if w.sum() <= 0:
        return d.copy()
    return d.sum() * w / w.sum()


def stock_loop(
    target: float,
    gain: float,
    decay: float = 0.0,
    steps: int = 80,
    demand: float | None = None,
) -> np.ndarray:
    """Integrating production controller (the V3 experiment that diverges without leak).

    ``production += gain * (target - stock)``; stock then decays and is restocked.
    """
    d = target if demand is None else demand
    stock = float(target)
    production = float(target)
    hist = np.empty(steps)
    for t in range(steps):
        production = production + gain * (target - stock)
        stock = (1.0 - decay) * stock + production - d
        hist[t] = stock
    return hist


def _diverges(hist: np.ndarray, target: float) -> bool:
    return bool(np.max(np.abs(hist - target)) > 5.0 * target or not np.isfinite(hist).all())


def _stable(hist: np.ndarray, target: float) -> bool:
    tail = hist[-15:]
    return bool(np.isfinite(tail).all() and np.max(np.abs(tail - target)) < 0.25 * target)


def run_checks() -> dict[str, bool]:
    target = 100.0
    prices = [vic3_price(s, target) for s in (0.0, target, 2.0 * target, 5.0 * target)]
    check_a = all(PRICE_FLOOR <= p <= PRICE_CEILING for p in prices) and prices[0] == PRICE_CEILING
    check_b = vic3_price(SATURATION_MULT * target, target) == PRICE_FLOOR
    base = np.array([50.0, 50.0])
    shifted = allocate(base, np.array([0.2, 1.0]))
    check_c = shifted[1] > shifted[0]
    gains = (0.3, 0.6, 1.0, 1.5, 2.0)
    check_d_div = all(_diverges(stock_loop(target, g, decay=0.0), target) for g in gains)
    check_d_stab = all(_stable(stock_loop(target, g, decay=0.10, steps=200), target) for g in gains)
    return {
        "A_price_band": check_a,
        "B_saturation_2x": check_b,
        "C_availability": check_c,
        "D_undamped_diverges": check_d_div,
        "D_decay10_stable": check_d_stab,
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Vic3 goods-loop comparison checks").parse_args(argv)
    results = run_checks()
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
