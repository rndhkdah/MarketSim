from __future__ import annotations

import numpy as np
import pytest

from vic3_compare import (
    PRICE_CEILING,
    PRICE_FLOOR,
    allocate,
    stock_loop,
    vic3_price,
)


@pytest.mark.validation
def test_price_rule_25_175() -> None:
    target = 80.0
    for stock in (0.0, 20.0, 80.0, 160.0, 400.0):
        p = vic3_price(stock, target)
        assert PRICE_FLOOR <= p <= PRICE_CEILING
    assert vic3_price(0.0, target) == PRICE_CEILING
    assert vic3_price(target, target) == pytest.approx(1.0)


@pytest.mark.validation
def test_saturation_at_2x() -> None:
    assert vic3_price(200.0, 100.0) == PRICE_FLOOR


@pytest.mark.validation
def test_availability_substitution() -> None:
    q = allocate(np.array([40.0, 60.0]), np.array([0.1, 1.0]))
    assert q[1] > q[0]
    assert q.sum() == pytest.approx(100.0)


@pytest.mark.validation
def test_undamped_diverges_damped_stabilises() -> None:
    target = 100.0
    for gain in (0.25, 0.5, 1.0, 1.5, 2.5):
        wild = stock_loop(target, gain, decay=0.0, steps=80)
        assert (not np.isfinite(wild).all()) or np.max(np.abs(wild - target)) > 5 * target
        calm = stock_loop(target, gain, decay=0.10, steps=220)
        assert np.isfinite(calm).all()
        assert np.max(np.abs(calm[-20:] - target)) < 0.25 * target
