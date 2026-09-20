"""T8.09 — regional RE index; collateral matches the traded price."""

from __future__ import annotations

import pytest

from marketsim.pricing.realestate import (
    DEFAULT_WEIGHTS,
    collateral_from_price,
    regional_prices,
    symbols,
    traded_national,
)
from marketsim.real.credit import collateral_index


def test_symbols_and_zero_xi_national_equals_v() -> None:
    assert symbols() == ("RE:CAPITAL", "RE:INDUSTRIAL", "RE:RESOURCE")
    v = 12.0
    prices = regional_prices(v)
    assert sum(DEFAULT_WEIGHTS) == pytest.approx(1.0)
    assert traded_national(v) == pytest.approx(v, abs=1e-12)
    assert prices["RE:CAPITAL"] == pytest.approx(v, abs=1e-12)
    assert prices["RE:RESOURCE"] == pytest.approx(v, abs=1e-12)


def test_collateral_channel_matches_traded_price() -> None:
    v = 10.0
    assert collateral_from_price(v, v) == pytest.approx(1.0)
    low = collateral_from_price(8.0, v)
    high = collateral_from_price(12.0, v)
    assert low < 1.0
    assert high > 1.0
    assert low == pytest.approx(collateral_index(__import__("math").log(8.0 / 10.0)))
    assert low < collateral_from_price(9.0, v)


def test_xi_moves_only_that_region() -> None:
    base = regional_prices(10.0)
    shocked = regional_prices(10.0, xi={"RE:CAPITAL": 0.1})
    assert shocked["RE:CAPITAL"] > base["RE:CAPITAL"]
    assert shocked["RE:INDUSTRIAL"] == pytest.approx(base["RE:INDUSTRIAL"])
