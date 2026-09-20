"""T3.08 — income-decile tiers."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.demand.tiers import build_tiers, gini, tier_real_income


def test_iota_sums_to_ten_mean_one() -> None:
    t = build_tiers()
    assert t.iota.shape == (10,)
    assert float(t.iota.sum()) == pytest.approx(10.0, abs=1e-12)
    assert float(t.iota.mean()) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(t.iota) > 0)


def test_gini_in_band() -> None:
    t = build_tiers()
    g = gini(t.iota)
    assert 0.38 <= g <= 0.42


def test_budget_shares_sum_to_one() -> None:
    t = build_tiers()
    assert float(t.budget_share.sum()) == pytest.approx(1.0, abs=1e-12)
    assert t.budget_share[-1] > t.budget_share[0]


def test_tier_income_scales() -> None:
    t = build_tiers()
    y = tier_real_income(t.iota, 1.5)
    assert y[0] == pytest.approx(1.5 * t.iota[0], abs=1e-12)
