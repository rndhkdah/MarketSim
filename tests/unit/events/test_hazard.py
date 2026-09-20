"""T4.04 — hazard rate, clipping, feature registry, cooldown."""

from __future__ import annotations

import math

import numpy as np
import pytest

from marketsim.core.errors import ConfigError
from marketsim.events.hazard import (
    DAYS_PER_YEAR,
    MULT_HI,
    MULT_LO,
    clip_multiplier,
    daily_hazard,
    should_fire,
)
from marketsim.events.schema import HazardSpec


def test_empirical_rate_matches_base_rate() -> None:
    spec = HazardSpec(base_rate_per_year=0.20, multipliers=[])
    rng = np.random.default_rng(0)
    years = 2000
    fires = 0
    for t in range(years * DAYS_PER_YEAR):
        if should_fire(spec, rng, tick=t):
            fires += 1
    rate = fires / years
    assert rate == pytest.approx(0.20, rel=0.10)


def test_multipliers_clipped() -> None:
    assert clip_multiplier(math.exp(8.0 * 2.0)) == MULT_HI
    assert clip_multiplier(math.exp(-8.0 * 2.0)) == MULT_LO
    spec = HazardSpec.model_validate(
        {
            "base_rate_per_year": 0.06,
            "multipliers": [{"feature": "utilisation_gap", "sector": "SEMIS", "coef": 8.0}],
        }
    )
    h_hi = daily_hazard(spec, {"utilisation_gap": {"SEMIS": 10.0}})
    h_lo = daily_hazard(spec, {"utilisation_gap": {"SEMIS": -10.0}})
    base = 0.06 / DAYS_PER_YEAR
    assert h_hi == pytest.approx(base * MULT_HI, rel=1e-12)
    assert h_lo == pytest.approx(base * MULT_LO, rel=1e-12)


def test_unknown_feature_rejected() -> None:
    spec = HazardSpec.model_validate(
        {
            "base_rate_per_year": 0.1,
            "multipliers": [{"feature": "moon_phase", "coef": 1.0}],
        }
    )
    with pytest.raises(ConfigError, match="unknown hazard feature"):
        daily_hazard(spec, {})


def test_cooldown_respected() -> None:
    spec = HazardSpec(base_rate_per_year=1.0e9, multipliers=[])
    rng = np.random.default_rng(1)
    assert should_fire(spec, rng, tick=0, last_fire_tick=None, cooldown_days=10)
    assert should_fire(spec, rng, tick=5, last_fire_tick=0, cooldown_days=10) is False
    assert should_fire(spec, rng, tick=10, last_fire_tick=0, cooldown_days=10) is True
