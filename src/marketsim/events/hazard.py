"""T4.04 — daily hazard rates and Bernoulli firing (§4.2)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from marketsim.core.errors import ConfigError
from marketsim.events.schema import HazardSpec

DAYS_PER_YEAR = 252
# §4.2: each multiplier = clip(exp(coef · feature), 0.2, 5.0)
MULT_LO = 0.2
MULT_HI = 5.0

FEATURES = frozenset(
    {
        "output_gap",
        "inflation",
        "policy_rate",
        "utilisation_gap",
        "bank_capital_ratio",
        "leverage",
        "price_fundamental_gap",
        "inventory_cover",
        "unemployment",
        "days_since_category",
    }
)


def clip_multiplier(raw: float) -> float:
    """``clip(exp(coef · x), 0.2, 5.0)`` already exponentiated."""
    return float(min(MULT_HI, max(MULT_LO, raw)))


def lookup_feature(features: Mapping[str, Any], name: str, sector: str | None) -> float:
    """Read a scalar, or a sector-keyed mapping when ``sector`` is set."""
    if name not in FEATURES:
        raise ConfigError(f"unknown hazard feature {name!r}")
    if name not in features:
        return 0.0
    val = features[name]
    if sector is not None:
        if isinstance(val, Mapping):
            return float(val.get(sector, 0.0))
        raise ConfigError(f"hazard feature {name!r} needs a sector mapping")
    if isinstance(val, Mapping):
        raise ConfigError(f"hazard feature {name!r} is sector-keyed")
    return float(val)


def daily_hazard(spec: HazardSpec, features: Mapping[str, Any] | None = None) -> float:
    """``h = base_rate/252 · Π multipliers`` per day (1/day)."""
    feats = features or {}
    h = float(spec.base_rate_per_year) / DAYS_PER_YEAR
    for m in spec.multipliers:
        if m.feature not in FEATURES:
            raise ConfigError(f"unknown hazard feature {m.feature!r}")
        x = lookup_feature(feats, m.feature, m.sector)
        h *= clip_multiplier(math.exp(float(m.coef) * x))
    return float(h)


def fire_probability(h: float) -> float:
    """``p = 1 − exp(−h)``."""
    if h <= 0.0:
        return 0.0
    return float(1.0 - math.exp(-float(h)))


def in_cooldown(tick: int, last_fire_tick: int | None, cooldown_days: int) -> bool:
    """True when ``tick`` is still inside the per-event cooldown window."""
    if last_fire_tick is None or cooldown_days <= 0:
        return False
    return int(tick) - int(last_fire_tick) < int(cooldown_days)


def should_fire(
    spec: HazardSpec,
    rng: np.random.Generator,
    *,
    features: Mapping[str, Any] | None = None,
    tick: int = 0,
    last_fire_tick: int | None = None,
    cooldown_days: int = 0,
) -> bool:
    """Cooldown gate, then Bernoulli with ``p = 1 − exp(−h)`` from stream ``events``."""
    if in_cooldown(tick, last_fire_tick, cooldown_days):
        return False
    p = fire_probability(daily_hazard(spec, features))
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    return bool(rng.random() < p)
