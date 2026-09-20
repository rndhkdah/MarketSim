"""T4.02 — compose YAML events into ``ShockBus.inject`` calls (§4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import truncnorm

from marketsim.events.schema import DistSpec, EventSpec, ShockSpec
from marketsim.real.shocks import ShockBus, mid_month_weight

DAYS_PER_MONTH = 21


def sample_dist(spec: DistSpec, rng: np.random.Generator) -> float:
    """Draw one truncated sample.

    Units: dimensionless magnitude, or days when ``low_days``/``high_days`` are set.
    ``sign`` is applied after the positive draw (lognormal / uniform / triangular / normal).
    """
    sign = float(spec.sign)
    if spec.dist == "fixed":
        return sign * float(spec.value)
    if spec.low_days is not None:
        return float(rng.uniform(float(spec.low_days), float(spec.high_days)))
    lo = float(spec.min)
    hi = float(spec.max)
    if spec.dist == "uniform":
        return sign * float(rng.uniform(lo, hi))
    if spec.dist == "triangular":
        return sign * float(rng.triangular(lo, float(spec.mode), hi))
    if spec.dist == "normal":
        mu = float(spec.mean if spec.mean is not None else spec.median or 0.0)
        sigma = float(spec.sigma)
        if sigma <= 0.0:
            return sign * float(np.clip(mu, lo, hi))
        a, b = (lo - mu) / sigma, (hi - mu) / sigma
        return sign * float(truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=rng))
    if spec.dist == "lognormal":
        median = float(spec.median)
        sigma = float(spec.sigma)
        mu = float(np.log(median))
        lo_m = max(lo, 1e-18)
        hi_m = max(hi, lo_m)
        if sigma <= 0.0:
            return sign * float(np.clip(median, lo_m, hi_m))
        a = (np.log(lo_m) - mu) / sigma
        b = (np.log(hi_m) - mu) / sigma
        z = float(truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=rng))
        return sign * float(np.exp(z))
    raise ValueError(f"unknown dist {spec.dist!r}")


def resolve_sector_targets(shock: ShockSpec, event: EventSpec) -> list[str] | None:
    """Sector mask for this shock, falling back to the event-level mask. ``None`` = default."""
    mask = shock.targets if shock.targets else event.targets
    if not mask:
        return None
    sectors = mask.get("sectors")
    if sectors is None or sectors == "all":
        return None
    return [str(c) for c in sectors]


@dataclass(frozen=True)
class AppliedShock:
    """One primitive after sampling. ``injected`` is the bus increment (already mid-month weighted)."""

    kind: str
    magnitude: float
    weight: float
    injected: float
    targets: list[str] | None


def apply_composition(
    event: EventSpec,
    bus: ShockBus,
    rng: np.random.Generator,
    *,
    day: int = 0,
    economy: Any | None = None,
    tick: int = 0,
    days_per_month: int = DAYS_PER_MONTH,
) -> list[AppliedShock]:
    """Sample each primitive and inject with weight ``(21 − d)/21`` (ShockBus mid-month)."""
    applied: list[AppliedShock] = []
    weight = mid_month_weight(day, days_per_month)
    for shock in event.composition:
        mag = sample_dist(shock.magnitude, rng)
        targets = resolve_sector_targets(shock, event)
        injected = bus.inject(
            shock.shock,
            mag,
            persistence_q=float(shock.persistence_q),
            targets=targets,
            day=day,
            days_per_month=days_per_month,
            economy=economy,
            tick=tick,
        )
        applied.append(
            AppliedShock(
                kind=shock.shock,
                magnitude=mag,
                weight=weight,
                injected=float(injected),
                targets=targets,
            )
        )
    return applied
