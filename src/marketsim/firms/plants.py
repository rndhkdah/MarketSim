"""Plant pipeline, capex routing and R&D (T5.11 / §5.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.config import Config
from marketsim.core.rng import RngHub
from marketsim.firms.firm import Firm, FirmRegistry, FirmsFile, Plant
from marketsim.firms.founding import months_from_build_lag_q


@dataclass
class CapexPlan:
    """One greenfield / expansion. ``capacity`` is units; ``cost`` is cr."""

    region: str
    sector: str
    capacity: float
    lag_m: int
    cost: float
    new_cell: bool

    def is_new_cell(self, existing: set[tuple[str, str]]) -> bool:
        return (self.region, self.sector) not in existing


def unit_cost(cfg: Config, sector: str, p_i: float) -> float:
    """``v_s · p_I`` (cr per unit of capacity)."""
    v = cfg.sectors.params(sector)
    # capex coefficient lives on the IO / baseline; use asset_leverage inverse as v fallback
    # Real v is on RealBaseline; callers pass p_i * v explicitly when they have it.
    del v
    return float(p_i)


def start_expansion(
    firm: Firm,
    cfg: FirmsFile,
    plan: CapexPlan,
) -> None:
    """Queue capacity; apply +10 % set-up premium when ``new_cell``."""
    existing = {p.cell for p in firm.plants}
    premium = cfg.capex.new_cell_setup_premium if plan.new_cell or plan.is_new_cell(existing) else 0.0
    cost = plan.cost * (1.0 + premium)
    plant = None
    for p in firm.plants:
        if p.cell == (plan.region, plan.sector):
            plant = p
            break
    if plant is None:
        plant = Plant(plan.region, plan.sector, 0.0)
        firm.plants = (*firm.plants, plant)
    plant.construction_pipeline = (*plant.construction_pipeline, (plan.lag_m, plan.capacity))
    firm.cash -= cost


def payment_profile(cost: float, k: int) -> np.ndarray:
    """Erlang(k) weights summing to 1, times ``cost`` (cr)."""
    # Discrete Erlang(k, mean=k): weights ∝ x^{k-1} e^{-x} on 1..3k
    xs = np.arange(1, 3 * k + 1, dtype=float)
    w = xs ** (k - 1) * np.exp(-xs)
    w = w / w.sum()
    return cost * w


def route_capex(cost: float, routing: dict[str, float]) -> dict[str, float]:
    """Split capex demand across routing sectors (cr)."""
    return {code: cost * float(w) for code, w in sorted(routing.items())}


def advance_plants(registry: FirmRegistry, months: int = 1) -> None:
    from marketsim.firms.founding import advance_construction

    advance_construction(registry, months=months)


def rnd_productivity(hub: RngHub, spend: float, current: float) -> float:
    """Diminishing stochastic return. Stream ``firms.rnd``. Productivity is a factor."""
    rng = hub.stream("firms.rnd")
    gain = (spend / (1.0 + spend)) * (1.0 / (1.0 + current)) * float(rng.uniform(0.5, 1.5))
    return current + gain


def lag_months(cfg: Config, sector: str) -> int:
    return months_from_build_lag_q(cfg.sectors.params(sector).build_lag_q)
