"""T5.11 — build lags, capex routing, deterministic R&D."""

from __future__ import annotations

import pytest

from marketsim.core.config import Config
from marketsim.core.rng import RngHub
from marketsim.firms.firm import Firm, FirmRegistry, FirmsFile, Plant, cells_from_lists
from marketsim.firms.plants import (
    CapexPlan,
    advance_plants,
    lag_months,
    payment_profile,
    rnd_productivity,
    route_capex,
    start_expansion,
)


def test_energy_lag_vs_bizsvc(cfg: Config) -> None:
    assert lag_months(cfg, "ENERGY") == 36
    assert lag_months(cfg, "BIZSVC") == 6
    cells = cells_from_lists(("RESOURCE", "CAPITAL"), ("ENERGY", "BIZSVC"))
    reg = FirmRegistry([
        Firm(id="e", operator="a", plants=(Plant("RESOURCE", "ENERGY", 0.0),)),
        Firm(id="b", operator="a", plants=(Plant("CAPITAL", "BIZSVC", 0.0),)),
    ], cells=cells)
    start_expansion(reg["e"], FirmsFile(), CapexPlan("RESOURCE", "ENERGY", 4.0, lag_months(cfg, "ENERGY"), 10.0, False))
    start_expansion(reg["b"], FirmsFile(), CapexPlan("CAPITAL", "BIZSVC", 4.0, lag_months(cfg, "BIZSVC"), 10.0, False))
    advance_plants(reg, 6)
    assert reg["b"].plants[0].capacity == pytest.approx(4.0)
    assert reg["e"].plants[0].capacity == pytest.approx(0.0)
    advance_plants(reg, 30)
    assert reg["e"].plants[0].capacity == pytest.approx(4.0)


def test_capex_lands_on_routing(cfg: Config) -> None:
    split = route_capex(100.0, cfg.edges.capex.routing)
    assert abs(sum(split.values()) - 100.0) < 1e-12
    assert set(split) == set(cfg.edges.capex.routing)
    w = payment_profile(90.0, 3)
    assert w.shape[0] == 9
    assert w.sum() == pytest.approx(90.0)


def test_rnd_deterministic_per_seed() -> None:
    a = rnd_productivity(RngHub(7), 2.0, 1.0)
    b = rnd_productivity(RngHub(7), 2.0, 1.0)
    c = rnd_productivity(RngHub(8), 2.0, 1.0)
    assert a == pytest.approx(b)
    assert a != pytest.approx(c)
