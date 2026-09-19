from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.economy import RealEconomy
from marketsim.real.shocks import ShockBus, ar1_rho, mid_month_weight


def test_persistence_half_life_matches_persistence_q(cfg) -> None:
    pq = cfg.edges.shocks["demand"].persistence_q
    rho = ar1_rho(pq)
    half_life = float(np.log(0.5) / np.log(rho))
    assert half_life == pytest.approx(3.0 * pq * np.log(2.0), abs=1e-12)

    bus = ShockBus.from_config(cfg)
    bus.inject("demand", 0.10, persistence_q=pq, day=0)
    z0 = bus.states["dem"]
    n = 12
    for _ in range(n):
        bus.decay()
    assert bus.states["dem"] == pytest.approx(z0 * rho**n, abs=1e-12)
    assert abs(z0 * rho**half_life - 0.5 * z0) < 1e-12


def test_cost_push_mask_hits_only_targets(cfg) -> None:
    bus = ShockBus.from_config(cfg)
    bus.inject("cost_push", 0.30, day=0)
    codes = list(cfg.codes)
    cost = bus.states["cost"]
    assert cost[codes.index("ENERGY")] == pytest.approx(0.30, abs=1e-12)
    assert cost[codes.index("AGRIFOOD")] == pytest.approx(0.09, abs=1e-12)
    for i, code in enumerate(codes):
        if code not in ("ENERGY", "AGRIFOOD"):
            assert cost[i] == 0.0
    assert bus.states["imp"] == pytest.approx(0.30, abs=1e-12)


def test_supply_mask_hits_only_named_targets(cfg) -> None:
    bus = ShockBus.from_config(cfg)
    bus.inject("supply", -0.03, targets=["ENERGY"], day=0)
    codes = list(cfg.codes)
    sup = bus.states["sup"]
    assert sup[codes.index("ENERGY")] == pytest.approx(-0.03, abs=1e-12)
    assert np.count_nonzero(sup) == 1


def test_catastrophe_conserves_money(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
    targets = ("CONSTRUCT", "REALESTATE", "AUTOS")
    claims_to = ("CONSTRUCT", "AUTOS", "HEALTH")
    insurer = "NPC:0:INSURANCE"
    before_ins = eco.ledger.position(insurer, "DEP")
    before_claimants = {c: eco.ledger.position(f"NPC:0:{c}", "DEP") for c in claims_to}
    k_before = {c: float(eco.k[eco.codes.index(c)]) for c in targets}

    claims = eco.inject("catastrophe", 0.05, targets=list(targets), tick=1)

    for c in targets:
        assert eco.k[eco.codes.index(c)] == pytest.approx(0.95 * k_before[c], rel=1e-12)
    outflow = before_ins - eco.ledger.position(insurer, "DEP")
    inflow = sum(eco.ledger.position(f"NPC:0:{c}", "DEP") - before_claimants[c] for c in claims_to)
    assert outflow == pytest.approx(claims, abs=1e-9)
    assert inflow == pytest.approx(claims, abs=1e-9)
    assert outflow == pytest.approx(inflow, abs=1e-9)


def test_mid_month_weighting() -> None:
    assert mid_month_weight(0) == pytest.approx(1.0)
    assert mid_month_weight(7) == pytest.approx((21 - 7) / 21)
    assert mid_month_weight(21) == pytest.approx(0.0)


def test_mid_month_inject_scales_state(cfg) -> None:
    bus = ShockBus.from_config(cfg)
    bus.inject("fiscal", 0.05, day=7)
    assert bus.states["fisc"] == pytest.approx(0.05 * (21 - 7) / 21, abs=1e-12)
