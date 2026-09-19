from __future__ import annotations

import math

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy


def _stoch_run(config_dir, overrides: dict, months: int = 1200, seed: int = 3) -> dict[str, float]:
    from sweep import STOCH

    cfg = load_config(config_dir, overrides)
    eco = RealEconomy(cfg, load_io(resolve_io_path(cfg)), pi_star=0.02, check_sfc=False)
    rng = np.random.default_rng(seed)
    energy = eco.codes.index("ENERGY")
    for k in ("dem", "sup", "cost", "imp"):
        eco.bus.rho[k] = 1.0
    z_dem = z_sup = z_cost = 0.0
    max_gap = 0.0
    u_min, u_max = 1.0, 0.0
    for _ in range(months):
        z_dem = STOCH["dem"][0] * z_dem + rng.normal(0.0, STOCH["dem"][1])
        z_sup = STOCH["sup"][0] * z_sup + rng.normal(0.0, STOCH["sup"][1])
        z_cost = STOCH["cost"][0] * z_cost + rng.normal(0.0, STOCH["cost"][1])
        eco.sh["dem"] = z_dem
        eco.sh["sup"][:] = z_sup
        eco.sh["cost"][energy] = z_cost
        rec = eco.step_month()
        assert math.isfinite(rec["gdp"]) and math.isfinite(rec["cpi"])
        gap = rec["gdp"] / eco.fin.gdp0 - 1.0
        max_gap = max(max_gap, abs(gap))
        u_min = min(u_min, rec["U"])
        u_max = max(u_max, rec["U"])
    return {"max_abs_gap": max_gap, "u_min": u_min, "u_max": u_max}


STRATS = {
    "makeup": {
        "policy.monetary.strategy.makeup": 0.2,
        "policy.monetary.strategy.makeup_decay": 0.98,
        "policy.monetary.strategy.makeup_clip": 0.02,
    },
    "sahm": {"policy.monetary.risk_management.enabled": True},
    "fci": {"policy.monetary.financial_conditions.phi_fci": 0.5},
}


@pytest.mark.slow
@pytest.mark.validation
@pytest.mark.parametrize("name", list(STRATS))
def test_each_strategy_100y_bounded(config_dir, name: str) -> None:
    rec = _stoch_run(config_dir, STRATS[name], months=1200, seed=3)
    assert rec["max_abs_gap"] < 0.20
    assert rec["u_min"] > 0.005
    assert rec["u_max"] < 0.20


@pytest.mark.validation
def test_strategies_short_run_bounded(config_dir) -> None:
    over = {}
    for block in STRATS.values():
        over.update(block)
    rec = _stoch_run(config_dir, over, months=120, seed=3)
    assert rec["max_abs_gap"] < 0.20
    assert 0.005 < rec["u_min"] < rec["u_max"] < 0.20
