from __future__ import annotations

import math

import pytest

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy
from marketsim.real.policy.authority import PolicyDecision


@pytest.mark.validation
def test_extreme_policy_finite_sfc_and_signs(config_dir) -> None:
    cfg = load_config(config_dir, {"dynamics.banks.mode": "full"})
    eco = RealEconomy(cfg, load_io(resolve_io_path(cfg)), pi_star=0.0, check_sfc=True)
    g0 = float(eco.real.flat(eco.real.G0).sum())
    eco.policy.govt.submit(
        PolicyDecision(
            source="agent",
            purchases_level=g0 + 0.10 * eco.fin.gdp0,
            tau_y=0.0,
            tau_c=0.0,
            fiscal_rule_on=False,
        ),
        month=0,
        lag_m=0,
    )
    eco.policy.cenbank.submit(PolicyDecision(source="agent", rate=0.0), month=0, lag_m=0)
    debt0 = eco.b
    for m in range(120):
        rec = eco.step_month()
        assert math.isfinite(rec["gdp"])
        assert math.isfinite(rec["cpi"])
        assert math.isfinite(rec["U"])
        if m == 11:
            assert rec["cpi"] > 1.0
            assert rec["gdp"] > eco.fin.gdp0
            assert eco.b > debt0
    assert eco.b > debt0
    assert eco.cb.r == pytest.approx(0.0, abs=1e-12)
    assert eco.ledger.position("GOVT", "DEP") >= -1e-9
