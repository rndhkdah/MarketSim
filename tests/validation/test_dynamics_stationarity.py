from __future__ import annotations

import numpy as np
import pytest

from marketsim.scenarios.irf import make_economy

MODES = {
    "passthrough": {"dynamics.banks.mode": "passthrough"},
    "banks": {"dynamics.banks.mode": "full"},
    "credit": {"dynamics.banks.mode": "full", "dynamics.credit.gate_enabled": True},
}


@pytest.mark.validation
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_no_shock_stationarity_360(config_dir, mode: str, pi_star: float) -> None:
    eco = make_economy(config_dir, pi_star=pi_star, overrides=MODES[mode], check_sfc=True)
    x0 = eco.x.copy()
    max_dev = 0.0
    for _ in range(360):
        rec = eco.step_month()
        max_dev = max(max_dev, float(np.max(np.abs(rec["x"] / x0 - 1.0))))
        assert eco.last_agg is not None
        assert abs(eco.last_agg.gdp_prod_real - eco.last_agg.gdp_exp_real) <= 1e-9 * max(
            abs(eco.last_agg.gdp_prod_real), 1.0
        )
    assert max_dev < 1e-9
    assert eco.cb.pi12() == pytest.approx(pi_star, abs=1e-9)
