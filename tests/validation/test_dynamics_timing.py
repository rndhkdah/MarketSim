from __future__ import annotations

import numpy as np
import pytest

from marketsim.scenarios.irf import run_irf

MODES = {
    "passthrough": {"dynamics.banks.mode": "passthrough"},
    "banks": {"dynamics.banks.mode": "full"},
    "credit": {"dynamics.banks.mode": "full", "dynamics.credit.gate_enabled": True},
}


@pytest.mark.validation
@pytest.mark.parametrize(
    "mode",
    [
        "passthrough",
        "banks",
        pytest.param(
            "credit",
            marks=pytest.mark.xfail(
                strict=True,
                reason="QUESTIONS T2.23: credit-on monetary trough m47 / depth 0.86% vs [9,24] / 0.15–0.60%",
            ),
        ),
    ],
)
def test_monetary_hump_and_sector_order(config_dir, mode: str) -> None:
    r = run_irf(
        "monetary",
        0.01,
        4.0,
        240,
        config_dir=config_dir,
        pi_star=0.0,
        overrides=MODES[mode],
    )
    trough_i = int(np.argmin(r.gap[:120]))
    trough_m = trough_i + 1
    depth = -float(r.gap[trough_i])
    assert 9 <= trough_m <= 24
    assert 0.0015 <= depth <= 0.0060
    after = r.gap[trough_i + 1 : 120]
    rebound = float(after.max()) / depth
    assert rebound < 0.6
    autos_m = int(np.argmin(r.sector_gap("AUTOS")[:60])) + 1
    cons_m = int(np.argmin(r.sector_gap("CONSTRUCT")[:60])) + 1
    cap_m = int(np.argmin(r.sector_gap("CAPGOODS")[:60])) + 1
    assert autos_m < cons_m < cap_m
