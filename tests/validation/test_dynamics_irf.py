from __future__ import annotations

import numpy as np
import pytest

from marketsim.scenarios.irf import run_irf

MODES = {
    "passthrough": {"dynamics.banks.mode": "passthrough"},
    "banks": {"dynamics.banks.mode": "full"},
    "credit": {"dynamics.banks.mode": "full", "dynamics.credit.gate_enabled": True},
}


def _irf(config_dir, kind, size, mode: str, persistence_q=None, months=240):
    return run_irf(
        kind,
        size,
        persistence_q,
        months,
        config_dir=config_dir,
        pi_star=0.0,
        overrides=MODES[mode],
    )


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
                reason="QUESTIONS T2.23: stub V_RE collateral flips demand 24m sum with gate on",
            ),
        ),
    ],
)
def test_demand_signs(config_dir, mode: str) -> None:
    r = _irf(config_dir, "demand", 0.02, mode)
    assert float(r.gap[0:24].sum()) > 0
    assert float(r.lvl[17:36].mean()) > 0


@pytest.mark.validation
@pytest.mark.parametrize("mode", list(MODES))
def test_monetary_signs(config_dir, mode: str) -> None:
    r = _irf(config_dir, "monetary", 0.01, mode)
    assert float(r.gap[0:36].sum()) < 0
    assert float(r.lvl[17:36].mean()) < 0


@pytest.mark.validation
@pytest.mark.parametrize("mode", list(MODES))
def test_cost_push_signs(config_dir, mode: str) -> None:
    r = _irf(config_dir, "cost_push", 0.30, mode)
    assert float(r.gap[5:48].sum()) < 0
    assert float(r.lvl[5:24].mean()) > 0


@pytest.mark.validation
@pytest.mark.parametrize("mode", list(MODES))
def test_supply_signs(config_dir, mode: str) -> None:
    r = _irf(config_dir, "supply", -0.03, mode)
    assert float(r.gap[0:48].sum()) < 0
    assert float(r.lvl[5:36].mean()) > 0


@pytest.mark.validation
@pytest.mark.parametrize("mode", list(MODES))
def test_fiscal_signs(config_dir, mode: str) -> None:
    r = _irf(config_dir, "fiscal", 0.05, mode)
    assert float(r.gap[0:24].sum()) > 0
    assert float(r.lvl[17:36].mean()) > 0


@pytest.mark.validation
@pytest.mark.parametrize("mode", list(MODES))
def test_row_signs(config_dir, mode: str) -> None:
    r = _irf(config_dir, "row", -0.10, mode)
    assert float(r.gap[0:24].sum()) < 0


@pytest.mark.validation
@pytest.mark.parametrize("mode", list(MODES))
def test_risk_appetite_signs(config_dir, mode: str) -> None:
    r = _irf(config_dir, "risk_appetite", 0.01, mode)
    assert float(r.gap[0:24].sum()) < 0


@pytest.mark.validation
@pytest.mark.xfail(
    strict=True,
    reason="QUESTIONS T2.23: same-month claims FD raises gap[1..5]; spec wants gap[1..6]<0",
)
@pytest.mark.parametrize("mode", list(MODES))
def test_catastrophe_signs(config_dir, mode: str) -> None:
    r = _irf(config_dir, "catastrophe", 0.05, mode)
    assert np.all(r.gap[0:6] < 0)
    cons = r.sector_gap("CONSTRUCT")
    assert np.any(cons[0:24] > 0)
    assert float(r.lvl[2:18].mean()) > 0
