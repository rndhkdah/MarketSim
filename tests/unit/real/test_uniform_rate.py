from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy
from marketsim.scenarios.irf import run_irf


def _eco(config_dir, *, pricing: str, gate: bool = False, pi_star: float = 0.0, check_sfc: bool = False) -> RealEconomy:
    cfg = load_config(
        config_dir,
        {
            "dynamics.banks.mode": "full",
            "dynamics.credit.pricing": pricing,
            "dynamics.credit.gate_enabled": gate,
        },
    )
    return RealEconomy(cfg, load_io(resolve_io_path(cfg)), pi_star=pi_star, check_sfc=check_sfc)


def _rates(eco: RealEconomy) -> np.ndarray:
    return eco.cb.r + eco.spread


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
@pytest.mark.parametrize("pricing", ["uniform", "risk_based"])
def test_steady_state_exact_both_pricing(config_dir, pricing: str, pi_star: float) -> None:
    eco = _eco(config_dir, pricing=pricing, pi_star=pi_star)
    x0 = eco.x.copy()
    max_dev = 0.0
    for _ in range(36):
        rec = eco.step_month()
        max_dev = max(max_dev, float(np.max(np.abs(rec["x"] / x0 - 1.0))))
    assert max_dev < 1e-9


def test_uniform_rates_equal_in_shocked_run(config_dir) -> None:
    eco = _eco(config_dir, pricing="uniform", gate=True)
    eco.inject("demand", 0.02)
    eco.inject("monetary", 0.01)
    for _ in range(24):
        eco.step_month()
        rates = _rates(eco)
        assert float(np.ptp(rates)) < 1e-12


def test_one_sector_default_raises_all_rates_equally(config_dir) -> None:
    eco = _eco(config_dir, pricing="uniform", gate=True)
    for _ in range(3):
        eco.step_month()
    before = float(_rates(eco)[0])
    i = eco.codes.index("AUTOS")
    eco.debt[i] *= 0.4
    eco._ll_bar = eco.credit.ll_bar0 + 0.01
    for _ in range(3):
        eco.step_month()
        rates = _rates(eco)
        assert float(np.ptp(rates)) < 1e-12
    assert float(_rates(eco)[0]) > before


@pytest.mark.validation
@pytest.mark.parametrize("pricing", ["uniform", "risk_based"])
@pytest.mark.parametrize(
    "kind,size,gap_slc,sign",
    [
        ("demand", 0.02, slice(0, 24), 1),
        ("monetary", 0.01, slice(0, 36), -1),
        ("fiscal", 0.05, slice(0, 24), 1),
    ],
)
def test_section_212_signs_both_pricing(config_dir, pricing: str, kind: str, size: float, gap_slc, sign: int) -> None:
    r = run_irf(
        kind,
        size,
        months=48,
        config_dir=config_dir,
        pi_star=0.0,
        overrides={"dynamics.banks.mode": "full", "dynamics.credit.pricing": pricing},
    )
    total = float(r.gap[gap_slc].sum())
    assert total * sign > 0
