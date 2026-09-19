from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.residential import ResidentialBlock, residential_desired
from marketsim.real.steady_state import compute_real_baseline


def test_baseline_is_20_percent_of_i(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    block = ResidentialBlock(cfg, real)
    assert block.share == pytest.approx(0.20)
    assert real.res0 == pytest.approx(0.20 * real.I0, abs=1e-12)
    assert block.step(1.0, 0.0) == pytest.approx(real.res0, abs=1e-9)


def test_income_elasticity_one(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    assert cfg.dynamics.residential.income_elasticity == pytest.approx(1.0)
    got = residential_desired(real.res0, 1.10, 0.0, 1.0, -4.0)
    assert got == pytest.approx(1.10 * real.res0, rel=1e-12)


def test_100bp_cuts_four_percent_after_lag(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    block = ResidentialBlock(cfg, real)
    target = residential_desired(real.res0, 1.0, 1.0, 1.0, -4.0)
    assert target / real.res0 == pytest.approx(np.exp(-0.04), rel=1e-12)
    out = real.res0
    for _ in range(36):
        out = block.step(1.0, 1.0)
    assert out == pytest.approx(target, rel=1e-4)
    assert out / real.res0 == pytest.approx(np.exp(-0.04), rel=1e-3)
