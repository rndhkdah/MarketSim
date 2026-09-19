from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.erlang import ErlangSmoother
from marketsim.real.economy import RealEconomy
from marketsim.real.edges import DEFAULT_SATURATION, TypedEdgeBlock


def _block(cfg, io) -> TypedEdgeBlock:
    return TypedEdgeBlock(cfg, io.codes)


def test_no_effect_at_baseline(cfg, io) -> None:
    block = _block(cfg, io)
    p = np.ones(len(io.codes))
    sh = block.shifters(p, 1.0)
    assert np.allclose(sh, 1.0, atol=1e-15)


def test_energy_relative_price_long_run(cfg, io) -> None:
    block = _block(cfg, io)
    codes = list(io.codes)
    p = np.ones(len(codes))
    p[codes.index("ENERGY")] = 1.2
    pc = 1.0
    for _ in range(120):
        sh = block.shifters(p, pc)
    ln = float(np.log(1.2))
    autos = float(np.exp(np.clip(-0.30 * ln, -0.3, 0.3)))
    util = float(np.exp(np.clip(0.35 * ln, -0.3, 0.3)))
    assert sh[codes.index("AUTOS")] == pytest.approx(autos, rel=1e-4)
    assert sh[codes.index("UTILITIES")] == pytest.approx(util, rel=1e-4)
    assert sh[codes.index("AUTOS")] == pytest.approx(0.9468, rel=0.01)
    assert sh[codes.index("UTILITIES")] == pytest.approx(1.066, rel=0.01)


def test_erlang_lag_profile(cfg, io) -> None:
    block = _block(cfg, io)
    codes = list(io.codes)
    p = np.ones(len(codes))
    p[codes.index("ENERGY")] = 1.2
    ln = float(np.log(1.2))
    sm = ErlangSmoother(2, 6.0, 0.0)
    autos_i = codes.index("AUTOS")
    util_i = codes.index("UTILITIES")
    for t in range(24):
        sh = block.shifters(p, 1.0)
        expected = float(sm.push(ln))
        assert sh[autos_i] == pytest.approx(np.exp(np.clip(-0.30 * expected, -0.3, 0.3)), abs=1e-12)
        assert sh[util_i] == pytest.approx(np.exp(np.clip(0.35 * expected, -0.3, 0.3)), abs=1e-12)
        if t == 0:
            assert abs(sh[autos_i] - 1.0) < abs(np.exp(-0.30 * ln) - 1.0)


def test_saturation_caps(cfg, io) -> None:
    block = _block(cfg, io)
    codes = list(io.codes)
    p = np.ones(len(codes))
    p[codes.index("ENERGY")] = float(np.exp(4.0))
    for _ in range(80):
        sh = block.shifters(p, 1.0)
    cap = float(np.exp(DEFAULT_SATURATION))
    floor = float(np.exp(-DEFAULT_SATURATION))
    assert sh[codes.index("UTILITIES")] == pytest.approx(cap, rel=1e-6)
    assert sh[codes.index("AUTOS")] == pytest.approx(floor, rel=1e-6)


def test_no_effect_at_baseline_in_economy(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=True)
    x0 = eco.x.copy()
    for _ in range(12):
        eco.step_month()
        assert np.allclose(eco.last_fd_shift, 1.0, atol=1e-12)
    assert np.max(np.abs(eco.x / x0 - 1.0)) < 1e-9
