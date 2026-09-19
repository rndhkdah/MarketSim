from __future__ import annotations

import time

import numpy as np
import pytest

from marketsim.real.economy import RealEconomy, make_real_world


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_gate_stationarity_360(cfg, io, pi_star: float) -> None:
    eco = RealEconomy(cfg, io, pi_star=pi_star, check_sfc=True)
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


def test_production_equals_expenditure(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.02, check_sfc=True)
    for _ in range(24):
        eco.step_month()
        a = eco.last_agg
        assert a is not None
        rel = abs(a.gdp_prod_real - a.gdp_exp_real) / max(abs(a.gdp_prod_real), 1e-12)
        assert rel < 1e-9


def test_world_hash_deterministic(config_dir) -> None:
    hashes = []
    for _ in range(2):
        w = make_real_world(config_dir, seed=4, pi_star=0.0, check_sfc=False)
        w.step(21 * 3)
        hashes.append(w.state_hash())
    assert hashes[0] == hashes[1]


def test_save_load_mid_run(config_dir, tmp_path) -> None:
    live = make_real_world(config_dir, seed=9, pi_star=0.0, check_sfc=False)
    live.step(21 * 4)
    path = tmp_path / "eco.json"
    live.save(path)
    live.step(21 * 4)
    from marketsim.layer1.io import load_io, resolve_io_path
    from marketsim.real.economy import RealEconomy
    from marketsim.world import World

    cfg = live.cfg
    eco = RealEconomy(cfg, load_io(resolve_io_path(cfg)), pi_star=0.0, check_sfc=False)
    restored = World.load(path, config_dir, modules=[eco])
    restored.step(21 * 4)
    assert restored.state_hash() == live.state_hash()


def test_monthly_step_under_5ms(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=True)
    eco.step_month()
    t0 = time.perf_counter()
    eco.step_month()
    dt = (time.perf_counter() - t0) * 1000.0
    assert dt < 5.0


def test_published_lag(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
    eco.step_month()
    assert eco.published("gdp", 0) == pytest.approx(eco.last_agg.gdp_prod_real, abs=1e-9)
    assert eco.published("infl", 1) == pytest.approx(0.0, abs=1e-12)
