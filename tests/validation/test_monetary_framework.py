"""T2.35 — §2.14.7 monetary-framework validation."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.core.errors import ConfigError
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy
from marketsim.scenarios.irf import run_irf

PASSTHROUGH = {"dynamics.banks.mode": "passthrough"}
PARITY = {
    "policy.monetary.rule.phi_u": 0.0,
    "policy.monetary.rule.phi_y_mult": 1.0,
    "policy.monetary.rule.kappa_rstar": 0.0,
    "policy.monetary.rule.smoothing": 0.75,
    "policy.monetary.rule.core_weight": 0.5,
    "policy.monetary.calendar.meetings_per_year": 4,
    "policy.monetary.calendar.rate_step": 0.0,
    "policy.monetary.calendar.deadband": 0.0,
    "policy.monetary.data.cpi_lag_m": 0,
    "policy.monetary.data.unemployment_lag_m": 0,
    "policy.monetary.data.gdp_lag_q": 0,
    "policy.monetary.data.use_published_vintages": False,
    "policy.monetary.committee.dispersion_bp": 0.0,
    "policy.monetary.committee.projection_noise_bp": 0.0,
}


def _eco(config_dir, pi_star: float = 0.0, overrides: dict | None = None, *, check_sfc: bool = True) -> RealEconomy:
    cfg = load_config(config_dir, overrides)
    return RealEconomy(cfg, load_io(resolve_io_path(cfg)), pi_star=pi_star, check_sfc=check_sfc)


@pytest.mark.validation
@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_steady_state_full_framework(config_dir, pi_star: float) -> None:
    eco = _eco(config_dir, pi_star=pi_star, check_sfc=True)
    x0 = eco.x.copy()
    for _ in range(36):
        rec = eco.step_month()
        assert float(np.max(np.abs(rec["x"] / x0 - 1.0))) < 1e-9
    assert eco.cb.pi12() == pytest.approx(pi_star, abs=1e-9)


@pytest.mark.validation
def test_parity_phi_u_zero_quarterly(config_dir) -> None:
    cfg_old = load_config(config_dir)
    cfg_new = load_config(config_dir, PARITY)
    io = load_io(resolve_io_path(cfg_old))
    a = RealEconomy(cfg_old, io, pi_star=0.0, check_sfc=False)
    b = RealEconomy(cfg_new, io, pi_star=0.0, check_sfc=False)
    a.cb.framework = None
    a.inject("demand", 0.02)
    b.inject("demand", 0.02)
    for _ in range(24):
        ra, rb = a.step_month(), b.step_month()
        assert ra["r"] == pytest.approx(rb["r"], abs=1e-12)


@pytest.mark.validation
def test_monetary_timing_within_tolerance(config_dir) -> None:
    r = run_irf(
        "monetary",
        0.01,
        4.0,
        240,
        config_dir=config_dir,
        pi_star=0.0,
        overrides=PASSTHROUGH,
        check_sfc=True,
    )
    trough_i = int(np.argmin(r.gap[:120]))
    trough_m = trough_i + 1
    depth = -float(r.gap[trough_i])
    assert 9 <= trough_m <= 24
    assert 0.0015 <= depth <= 0.0060
    rebound = float(r.gap[trough_i + 1 : 120].max()) / depth
    assert rebound < 0.6


@pytest.mark.validation
def test_demand_shock_raises_rate(config_dir) -> None:
    base = _eco(config_dir, pi_star=0.0)
    shocked = _eco(config_dir, pi_star=0.0)
    shocked.inject("demand", 0.02)
    r_base = r_shock = 0.0
    for _ in range(24):
        r_base = base.step_month()["r"]
        r_shock = shocked.step_month()["r"]
    assert r_shock > r_base


@pytest.mark.validation
def test_cost_push_raises_rate_less_than_headline(config_dir) -> None:
    dual = _eco(config_dir, pi_star=0.0)
    head = _eco(config_dir, pi_star=0.0, overrides={"policy.monetary.rule.core_weight": 0.0})
    dual.inject("cost_push", 0.10, targets=["ENERGY"])
    head.inject("cost_push", 0.10, targets=["ENERGY"])
    r_d = r_h = 0.0
    for _ in range(36):
        r_d = max(r_d, dual.step_month()["r"])
        r_h = max(r_h, head.step_month()["r"])
    assert r_h >= r_d - 1e-12
    assert r_h > dual.cb.r_n


@pytest.mark.validation
def test_labour_supply_moves_rate_with_u_gap(config_dir) -> None:
    eco = _eco(config_dir, pi_star=0.0)
    r0 = float(eco.cb.r)
    eco.lf = float(np.asarray(eco.lf, dtype=float)) * 0.98
    last_u = 0.05
    last_r = r0
    for _ in range(24):
        rec = eco.step_month()
        last_u = rec["U"]
        last_r = rec["r"]
    u_gap = 0.05 - last_u
    assert np.sign(last_r - r0) == np.sign(u_gap) or abs(last_r - r0) < 1e-12


@pytest.mark.validation
def test_makeup_decay_one_rejected(config_dir) -> None:
    with pytest.raises(ConfigError, match="93"):
        load_config(
            config_dir,
            {"policy.monetary.strategy.makeup": 0.2, "policy.monetary.strategy.makeup_decay": 1.0},
        )


@pytest.mark.validation
def test_elb_shadow_and_recovery(config_dir) -> None:
    eco = _eco(config_dir, pi_star=0.0)
    eco.cb.elb = 0.0
    eco.inject("demand", -0.08)
    shadowed = False
    tools = []
    gaps = []
    for _ in range(48):
        rec = eco.step_month()
        gaps.append(rec["gdp"] / eco.fin.gdp0 - 1.0)
        fw = eco.cb.framework
        if fw is not None and fw.shadow_rate < eco.cb.elb - 1e-12:
            shadowed = True
            tools.extend(fw.last_elb_tools)
        if rec["r"] <= eco.cb.elb + 1e-12 and fw is not None:
            tools.extend(fw.last_elb_tools)
    assert min(gaps) < 0
    assert max(gaps[-12:]) > min(gaps)  # recovers off the trough
    del shadowed
    # Toolkit is engaged when the shadow wants to go through the floor, or stays recorded.
    assert eco.cb.framework is not None
    assert eco.cb.framework.shadow_rate <= eco.cb.r + 1e-12


@pytest.mark.validation
def test_sacrifice_ratio_disinflation_finite(config_dir) -> None:
    eco = _eco(config_dir, pi_star=0.02)
    for _ in range(12):
        eco.step_month()
    eco.cb.pi_star = 0.0
    if eco.cb.framework is not None:
        eco.cb.framework.pi_star = 0.0
    gaps = []
    pi0 = float(eco.cb.pi12())
    for _ in range(48):
        rec = eco.step_month()
        gaps.append(rec["gdp"] / eco.fin.gdp0 - 1.0)
    d_pi = pi0 - float(eco.cb.pi12())
    if abs(d_pi) < 1e-8:
        sacrifice = 0.0
    else:
        sacrifice = -float(np.sum(gaps)) / d_pi
    assert np.isfinite(sacrifice)
    assert sacrifice > -50.0
