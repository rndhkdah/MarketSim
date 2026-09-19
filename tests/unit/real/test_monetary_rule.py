from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.cenbank import taylor_target
from marketsim.real.economy import RealEconomy
from marketsim.real.policy.monetary_rule import (
    committee_draw,
    is_meeting_month,
    meeting_months,
    reaction_target,
    rho_per_meeting,
)


def test_meeting_months_quarterly_parity() -> None:
    assert meeting_months(4) == (2, 5, 8, 11)
    for m in range(12):
        assert is_meeting_month(m, 4) == ((m + 1) % 3 == 0)


def test_hand_computed_target_three_states(cfg) -> None:
    tay = cfg.edges.policy.taylor
    r_n, pi_s = tay.r_neutral, 0.02
    t0 = reaction_target(
        r_n=r_n, pi_star=pi_s, phi_pi=1.5, phi_u=1.0, phi_y=0.0,
        pi_pol=pi_s, u=0.05, u_star=0.05, gap=0.0, kappa_rstar=0.0,
    )
    assert t0 == pytest.approx(r_n + pi_s)
    t1 = reaction_target(
        r_n=r_n, pi_star=pi_s, phi_pi=1.5, phi_u=1.0, phi_y=0.0,
        pi_pol=pi_s + 0.02, u=0.05, u_star=0.05, gap=0.0,
    )
    assert t1 == pytest.approx(r_n + pi_s + 1.5 * 0.02)
    t2 = reaction_target(
        r_n=r_n, pi_star=pi_s, phi_pi=1.5, phi_u=1.0, phi_y=0.0,
        pi_pol=pi_s, u=0.04, u_star=0.05, gap=0.0,
    )
    assert t2 == pytest.approx(r_n + pi_s + 1.0 * 0.01)
    assert taylor_target(r_n, pi_s, 1.5, 0.5, pi_s, 0.0) == pytest.approx(t0)


def test_rho_per_meeting_quarterly() -> None:
    assert rho_per_meeting(0.75, 4) == pytest.approx(0.75)
    assert rho_per_meeting(0.80, 8) == pytest.approx(0.80 ** 0.5)


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
    "policy.monetary.committee.dispersion_bp": 0.0,
    "policy.monetary.committee.projection_noise_bp": 0.0,
}


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_steady_state_exact_with_framework(config_dir, pi_star: float) -> None:
    cfg = load_config(config_dir)
    eco = RealEconomy(cfg, load_io(resolve_io_path(cfg)), pi_star=pi_star, check_sfc=False)
    assert eco.cb.framework is not None
    x0 = eco.x.copy()
    max_dev = 0.0
    for _ in range(36):
        rec = eco.step_month()
        max_dev = max(max_dev, float(np.max(np.abs(rec["x"] / x0 - 1.0))))
    assert max_dev < 1e-9
    assert eco.cb.pi12() == pytest.approx(pi_star, abs=1e-9)


def test_parity_with_quarterly_rule_bitwise(config_dir) -> None:
    cfg_old = load_config(config_dir)
    cfg_new = load_config(config_dir, PARITY)
    old = RealEconomy(cfg_old, load_io(resolve_io_path(cfg_old)), pi_star=0.0, check_sfc=False)
    new = RealEconomy(cfg_new, load_io(resolve_io_path(cfg_new)), pi_star=0.0, check_sfc=False)
    old.cb.framework = None
    old.inject("demand", 0.02)
    new.inject("demand", 0.02)
    for _ in range(36):
        ro, rn = old.step_month(), new.step_month()
        assert ro["r"] == pytest.approx(rn["r"], abs=1e-15)
        assert np.allclose(ro["x"], rn["x"], atol=1e-14)


def test_dispersion_deterministic_per_seed() -> None:
    a = [committee_draw(7, n, 25.0) for n in range(8)]
    b = [committee_draw(7, n, 25.0) for n in range(8)]
    c = [committee_draw(8, n, 25.0) for n in range(8)]
    assert a == b
    assert a != c
    assert committee_draw(7, 0, 0.0) == 0.0


@pytest.mark.validation
def test_grid_deadband_move_frequency(config_dir) -> None:
    from sweep import STOCH

    cfg = load_config(config_dir)
    eco = RealEconomy(cfg, load_io(resolve_io_path(cfg)), pi_star=0.02, check_sfc=False)
    rng = np.random.default_rng(11)
    energy = eco.codes.index("ENERGY")
    for k in ("dem", "sup", "cost", "imp"):
        eco.bus.rho[k] = 1.0
    rates = []
    z_dem = z_sup = z_cost = 0.0
    for _ in range(1200):
        z_dem = STOCH["dem"][0] * z_dem + rng.normal(0.0, STOCH["dem"][1])
        z_sup = STOCH["sup"][0] * z_sup + rng.normal(0.0, STOCH["sup"][1])
        z_cost = STOCH["cost"][0] * z_cost + rng.normal(0.0, STOCH["cost"][1])
        eco.sh["dem"] = z_dem
        eco.sh["sup"][:] = z_sup
        eco.sh["cost"][energy] = z_cost
        rec = eco.step_month()
        rates.append(rec["r"])
    dr = np.diff(rates)
    moved = np.abs(dr) > 1e-12
    n_years = 100.0
    avg_moves = float(moved.sum()) / n_years
    avg_bp = float(np.abs(dr[moved]).mean()) * 10_000 if moved.any() else 0.0
    assert 3.0 <= avg_moves <= 6.0
    assert 20.0 <= avg_bp <= 40.0
