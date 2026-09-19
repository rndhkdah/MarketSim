from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.banks import capital_ratio, expected_loss, steady_bank_profit
from marketsim.real.economy import RealEconomy
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def _full_cfg(config_dir):
    return load_config(config_dir, {"dynamics.banks.mode": "full"})


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_gate1_with_banks_on(config_dir, pi_star: float) -> None:
    cfg = _full_cfg(config_dir)
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=pi_star, check_sfc=True)
    x0 = eco.x.copy()
    max_dev = 0.0
    for _ in range(360):
        rec = eco.step_month()
        max_dev = max(max_dev, float(np.max(np.abs(rec["x"] / x0 - 1.0))))
    assert max_dev < 1e-9
    assert eco.cb.pi12() == pytest.approx(pi_star, abs=1e-9)


def test_bank_profit_rises_with_policy_rate(config_dir) -> None:
    cfg = _full_cfg(config_dir)
    io = load_io(resolve_io_path(cfg))
    p1 = steady_bank_profit(cfg, io, r=0.01)
    p4 = steady_bank_profit(cfg, io, r=0.04)
    assert p4 > p1


def test_capital_ratio_at_baseline(config_dir) -> None:
    cfg = _full_cfg(config_dir)
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=True)
    assert capital_ratio(eco.ledger) == pytest.approx(0.125, abs=1e-9)
    eco.step_month()
    assert capital_ratio(eco.ledger) == pytest.approx(0.125, abs=1e-8)


def test_expected_loss_rises_when_icr_falls(cfg) -> None:
    nd = np.array([2.5, 5.0])
    icr0 = np.array([4.0, 4.0])
    ll_ss = expected_loss(nd, icr0, icr0, ll0=0.004, kappa_ll=1.0, cap_mult=5.0)
    ll_bad = expected_loss(nd, icr0 * 0.5, icr0, ll0=0.004, kappa_ll=1.0, cap_mult=5.0)
    assert ll_ss[0] == pytest.approx(0.004, abs=1e-12)
    assert np.all(ll_bad > ll_ss)


def test_full_mode_alpha2_positive(config_dir) -> None:
    cfg = _full_cfg(config_dir)
    io = load_io(resolve_io_path(cfg))
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.02)
    assert fin.alpha2 > 0.0
    assert fin.bank_equity == pytest.approx(0.125 * fin.bank_loans, abs=1e-9)
