from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.credit import write_off_bank_equity
from marketsim.real.economy import RealEconomy, make_real_world
from marketsim.real.policy.authority import PolicyDecision


def _eco(config_dir, **over) -> RealEconomy:
    cfg = load_config(config_dir, over or None)
    return RealEconomy(cfg, load_io(resolve_io_path(cfg)), pi_star=0.0, check_sfc=False)


@pytest.mark.validation
def test_autopilot_parity_bitwise(config_dir) -> None:
    a = make_real_world(config_dir, seed=3, pi_star=0.0, check_sfc=False)
    b = make_real_world(config_dir, seed=3, pi_star=0.0, check_sfc=False)
    a.step(21 * 24)
    b.step(21 * 24)
    assert a.state_hash() == b.state_hash()


@pytest.mark.validation
def test_bond_financed_g_multiplier_and_cpi(config_dir) -> None:
    base = _eco(config_dir)
    shocked = _eco(config_dir)
    extra = 0.01 * shocked.fin.gdp0
    level = float(shocked.real.flat(shocked.real.G0).sum()) + extra
    shocked.policy.govt.submit(PolicyDecision(source="agent", purchases_level=level), month=0, lag_m=0)
    d_gdp = 0.0
    d_g = 0.0
    for _ in range(24):
        rb = base.step_month()
        rs = shocked.step_month()
        d_gdp += rs["gdp"] - rb["gdp"]
        d_g += extra
    mult = d_gdp / d_g
    assert 0.5 <= mult <= 2.0
    assert shocked.last_agg.cpi > base.last_agg.cpi


@pytest.mark.validation
def test_income_tax_cut_raises_output(config_dir) -> None:
    base = _eco(config_dir)
    cut = _eco(config_dir)
    cut.policy.govt.submit(
        PolicyDecision(source="agent", tau_y=max(0.0, cut.tau_eff - 0.03), fiscal_rule_on=False),
        month=0,
        lag_m=0,
    )
    for _ in range(18):
        base.step_month()
        cut.step_month()
    assert cut.last_agg.gdp_prod_real > base.last_agg.gdp_prod_real


@pytest.mark.validation
def test_vat_step_and_no_permanent_inflation(config_dir) -> None:
    eco = _eco(config_dir)
    eco.policy.govt.submit(PolicyDecision(source="agent", vat=0.02), month=0, lag_m=0)
    rec = eco.step_month()
    producer = float((eco.theta * eco.p).sum())
    assert rec["cpi"] == pytest.approx(1.02 * producer, rel=1e-9)
    assert rec["cpi"] > producer
    for _ in range(23):
        eco.step_month()
    assert abs(eco.cb.pi12()) < 0.015


@pytest.mark.validation
def test_tariff_raises_import_prices_and_cpi(config_dir) -> None:
    base = _eco(config_dir)
    tar = _eco(config_dir)
    tar.policy.govt.submit(PolicyDecision(source="agent", tariff=0.10), month=0, lag_m=0)
    for _ in range(12):
        base.step_month()
        tar.step_month()
    assert tar.prices.p_imp * 1.10 > base.prices.p_imp
    assert tar.last_agg.cpi > base.last_agg.cpi
    assert tar.last_flows is not None
    assert tar.last_flows.tariff == pytest.approx(0.10 * float(tar.last_flows.imp_nom.sum()), rel=1e-8)


@pytest.mark.validation
def test_bank_recap_reopens_gate(config_dir) -> None:
    over = {"dynamics.banks.mode": "full", "dynamics.credit.gate_enabled": True}
    eco = _eco(config_dir, **over)
    for _ in range(3):
        eco.step_month()
    eq0 = eco.bank_equity
    wo = write_off_bank_equity(eco.ledger, eco.debt, eco.codes, 0.40, tick=eco.month)
    eco.debt = eco.debt - wo
    eco._refresh_bank_sheet()
    for _ in range(3):
        eco.step_month()
    crushed = eco.credit.gate
    assert crushed < 0.6
    eco.policy.govt.submit(PolicyDecision(source="agent", rescue_banksys=0.45 * eq0), month=eco.month, lag_m=0)
    for _ in range(2):
        eco.step_month()
    assert eco.credit.gate > crushed


@pytest.mark.validation
def test_capital_requirement_lowers_capacity(config_dir) -> None:
    over = {"dynamics.banks.mode": "full", "dynamics.credit.gate_enabled": True}
    eco = _eco(config_dir, **over)
    eco.policy.cenbank.submit(PolicyDecision(source="agent", capital_requirement=0.145), month=0, lag_m=0)
    eco.step_month()
    assert eco.credit.gate < 1.0
    assert eco.credit.lam < 1.0
