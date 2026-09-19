from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.core.gates import asymmetric_gate
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.pricing.provider import StubAssetPriceProvider
from marketsim.real.banks import capital_ratio
from marketsim.real.credit import CreditBlock, collateral_index, write_off_bank_equity
from marketsim.real.economy import RealEconomy
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def _cfg(config_dir, **over):
    base = {"dynamics.banks.mode": "full"}
    base.update(over)
    return load_config(config_dir, base)


def test_baseline_bitwise_equal_with_gate_on(config_dir) -> None:
    cfg_off = _cfg(config_dir, **{"dynamics.credit.gate_enabled": False})
    cfg_on = _cfg(config_dir, **{"dynamics.credit.gate_enabled": True})
    io_off = load_io(resolve_io_path(cfg_off))
    io_on = load_io(resolve_io_path(cfg_on))
    a = RealEconomy(cfg_off, io_off, pi_star=0.0, check_sfc=False)
    b = RealEconomy(cfg_on, io_on, pi_star=0.0, check_sfc=False)
    for _ in range(24):
        ra, rb = a.step_month(), b.step_month()
        assert np.array_equal(ra["x"], rb["x"])
        assert ra["gdp"] == rb["gdp"]


def test_asymmetric_collateral_three_times_down() -> None:
    down = collateral_index(float(np.log(0.80)))
    up = collateral_index(float(np.log(1.20)))
    cut = 1.0 - down
    rise = up - 1.0
    assert cut / rise == pytest.approx(3.2, rel=0.15)
    assert down < 1.0 < up


def test_collateral_round_trip_no_ratchet() -> None:
    assert collateral_index(0.0) == 1.0
    assert asymmetric_gate(0.0, 0.55, 0.6, 1.8) == 1.0
    assert collateral_index(0.2) == pytest.approx(asymmetric_gate(0.2, 0.55, 0.6, 1.8), abs=1e-15)
    assert collateral_index(-0.2) == pytest.approx(asymmetric_gate(-0.2, 0.55, 0.6, 1.8), abs=1e-15)


def test_credit_block_at_baseline_is_one(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    block = CreditBlock(cfg, real, fin, StubAssetPriceProvider.from_baseline(real, fin, cfg))
    block.enabled = True
    lam = block.update(capital=0.125, r=fin.r0, pi_e=0.0, z_risk=0.0, ll_bar=block.ll_bar0)
    assert lam == pytest.approx(1.0, abs=1e-12)
    assert block.spread == pytest.approx(cfg.dynamics.credit.s0, abs=1e-12)


def test_writeoff_binds_gate_within_quarter(config_dir) -> None:
    cfg = _cfg(config_dir, **{"dynamics.credit.gate_enabled": True})
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=True)
    for _ in range(3):
        eco.step_month()
    wo = write_off_bank_equity(eco.ledger, eco.debt, eco.codes, 0.40, tick=eco.month)
    eco.debt = eco.debt - wo
    eco._refresh_bank_sheet()
    gates = []
    for _ in range(3):
        eco.step_month()
        gates.append(eco.credit.gate)
    assert min(gates) < 0.6
    assert capital_ratio(eco.ledger) < 0.125


@pytest.mark.validation
def test_credit_crunch_gate_and_recovery(config_dir) -> None:
    cfg_on = _cfg(config_dir, **{"dynamics.credit.gate_enabled": True})
    cfg_off = _cfg(config_dir, **{"dynamics.credit.gate_enabled": False})
    io = load_io(resolve_io_path(cfg_on))
    on = RealEconomy(cfg_on, io, pi_star=0.0, check_sfc=True)
    off = RealEconomy(cfg_off, io, pi_star=0.0, check_sfc=True)
    gdp0 = on.fin.gdp0
    gaps_on, gaps_off = [], []
    gate_after = []
    for t in range(156):
        if t == 12:
            wo = write_off_bank_equity(on.ledger, on.debt, on.codes, 0.40, tick=t)
            on.debt = on.debt - wo
            on._refresh_bank_sheet()
            wo_off = write_off_bank_equity(off.ledger, off.debt, off.codes, 0.40, tick=t)
            off.debt = off.debt - wo_off
            off._refresh_bank_sheet()
        on.step_month()
        off.step_month()
        gaps_on.append(on.last_agg.gdp_prod_real / gdp0 - 1.0)
        gaps_off.append(off.last_agg.gdp_prod_real / gdp0 - 1.0)
        if 12 <= t <= 15:
            gate_after.append(on.credit.gate)
        if t == 12 + 96:
            assert capital_ratio(on.ledger) > 0.11
    assert min(gate_after) < 0.6
    trough_on = min(gaps_on)
    trough_off = min(gaps_off)
    assert trough_on <= 1.3 * trough_off + 1e-15
    assert abs(gaps_on[-1]) < 0.005
