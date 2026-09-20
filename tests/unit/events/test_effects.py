"""T4.03 — extra-effect shapes, revert, ledger SFC."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.events.effects import ExtraEffects, shape_factor
from marketsim.events.schema import ExtraEffect
from marketsim.ledger.sfc import assert_consistent, net_financial_assets
from marketsim.real.economy import RealEconomy


def test_shape_profiles() -> None:
    d = 10
    assert shape_factor("step", 0, d) == 1.0
    assert shape_factor("step", d - 1, d) == 1.0
    assert shape_factor("step", d, d) == 0.0
    assert shape_factor("pulse", 0, d) == 1.0
    assert shape_factor("pulse", d, d) == 0.0
    assert shape_factor("ramp", 0, d) == pytest.approx(0.1)
    assert shape_factor("ramp", d - 1, d) == pytest.approx(1.0)
    assert shape_factor("ramp", d, d) == 0.0
    dec_mid = shape_factor("decay", 5, d)
    assert 0.0 < dec_mid < 1.0
    assert shape_factor("decay", d, d) == 0.0
    assert shape_factor("step", 0, 0) == 1.0
    assert shape_factor("step", 1, 0) == 0.0


def test_effects_revert() -> None:
    rng = np.random.default_rng(0)
    ex = ExtraEffects()
    spec = ExtraEffect(variable="world_demand", shape="step", magnitude=-0.10, duration=5)
    ex.start([spec], rng, tick=0)
    assert ex.world_demand == pytest.approx(-0.10)
    ex.apply(4)
    assert ex.world_demand == pytest.approx(-0.10)
    ex.apply(5)
    assert ex.world_demand == pytest.approx(0.0)
    assert ex.live == []


def test_want_shift_and_capex_setters() -> None:
    rng = np.random.default_rng(1)
    ex = ExtraEffects()
    extras = [
        ExtraEffect(variable="want_shift[EATING_OUT_LEISURE]", shape="step", magnitude=-0.5, duration=3),
        ExtraEffect(variable="capex_preference[SEMIS]", shape="ramp", magnitude=0.2, duration=4),
        ExtraEffect(variable="labour_supply[INDUSTRIAL]", shape="decay", magnitude=-0.1, duration=8),
        ExtraEffect(variable="sentiment", shape="pulse", magnitude=0.3, duration=2),
        ExtraEffect(variable="collateral_value", shape="step", magnitude=-0.25, duration=2),
        ExtraEffect(variable="link_capacity[CAPITAL,INDUSTRIAL]", shape="pulse", magnitude=-0.5, duration=6),
        ExtraEffect(variable="link_cost[CAPITAL,INDUSTRIAL]", shape="step", magnitude=0.1, duration=6),
        ExtraEffect(variable="import_price[ENERGY]", shape="step", magnitude=0.2, duration=4),
    ]
    ex.start(extras, rng, tick=0)
    assert ex.want_shift["EATING_OUT_LEISURE"] == pytest.approx(-0.5)
    assert ex.capex_pref["SEMIS"] == pytest.approx(0.05)  # ramp age 0 → 1/4
    assert ex.labour_mult["INDUSTRIAL"] == pytest.approx(0.9)
    assert ex.sentiment == pytest.approx(0.3)
    assert ex.collateral_mult == pytest.approx(0.75)
    assert ex.link_capacity_mult["CAPITAL,INDUSTRIAL"] == pytest.approx(0.5)
    ex.apply(6)
    assert ex.want_shift == {}
    assert ex.capex_pref == {}
    assert ex.collateral_mult == pytest.approx(1.0)
    assert ex.sentiment == pytest.approx(0.0)


def test_bank_equity_and_vat_sfc(config_dir, io) -> None:
    from marketsim.core.config import load_config

    cfg = load_config(config_dir, {"dynamics.banks.mode": "full"})
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=True)
    eq0 = float(net_financial_assets(eco.ledger)[eco.ledger.entities.id("BANKSYS")])
    hh0 = eco.ledger.position("HH:0", "DEP")
    gov0 = eco.ledger.position("GOVT", "DEP")
    rng = np.random.default_rng(2)
    extras = [
        ExtraEffect(variable="bank_equity", shape="pulse", magnitude=0.05, duration=1),
        ExtraEffect(variable="vat", shape="step", magnitude=0.05, duration=3),
    ]
    ex = ExtraEffects()
    ex.start(extras, rng, tick=1, economy=eco)
    assert_consistent(eco.ledger, 1)
    eq1 = float(net_financial_assets(eco.ledger)[eco.ledger.entities.id("BANKSYS")])
    assert eq1 < eq0
    assert eco.ledger.position("HH:0", "DEP") == pytest.approx(hh0 - 0.05)
    assert eco.ledger.position("GOVT", "DEP") == pytest.approx(gov0 + 0.05)
    ex.apply(4, eco)
    assert ex.vat == pytest.approx(0.0)
    assert eco.policy.govt.effective.get("vat") in (None, 0.0)
    assert_consistent(eco.ledger, 1)
