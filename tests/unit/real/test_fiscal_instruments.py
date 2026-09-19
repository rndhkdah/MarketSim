from __future__ import annotations

import numpy as np
import pytest

from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.opening import open_passthrough_books
from marketsim.ledger.sfc import assert_consistent
from marketsim.real.economy import RealEconomy
from marketsim.real.government import post_bond_issue
from marketsim.real.policy.authority import PolicyDecision
from marketsim.real.policy.fiscal import (
    apply_purchases,
    consumer_prices,
    cover_govt_shortfall,
    excise_revenue,
    post_excise,
    post_rescue_banksys,
    post_subsidy,
    post_tariff,
    subsidised_v,
    tariff_revenue,
    vat_revenue,
)
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def test_vat_excise_raises_consumer_not_producer() -> None:
    p = np.array([1.0, 2.0, 0.5])
    excise = np.array([0.0, 0.10, 0.0])
    pc = consumer_prices(p, vat=0.20, excise=excise)
    assert pc[0] == pytest.approx(1.2)
    assert pc[1] == pytest.approx(2.0 * 1.30)
    assert pc[2] == pytest.approx(0.5 * 1.20)
    assert np.array_equal(p, np.array([1.0, 2.0, 0.5]))


def test_tariff_revenue_equals_rate_times_import_value() -> None:
    assert tariff_revenue(import_value=80.0, rate=0.10) == pytest.approx(8.0)
    assert vat_revenue(producer_spend=50.0, vat=0.2) == pytest.approx(10.0)
    assert excise_revenue(np.array([10.0, 20.0]), np.array([0.1, 0.0])) == pytest.approx(1.0)


def test_every_instrument_posts_balanced_tx() -> None:
    led = Ledger.empty()
    for name in ("HH:0", "NPC:0:ENERGY", "BANKSYS", "GOVT", "ROW"):
        led.register_entity(name)
    led.post(Tx(0, "opening", [Entry("HH:0", "DEP", 200.0), Entry("BANKSYS", "DEP", -200.0)]))
    post_tariff(led, 4.0, tick=1, firm="NPC:0:ENERGY")
    post_excise(led, 1.5, tick=1)
    post_subsidy(led, {"ENERGY": 2.0}, tick=1, region=0)
    post_rescue_banksys(led, 5.0, tick=1)
    assert_consistent(led, 1)
    assert led.position("GOVT", "DEP") == pytest.approx(4.0 + 1.5 - 2.0 - 5.0)


def test_govt_deposits_never_negative() -> None:
    led = Ledger.empty()
    for name in ("HH:0", "GOVT", "BANKSYS"):
        led.register_entity(name)
    for inst in ("GB_BILL", "GB_NOTE", "GB_BOND"):
        led.register_instrument(inst, financial=True)
    led.post(Tx(0, "opening", [Entry("HH:0", "DEP", 50.0), Entry("BANKSYS", "DEP", -50.0)]))
    led.post(Tx(1, "govt_purchases", [Entry("GOVT", "DEP", -30.0), Entry("HH:0", "DEP", 30.0)]))
    assert led.position("GOVT", "DEP") < 0
    issued = cover_govt_shortfall(led, tick=1)
    assert issued == pytest.approx(30.0)
    assert led.position("GOVT", "DEP") == pytest.approx(0.0, abs=1e-12)
    assert_consistent(led, 1)


def test_capex_subsidy_lowers_effective_v() -> None:
    v = np.array([2.0, 4.0, 1.0])
    codes = ("A", "B", "C")
    out = subsidised_v(v, {"B": 0.25}, codes)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(3.0)
    assert out[2] == pytest.approx(1.0)


def test_purchase_mix_and_level() -> None:
    g0 = np.array([4.0, 1.0, 0.0])
    codes = ("CONSTRUCT", "HEALTH", "BANKS")
    g = apply_purchases(g0, 0.0, level=10.0, mix={"CONSTRUCT": 0.7, "HEALTH": 0.3}, codes=codes)
    assert g.sum() == pytest.approx(10.0)
    assert g[0] == pytest.approx(7.0)
    assert g[1] == pytest.approx(3.0)


def test_bond_issue_helper_still_balanced(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    led = open_passthrough_books(cfg, real, fin)
    post_bond_issue(led, 3.0, tick=1)
    assert_consistent(led, 1)


def test_economy_vat_wedge_and_sfc(config_dir, cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=True)
    eco.policy.govt.submit(PolicyDecision(source="agent", vat=0.10), month=eco.month, lag_m=0)
    rec = eco.step_month()
    assert eco.last_flows is not None
    c_prod = float(eco.last_flows.c_nom.sum())
    assert eco.last_flows.vat == pytest.approx(0.10 * c_prod, rel=1e-9)
    producer_cpi = float((eco.theta * eco.p).sum())
    assert rec["cpi"] == pytest.approx(1.10 * producer_cpi, rel=1e-9)
    assert rec["cpi"] > producer_cpi
    assert eco.ledger.position("GOVT", "DEP") >= -1e-9


def test_economy_tariff_and_rescue_sfc(config_dir, cfg, io) -> None:
    from marketsim.core.config import load_config
    from marketsim.layer1.io import load_io, resolve_io_path

    full = load_config(config_dir, {"dynamics.banks.mode": "full"})
    eco = RealEconomy(full, load_io(resolve_io_path(full)), pi_star=0.0, check_sfc=True)
    eco.policy.govt.submit(
        PolicyDecision(source="agent", tariff=0.10, rescue_banksys=1.0),
        month=eco.month,
        lag_m=0,
    )
    eco.step_month()
    flows = eco.last_flows
    assert flows is not None
    assert flows.tariff == pytest.approx(0.10 * float(flows.imp_nom.sum()), rel=1e-9)
    assert flows.rescue == pytest.approx(1.0)
    assert eco.ledger.position("GOVT", "DEP") >= -1e-9
