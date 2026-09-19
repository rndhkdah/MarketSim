from __future__ import annotations

import pytest

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent
from marketsim.pricing.provider import StubAssetPriceProvider
from marketsim.real.credit import CreditBlock
from marketsim.real.economy import RealEconomy
from marketsim.real.policy.authority import PolicyDecision
from marketsim.real.policy.monetary import DeferredQE, MonetaryLevers, post_lolr
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def test_lolr_posting_balanced() -> None:
    led = Ledger.empty()
    for name in ("BANKSYS", "CB"):
        led.register_entity(name)
    led.post(
        Tx(
            0,
            "opening",
            [Entry("BANKSYS", "RES", 8.0), Entry("CB", "RES", -8.0)],
        )
    )
    post_lolr(led, 3.0, tick=1)
    assert_consistent(led, 1)
    assert led.position("BANKSYS", "RES") == pytest.approx(11.0)
    assert led.position("CB", "RES") == pytest.approx(-11.0)
    assert led.position("CB", "LOAN") == pytest.approx(3.0)
    assert led.position("BANKSYS", "LOAN") == pytest.approx(-3.0)


def test_qe_interface_declared() -> None:
    qe = DeferredQE()
    with pytest.raises(NotImplementedError, match="T6.28"):
        qe.purchase(1.0, "GB_NOTE")
    with pytest.raises(NotImplementedError, match="T6.28"):
        qe.sale(1.0, "GB_BOND")


def test_higher_capital_requirement_lowers_gate(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    block = CreditBlock(cfg, real, fin, StubAssetPriceProvider.from_baseline(real, fin, cfg))
    block.enabled = True
    base = block.update(capital=0.125, r=fin.r0, pi_e=0.0, z_risk=0.0, ll_bar=block.ll_bar0)
    assert base == pytest.approx(1.0, abs=1e-12)
    block.capital_requirement = 0.145
    raised = block.update(capital=0.125, r=fin.r0, pi_e=0.0, z_risk=0.0, ll_bar=block.ll_bar0)
    assert block.gate < 0.95
    assert raised < 1.0


def test_rate_override_only_at_meetings(config_dir) -> None:
    cfg = load_config(config_dir)
    io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
    r0 = eco.cb.r
    eco.policy.cenbank.submit(PolicyDecision(source="agent", rate=0.05), month=eco.month, lag_m=0)
    eco.step_month()
    assert eco.cb.r == pytest.approx(r0, abs=1e-12)
    eco.step_month()
    assert eco.cb.r == pytest.approx(r0, abs=1e-12)
    eco.step_month()
    assert eco.cb.r == pytest.approx(0.05, abs=1e-12)


def test_guidance_path_stored() -> None:
    lev = MonetaryLevers(guidance_path=[0.02, 0.025, 0.03])
    assert lev.guidance_path == [0.02, 0.025, 0.03]
    assert lev.rate is None
