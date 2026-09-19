from __future__ import annotations

import pytest

from marketsim.core.config import load_config
from marketsim.core.errors import ConfigError
from marketsim.real.policy.monetary_rule import (
    MonetaryRule,
    elb_toolkit,
    fci_tighten,
    reaction_target,
    sahm_risk_off,
    step_plevel_gap,
)

MAKEUP_MSG = "−93 %"


def test_makeup_no_leak_rejected(config_dir) -> None:
    with pytest.raises(ConfigError, match=MAKEUP_MSG):
        load_config(config_dir, {"policy.monetary.strategy.makeup": 0.2, "policy.monetary.strategy.makeup_decay": 1.0})
    with pytest.raises(ConfigError, match="73 %"):
        load_config(config_dir, {"policy.monetary.strategy.makeup": 0.2, "policy.monetary.strategy.makeup_clip": 0.0})


def test_makeup_zero_allows_decay_one(config_dir) -> None:
    cfg = load_config(config_dir, {"policy.monetary.strategy.makeup_decay": 1.0})
    assert cfg.policy.monetary.strategy.makeup == 0.0


def test_plevel_gap_leaks_and_clips() -> None:
    g = step_plevel_gap(0.0, pi_pol=0.50, pi_star=0.02, decay=0.98, clip=0.02)
    assert g == pytest.approx(0.02)
    g = 0.0
    for _ in range(5):
        g = step_plevel_gap(g, pi_pol=0.05, pi_star=0.02, decay=0.98, clip=0.02)
    assert 0.0 < g < 0.02
    leaked = step_plevel_gap(0.02, pi_pol=0.00, pi_star=0.02, decay=0.5, clip=0.02)
    assert leaked < 0.02


def test_sahm_recession_not_soft_landing() -> None:
    rec = [0.05] * 12 + [0.055, 0.058, 0.062]
    soft = [0.05] * 12 + [0.051, 0.0515, 0.052]
    off_r, _ = sahm_risk_off(rec, threshold=0.005, cut=0.005, decay_m=12, residual=0.0)
    off_s, _ = sahm_risk_off(soft, threshold=0.005, cut=0.005, decay_m=12, residual=0.0)
    assert off_r == pytest.approx(0.005)
    assert off_s == pytest.approx(0.0)


def test_phi_fci_wider_spread_lowers_rate() -> None:
    base = reaction_target(
        r_n=0.01, pi_star=0.02, phi_pi=1.5, phi_u=1.0, phi_y=0.0,
        pi_pol=0.02, u=0.05, u_star=0.05, gap=0.0,
        fci_tighten=0.0,
    )
    tight = fci_tighten(spread=0.035, s0=0.015, gate=1.0, phi_fci=1.0, w_spread=0.4, w_gate=0.3)
    lower = reaction_target(
        r_n=0.01, pi_star=0.02, phi_pi=1.5, phi_u=1.0, phi_y=0.0,
        pi_pol=0.02, u=0.05, u_star=0.05, gap=0.0,
        fci_tighten=tight,
    )
    assert tight > 0
    assert lower < base


def test_elb_toolkit_order() -> None:
    # Guidance first; still at the floor → QE. A lift-off path skips QE.
    engaged, r, qe = elb_toolkit(
        shadow=-0.01, elb=0.0, gap=-0.02, guidance=-0.01, credibility=0.7, qe_per_gap=0.02
    )
    assert engaged[0] == "guidance"
    assert engaged[1] == "qe"
    assert r == pytest.approx(0.0)
    assert qe > 0
    lift, r_lift, qe_lift = elb_toolkit(
        shadow=-0.01, elb=0.0, gap=-0.02, guidance=0.01, credibility=0.7, qe_per_gap=0.02
    )
    assert lift == ("guidance",)
    assert r_lift > 0.0
    assert qe_lift == 0.0
    engaged2, r2, qe2 = elb_toolkit(shadow=0.02, elb=0.0, gap=0.0, guidance=None, credibility=0.7, qe_per_gap=0.02)
    assert engaged2 == ()
    assert r2 == pytest.approx(0.02)
    assert qe2 == 0.0


def test_rule_decide_uses_fci() -> None:
    rule = MonetaryRule(mpy=4, rate_step=0.0, deadband=0.0, phi_u=0.0, phi_y=0.0, kappa_rstar=0.0)
    rr0, ra0, _ = rule.decide(r_n=0.01, pi_star=0.02, pi_pol=0.02, u=0.05, u_star=0.05, gap=0.0, r_rule=0.03)
    rule.meeting_n = 0
    rr1, ra1, _ = rule.decide(
        r_n=0.01, pi_star=0.02, pi_pol=0.02, u=0.05, u_star=0.05, gap=0.0, r_rule=0.03, fci_tighten=0.02,
    )
    assert ra1 < ra0
    assert rr1 < rr0


def test_phi_credit_raises_target() -> None:
    base = reaction_target(
        r_n=0.01, pi_star=0.02, phi_pi=1.5, phi_u=1.0, phi_y=0.0,
        pi_pol=0.02, u=0.05, u_star=0.05, gap=0.0,
    )
    hiked = reaction_target(
        r_n=0.01, pi_star=0.02, phi_pi=1.5, phi_u=1.0, phi_y=0.0,
        pi_pol=0.02, u=0.05, u_star=0.05, gap=0.0,
        phi_credit=1.0, credit_gap=0.02,
    )
    assert hiked == pytest.approx(base + 0.02)


def test_update_strategy_uses_pi_star_and_sets_makeup() -> None:
    rule = MonetaryRule(makeup=0.2, makeup_decay=0.98, makeup_clip=0.02)
    mk, fci, off = rule.update_strategy(
        pi_pol=0.05, pi_star=0.02, u=0.05, spread=0.015, s0=0.015, gate=1.0
    )
    assert rule.plevel == pytest.approx(0.03 / 12.0)
    assert mk == pytest.approx(0.2 * rule.plevel)
    assert fci == 0.0
    assert off == 0.0
    mk0, _, _ = MonetaryRule(makeup=0.2, makeup_decay=0.98, makeup_clip=0.02).update_strategy(
        pi_pol=0.02, pi_star=0.02, u=0.05, spread=0.015, s0=0.015, gate=1.0
    )
    assert mk0 == pytest.approx(0.0)
