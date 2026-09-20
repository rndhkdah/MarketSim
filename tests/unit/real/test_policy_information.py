"""T2.33 — committee information set: published vintages vs oracle."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.events.releases import DataReleases, gdp_release_tick
from marketsim.real.cenbank import seed_price_history
from marketsim.real.economy import RealEconomy
from marketsim.real.policy.monetary_rule import MonetaryRule, vintage_pi_pol


def test_poisoning_unpublished_cpi_does_not_change_pi_pol() -> None:
    g = float(np.exp(0.02 / 12.0))
    hist = seed_price_history(g, n=16)
    hist[-1] = hist[-2] * 1.08  # unpublished current-month spike
    base = list(hist)
    pi0 = vintage_pi_pol(base, base, lag_m=1, core_weight=0.5)
    poisoned = list(base)
    poisoned[-1] *= 1.50
    pi1 = vintage_pi_pol(poisoned, poisoned, lag_m=1, core_weight=0.5)
    assert pi1 == pytest.approx(pi0, abs=1e-15)
    published = list(base)
    published[-2] *= 1.50
    pi2 = vintage_pi_pol(published, published, lag_m=1, core_weight=0.5)
    assert pi2 != pytest.approx(pi0, abs=1e-12)


def test_oracle_and_vintage_differ_after_gdp_revision() -> None:
    rel = DataReleases()
    rng = np.random.default_rng(8)
    gdp0 = 100.0
    for m in range(3):
        rel.record_month(
            m,
            cpi=1.0,
            unemployment=0.05,
            output=np.ones(18),
            gdp=gdp0,
            trade_balance=0.0,
            govt_balance=0.0,
            policy_rate=0.03,
            month_end_tick=m * 21 + 20,
            rng=rng,
        )
    truth = rel._quarter_gdp_truth[0]
    t0 = gdp_release_tick(0, 0)
    t2 = gdp_release_tick(0, 2)
    first = float(rel.visible(t0)["gdp"]["q:0"])
    last = float(rel.visible(t2)["gdp"]["q:0"])
    assert first != pytest.approx(last)
    gap_first = first / truth - 1.0
    gap_last = last / truth - 1.0
    oracle = MonetaryRule(use_published_vintages=False, rate_step=0.0, deadband=0.0, mpy=8, phi_y=1.0)
    vintage = MonetaryRule(use_published_vintages=True, rate_step=0.0, deadband=0.0, mpy=8, phi_y=1.0)
    kwargs = dict(r_n=0.01, pi_star=0.02, pi_pol=0.02, u=0.05, u_star=0.05, r_rule=0.03)
    _, r_or, _ = oracle.decide(**kwargs, gap=0.0)
    _, r_v0, _ = vintage.decide(**kwargs, gap=gap_first)
    _, r_v1, _ = MonetaryRule(
        use_published_vintages=True, rate_step=0.0, deadband=0.0, mpy=8, phi_y=1.0
    ).decide(**kwargs, gap=gap_last)
    assert r_or != pytest.approx(r_v0, abs=1e-12)
    assert r_v0 != pytest.approx(r_v1, abs=1e-12)


def test_poison_true_series_leaves_decision(config_dir, io) -> None:
    cfg = load_config(config_dir)
    eco = RealEconomy(cfg, io, pi_star=0.02, check_sfc=False)
    assert eco.cb.framework is not None
    eco.cb.framework.use_published_vintages = True
    eco.cb.framework.cpi_lag_m = 1
    for _ in range(8):
        eco.step_month()
    hist = list(eco.cb.cpi_hist)
    core = list(eco.cb.core_hist)
    pi0 = vintage_pi_pol(hist, core, lag_m=1, core_weight=eco.cb.framework.core_weight)
    hist[-1] *= 1.25
    core[-1] *= 1.25
    pi1 = vintage_pi_pol(hist, core, lag_m=1, core_weight=eco.cb.framework.core_weight)
    assert pi1 == pytest.approx(pi0, abs=1e-15)
    assert eco.cb.framework.last_info["source"] == "vintage"


def test_lag_raises_inflation_volatility_monotone() -> None:
    """Toy dual-mandate loop: sd(π) rises with the publication lag (§2.14.2 prototype)."""

    def sd_pi(lag: int) -> float:
        rng = np.random.default_rng(3)
        pi = 0.02
        r = 0.03
        buf = [0.02] * 6
        out: list[float] = []
        for _ in range(360):
            seen = buf[-1 - lag]
            target = 0.01 + 0.02 + 1.5 * (seen - 0.02)
            r = 0.8 * r + 0.2 * target
            pi = pi + 0.4 * (0.03 - r) + rng.normal(0.0, 0.003)
            buf.append(pi)
            out.append(pi)
        return float(np.std(out[60:]))

    sds = [sd_pi(lag) for lag in (0, 1, 2, 3)]
    assert sds[0] < sds[1] < sds[2] < sds[3]


def test_oracle_flag_uses_current_month() -> None:
    g = float(np.exp(0.02 / 12.0))
    hist = seed_price_history(g, n=16)
    hist[-1] = hist[-2] * 1.05
    oracle = vintage_pi_pol(hist, hist, lag_m=0, core_weight=0.5)
    lagged = vintage_pi_pol(hist, hist, lag_m=1, core_weight=0.5)
    assert oracle != pytest.approx(lagged)
    rule = MonetaryRule(use_published_vintages=False)
    info = rule.information_set(cpi_hist=hist, core_hist=hist, u=0.07, gap=0.02)
    assert info.source == "oracle"
    assert info.u == pytest.approx(0.07)
    assert info.gap == pytest.approx(0.02)
    assert info.pi_pol == pytest.approx(oracle)
