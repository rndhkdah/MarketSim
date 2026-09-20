"""T6.07 — §6.4 impact kernel: fit, concavity, decay, O(K) state, fills."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.market.impact import (
    FIT_HORIZON_D,
    ImpactKernel,
    daily_decays,
    delta_impact,
    fill_price,
    fit_kernel_weights,
    impulse_response,
    post_impact_mid,
    power_law_kernel,
)
from marketsim.market.instruments import ImpactCfg

# T6.07 card: relative error of Σ w ρ^τ vs G(τ) on 1–250 days.
REL_ERR_MAX = 0.10
EVAL_MIN_D = 1  # days; §6.4 card window
EVAL_MAX_D = 250  # days; §6.4 card window


def test_fitted_kernel_within_10pct_of_power_law_on_1_to_250_days() -> None:
    cfg = ImpactCfg()
    rho = daily_decays(cfg.half_lives_d)
    weights = fit_kernel_weights(cfg.half_lives_d, cfg.tau0_d, cfg.beta)
    assert FIT_HORIZON_D == 500.0
    assert weights.shape == (len(cfg.half_lives_d),)
    assert np.all(weights >= 0.0)
    tau = np.arange(EVAL_MIN_D, EVAL_MAX_D + 1, dtype=float)
    target = np.asarray(power_law_kernel(tau, cfg.tau0_d, cfg.beta), dtype=float)
    fitted = np.asarray(impulse_response(tau, weights, rho), dtype=float)
    rel = np.abs(fitted - target) / target
    assert float(np.max(rel)) <= REL_ERR_MAX


def test_single_order_concavity() -> None:
    cfg = ImpactCfg()
    assert cfg.delta < 1.0
    q, adv, sigma = 20.0, 100.0, 0.02
    a = ImpactKernel(cfg)
    b = ImpactKernel(cfg)
    xi_q = a.step(q, adv, sigma)
    xi_2q = b.step(2.0 * q, adv, sigma)
    assert float(xi_2q) < 2.0 * float(xi_q)
    d_q = delta_impact(q, adv, sigma, Y=cfg.Y, delta=cfg.delta)
    d_2q = delta_impact(2.0 * q, adv, sigma, Y=cfg.Y, delta=cfg.delta)
    assert abs(float(d_2q)) < 2.0 * abs(float(d_q))


def test_state_decays_to_zero_after_pulse() -> None:
    kn = ImpactKernel()
    kn.step(15.0, adv=80.0, sigma=0.03)
    assert abs(float(kn.xi)) > 0.0
    # Slowest half-life is 250 days; ~8 000 quiet days → 2^(-8000/250) ≈ 2e-10.
    n_quiet = 8000
    for _ in range(n_quiet):
        kn.step(0.0, adv=80.0, sigma=0.03)
    assert abs(float(kn.xi)) < 1e-8
    assert np.allclose(kn.I, 0.0, atol=1e-8)


def test_state_is_ok_per_instrument() -> None:
    cfg = ImpactCfg()
    kn = ImpactKernel(cfg)
    assert kn.n_components == len(cfg.half_lives_d)
    assert kn.I.shape == (len(cfg.half_lives_d),)
    assert len(kn.I) == len(cfg.half_lives_d)
    assert len(kn.to_state()["I"]) == len(cfg.half_lives_d)
    book = ImpactKernel(cfg, shape=(3,))
    assert book.I.shape == (3, len(cfg.half_lives_d))


def test_sign_of_q_sets_sign_of_delta_I() -> None:
    cfg = ImpactCfg()
    q, adv, sigma = 12.0, 90.0, 0.015
    d_pos = delta_impact(q, adv, sigma, Y=cfg.Y, delta=cfg.delta)
    d_neg = delta_impact(-q, adv, sigma, Y=cfg.Y, delta=cfg.delta)
    d_zero = delta_impact(0.0, adv, sigma, Y=cfg.Y, delta=cfg.delta)
    assert float(d_pos) > 0.0
    assert float(d_neg) < 0.0
    assert float(d_neg) == pytest.approx(-float(d_pos))
    assert float(d_zero) == pytest.approx(0.0)
    pos = ImpactKernel(cfg)
    neg = ImpactKernel(cfg)
    assert float(pos.step(q, adv, sigma)) > 0.0
    assert float(neg.step(-q, adv, sigma)) < 0.0
    assert float(neg.xi) == pytest.approx(-float(pos.xi))


def test_q_zero_decays_only() -> None:
    kn = ImpactKernel()
    kn.step(8.0, adv=50.0, sigma=0.02)
    I_after_pulse = kn.I.copy()
    xi_after_pulse = float(kn.xi)
    kn.step(0.0, adv=50.0, sigma=0.02)
    assert kn.I == pytest.approx(I_after_pulse * kn.rho)
    assert float(kn.xi) == pytest.approx(float((I_after_pulse * kn.rho).sum()))
    assert abs(float(kn.xi)) < abs(xi_after_pulse)


def test_fill_price_uses_post_impact_mid() -> None:
    kn = ImpactKernel()
    mid = 100.0
    spread = 0.01  # dimensionless quoted s; half-spread = s/2 (§6.4)
    q, adv, sigma = 25.0, 100.0, 0.02
    xi_pre = float(kn.xi)
    assert xi_pre == pytest.approx(0.0)
    p_buy = kn.fill(mid, spread, q, adv, sigma)
    xi_post = float(kn.xi)
    mid_post = float(post_impact_mid(mid, xi_pre, xi_post))
    assert mid_post > mid
    assert p_buy == pytest.approx(mid_post * (1.0 + spread / 2.0))
    # Not the pre-impact quote: nobody trades ahead of their own impact.
    assert p_buy > mid * (1.0 + spread / 2.0)
    assert p_buy == pytest.approx(fill_price(mid_post, spread, q))

    sell = ImpactKernel()
    p_sell = sell.fill(mid, spread, -q, adv, sigma)
    mid_post_sell = float(post_impact_mid(mid, 0.0, float(sell.xi)))
    assert mid_post_sell < mid
    assert p_sell == pytest.approx(mid_post_sell * (1.0 - spread / 2.0))
    assert p_sell < mid * (1.0 - spread / 2.0)


def test_state_roundtrip() -> None:
    kn = ImpactKernel()
    kn.step(4.0, adv=40.0, sigma=0.01)
    kn.step(-1.0, adv=40.0, sigma=0.01)
    restored = ImpactKernel.from_state(kn.to_state())
    assert restored.to_state() == kn.to_state()
    assert restored.I == pytest.approx(kn.I)
    assert float(restored.xi) == pytest.approx(float(kn.xi))


def test_vectorized_multi_name_matches_scalars() -> None:
    cfg = ImpactCfg()
    q = np.array([10.0, -6.0, 0.0])
    adv = np.array([80.0, 80.0, 80.0])
    sigma = np.array([0.02, 0.03, 0.01])
    book = ImpactKernel(cfg, shape=(3,))
    xi = np.asarray(book.step(q, adv, sigma), dtype=float)
    assert book.I.shape == (3, len(cfg.half_lives_d))
    for i in range(3):
        one = ImpactKernel(cfg)
        one.step(float(q[i]), float(adv[i]), float(sigma[i]))
        assert float(xi[i]) == pytest.approx(float(one.xi))
        assert book.I[i] == pytest.approx(one.I)
