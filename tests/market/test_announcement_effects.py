"""T6.32 — anticipated vs surprise policy announcements; surprise unforecastable."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import Config
from marketsim.pricing.bond_buckets import DURATION_REF_YIELD
from marketsim.pricing.curve import cash_flow_weights, expected_policy_path
from marketsim.pricing.policy_surprise import (
    announcement_effect,
    market_implied_path,
    ols_r_squared,
    policy_surprise,
)
from marketsim.real.policy.monetary_rule import projection_path


def _bucket_weights(cfg: Config) -> dict[str, np.ndarray]:
    assert cfg.bonds is not None
    y = DURATION_REF_YIELD
    return {name: cash_flow_weights(spec.decay, y) for name, spec in cfg.bonds.buckets.items()}


def test_anticipated_hike_barely_moves_prices(cfg: Config) -> None:
    r_n, pi = DURATION_REF_YIELD, 0.0
    r0, r1 = r_n, r_n + 0.01
    guidance = expected_policy_path(r1, r_n, pi)
    fx = announcement_effect(
        r_published=r0,
        decision=r1,
        r_n=r_n,
        pi_star=pi,
        weights=_bucket_weights(cfg),
        guidance_path=guidance,
        credibility=1.0,
    )
    assert fx.surprise == pytest.approx(0.0)
    assert abs(fx.dln_equity) < 1e-12
    assert all(abs(dy) < 1e-12 for dy in fx.dy.values())


def test_surprise_hike_raises_yields_and_lowers_equities(cfg: Config) -> None:
    r_n, pi = DURATION_REF_YIELD, 0.0
    r0, r1 = r_n, r_n + 0.01
    w = _bucket_weights(cfg)
    fx = announcement_effect(
        r_published=r0,
        decision=r1,
        r_n=r_n,
        pi_star=pi,
        weights=w,
        credibility=0.0,
    )
    assert fx.surprise == pytest.approx(0.01)
    assert fx.dln_equity < 0.0
    assert fx.xi_equity < 0.0
    assert fx.dy["GB_BILL"] > 0.0
    assert fx.dy["GB_BOND"] > 0.0
    # Front bucket moves most in yield terms; the 10-year least among governments.
    assert fx.dy["GB_BILL"] > fx.dy["GB_NOTE"] > fx.dy["GB_BOND"]
    # Path jump is the §6.3 decay: GB_BOND ≈ +35bp for +100bp.
    assert fx.dy["GB_BOND"] == pytest.approx(0.0035, abs=0.0010)


def test_surprise_unforecastable_from_public_data() -> None:
    """Projection noise is orthogonal to the published (r, π, U) information set."""
    r_n, pi_star = 0.02, 0.02
    rng = np.random.default_rng(0)
    n = 240
    r_pub = r_n + pi_star + 0.004 * rng.standard_normal(n)
    cpi = pi_star + 0.003 * rng.standard_normal(n)
    u_gap = 0.002 * rng.standard_normal(n)
    surprises = np.empty(n)
    for i in range(n):
        implied = market_implied_path(float(r_pub[i]), r_n, pi_star)
        # Noisy committee print vs the public implied short rate (§2.14.5).
        noisy = projection_path(
            float(r_pub[i]),
            float(r_pub[i]),
            0.80,
            0.0025,
            0.0010,
            float(r_pub[i]),
            n=1,
            noise_bp=25.0,
            seed=i + 1,
            meeting_n=i,
        )[0]
        surprises[i] = policy_surprise(noisy, implied)
    public = np.column_stack([r_pub, cpi, u_gap])
    r2 = ols_r_squared(surprises, public)
    assert r2 < 0.2
