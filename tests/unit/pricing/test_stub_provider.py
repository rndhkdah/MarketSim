from __future__ import annotations

import numpy as np
import pytest

from marketsim.pricing.provider import StubAssetPriceProvider
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def test_q_is_one_at_baseline(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    prov = StubAssetPriceProvider.from_baseline(real, fin, cfg)
    q = prov.q_tobin(fin.r0, fin.pi_star, 0.0)
    assert np.allclose(q, 1.0, atol=1e-12)


def test_plus_100bp_real_rate_cuts_v_by_duration(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    prov = StubAssetPriceProvider.from_baseline(real, fin, cfg)
    v0 = prov.values(fin.r0, fin.pi_star, 0.0)
    v1 = prov.values(fin.r0 + 0.01, fin.pi_star, 0.0)
    codes = list(real.codes)
    rel = v1 / v0 - 1.0
    re_i = codes.index("REALESTATE")
    au_i = codes.index("AUTOS")
    dur = np.array([cfg.sectors.params(c).cf_duration for c in real.codes])
    # Linearised: −0.35 × duration % per +100bp real rate (§2.10).
    assert float(rel[re_i]) == pytest.approx(-0.35 * dur[re_i] * 0.01, rel=0.05)
    assert float(rel[au_i]) == pytest.approx(-0.35 * dur[au_i] * 0.01, rel=0.05)
    assert float(rel[re_i]) == pytest.approx(-0.063, abs=0.003)
    assert float(rel[au_i]) == pytest.approx(-0.025, abs=0.002)
