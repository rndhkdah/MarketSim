"""T6.04 — discount rates, fundamental value, stub parity, factor_lite."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import Config, load_config
from marketsim.pricing.discount import delta_rho
from marketsim.pricing.fundamentals import fundamental_values
from marketsim.pricing.provider import AssetPriceProvider, StubAssetPriceProvider
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def test_q_is_one_at_baseline(cfg: Config, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    prov = AssetPriceProvider.from_baseline(real, fin, cfg)
    q = prov.q_tobin(fin.r0, fin.pi_star, 0.0)
    assert np.allclose(q, 1.0, atol=1e-12)


def test_duration_orders_rate_sensitivity(cfg: Config, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    prov = AssetPriceProvider.from_baseline(real, fin, cfg)
    v0 = prov.values(fin.r0, fin.pi_star, 0.0)
    v1 = prov.values(fin.r0 + 0.01, fin.pi_star, 0.0)
    rel = v1 / v0 - 1.0
    fin_codes = set(cfg.sectors.financials)
    mask = np.array([c not in fin_codes for c in real.codes])
    # Longer cash-flow duration → more negative rate response (non-financials).
    order = np.argsort(prov.duration[mask])
    rel_nf = rel[mask][order]
    assert rel_nf[0] > rel_nf[-1]
    assert np.all(np.diff(rel_nf) <= 1e-12)


def test_stub_parity_when_earnings_constant(cfg: Config, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    structural = AssetPriceProvider.from_baseline(real, fin, cfg)
    # Phase-2 closed form: ln V = ln E^e + ln PE0 − D · 0.35 · (r − π_e − r_n)
    drho_stub = 0.35 * 0.01
    v_stub = fundamental_values(structural.ee, structural.pe0, structural.duration, drho_stub)
    v_new = structural.values(fin.r0 + 0.01, fin.pi_star, 0.0)
    assert np.allclose(v_new, v_stub, rtol=0.0, atol=1e-12)
    # Same interface name still constructs.
    stub = StubAssetPriceProvider.from_baseline(real, fin, cfg)
    assert np.allclose(stub.q_tobin(fin.r0, 0.0), 1.0, atol=1e-12)


def test_factor_lite_q_one_and_uses_betas(config_dir, io) -> None:
    cfg = load_config(config_dir, {"markets.pricing.mode": "factor_lite"})
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    prov = AssetPriceProvider.from_baseline(real, fin, cfg)
    assert prov.mode == "factor_lite"
    assert prov.beta_rate is not None
    assert np.allclose(prov.q_tobin(fin.r0, fin.pi_star, 0.0), 1.0, atol=1e-12)
    v0 = prov.values(fin.r0, fin.pi_star, 0.0)
    v1 = prov.values(fin.r0 + 0.01, fin.pi_star, 0.0)
    banks = list(real.codes).index("BANKS")
    re_i = list(real.codes).index("REALESTATE")
    # BANKS is the only rate-positive sector in the published betas.
    assert v1[banks] > v0[banks]
    assert v1[re_i] < v0[re_i]


def test_delta_rho_matches_curve_pass_through() -> None:
    d = delta_rho(0.02, 0.01, 0.0)
    assert d == pytest.approx(0.0035, abs=0.0005)
