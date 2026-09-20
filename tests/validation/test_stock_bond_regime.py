"""T6.21 — gate 2 regime flip using structural equity V and GB_BOND returns."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import Config
from marketsim.core.rng import RngHub
from marketsim.market.metrics import corr
from marketsim.pricing.bonds import BOND_INDEX_DURATION, bond_index_return
from marketsim.pricing.curve import y10
from marketsim.pricing.provider import AssetPriceProvider
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline

# Gate 2 bands.
DEMAND_CORR_MAX = -0.15
SUPPLY_CORR_MIN = 0.15
# Episode length (months) and AR persistence of the primitive (1/month).
EPISODE_M = 36
FACTOR_PHI = 0.85
# Idiosyncratic monthly return noise so the factor loadings are not a perfect line.
IDIO = 0.002


def _elasticities(cfg: Config, io) -> tuple[float, float, float, float]:
    """Market-cap equity dlnV per unit growth / rate / z_risk, plus y10 pass-through."""
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    prov = AssetPriceProvider.from_baseline(real, fin, cfg)
    w = np.array([cfg.sectors.market_cap_weights[c] for c in real.codes], dtype=float)
    w = w / w.sum()
    r_n = cfg.edges.policy.taylor.r_neutral
    v0 = float(np.dot(w, prov.values(fin.r0, 0.0, 0.0)))
    v_g = float(np.dot(w, prov.values(fin.r0, 0.0, 0.0, delta_g_lr=0.01)))
    v_r = float(np.dot(w, prov.values(fin.r0 + 0.01, 0.0, 0.0)))
    v_z = float(np.dot(w, prov.values(fin.r0, 0.0, 0.01)))
    el_g = np.log(v_g / v0) / 0.01
    el_r = np.log(v_r / v0) / 0.01
    el_z = np.log(v_z / v0) / 0.01
    dy10 = y10(fin.r0 + 0.01, r_n, 0.0) - y10(fin.r0, r_n, 0.0)
    return float(el_g), float(el_r), float(el_z), float(dy10 / 0.01)


def _factor(n: int, phi: float = FACTOR_PHI) -> np.ndarray:
    f = np.empty(n, dtype=float)
    f[0] = 1.0
    for t in range(1, n):
        f[t] = phi * f[t - 1]
    return f


def _episode(eq_load: float, bond_load: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    f = _factor(EPISODE_M)
    eq = eq_load * f + IDIO * rng.standard_normal(EPISODE_M)
    bd = bond_load * f + IDIO * rng.standard_normal(EPISODE_M)
    return eq, bd


@pytest.mark.validation
def test_gate2_demand_negative_supply_positive(cfg: Config, io) -> None:
    el_g, el_r, el_z, y_pass = _elasticities(cfg, io)
    # Demand episode: growth + Taylor rate. Growth must dominate so equities rise.
    g, r_imp = 0.04, 0.01
    eq_d_load = el_g * g + el_r * r_imp
    assert eq_d_load > 0.0
    bond_d_load = -BOND_INDEX_DURATION * y_pass * r_imp
    assert bond_d_load < 0.0
    # Supply / cost-push: earnings down, rates up, risk-off — both negative.
    eq_s_load = el_g * (-0.03) + el_r * r_imp + el_z * 0.02
    bond_s_load = -BOND_INDEX_DURATION * y_pass * r_imp
    assert eq_s_load < 0.0
    assert bond_s_load < 0.0
    rng = RngHub(0).stream("regime")
    eq_d, bd_d = _episode(eq_d_load, bond_d_load, rng)
    eq_s, bd_s = _episode(eq_s_load, bond_s_load, rng)
    assert corr(eq_d, bd_d) < DEMAND_CORR_MAX
    assert corr(eq_s, bd_s) > SUPPLY_CORR_MIN
    assert float(eq_d[:6].sum()) > 0.0
    assert float(bd_d[:6].sum()) < 0.0
    assert float(eq_s[:6].sum()) < 0.0
    assert float(bd_s[:6].sum()) < 0.0


@pytest.mark.validation
def test_gb_bond_return_hook(cfg: Config, io) -> None:
    """Gate 11 hook: GB_BOND total return is the §6.3 bond-index formula."""
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    r_n = cfg.edges.policy.taylor.r_neutral
    y0 = y10(fin.r0, r_n, 0.0)
    y1 = y10(fin.r0 + 0.01, r_n, 0.0)
    assert bond_index_return(y0, y1) == pytest.approx(-BOND_INDEX_DURATION * (y1 - y0) + y0 / 12.0)
