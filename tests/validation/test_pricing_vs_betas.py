"""T6.05 — gate 1: valuation sensitivities emerge from the structural provider."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from marketsim.core.config import Config, load_config
from marketsim.layer1.betas import BetasParams, derive_betas
from marketsim.pricing.provider import AssetPriceProvider
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline

# Gate 1: Spearman ≥ 0.7 for rate/growth/oil, ≥ 0.6 for credit (§6.1).
SPEARMAN_RATE = 0.7
SPEARMAN_GROWTH = 0.7
SPEARMAN_OIL = 0.7
SPEARMAN_CREDIT = 0.6
# +100 bp / +1 % factor impulses (annual decimal or log-earnings).
RATE_IMPULSE = 0.01
FACTOR_IMPULSE = 0.01


def _vec(cfg: Config, key: str) -> np.ndarray:
    return np.array([getattr(cfg.sectors.params(c), key) for c in cfg.codes], dtype=float)


def _provider(cfg: Config, io) -> tuple[AssetPriceProvider, object, object]:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    return AssetPriceProvider.from_baseline(real, fin, cfg), real, fin


def _credit_el(cfg: Config) -> np.ndarray:
    codes = list(cfg.codes)
    idx = {c: i for i, c in enumerate(codes)}
    el = np.zeros(len(codes))
    for edge in cfg.edges.credit.edges:
        if edge.dst == "ALL_SECTORS":
            el[:] = edge.elasticity
    for edge in cfg.edges.credit.edges:
        if edge.dst in idx:
            el[idx[edge.dst]] = edge.elasticity
    return el


def _growth_dln_ee(io, cfg: Config) -> np.ndarray:
    """Layer-1 cyclical earnings channel (same ingredients as ``derive_betas``)."""
    p = BetasParams()
    x0 = io.L @ io.baseline_final_demand(100.0)
    d_cyc = 0.70 * io.final_demand["HOUSEHOLD"] + 0.30 * io.final_demand["INVESTMENT"]
    rev = (io.L @ d_cyc) / np.maximum(x0, 1e-9)
    rev_z = rev / float(np.mean(rev))
    eta = _vec(cfg, "eta")
    routing = np.zeros(io.n)
    for code, w in cfg.edges.capex.routing.items():
        routing[io.index[code]] = w
    inv_z = routing / (float(np.mean(routing[routing > 0])) if np.any(routing > 0) else 1.0)
    cyc = p.cyclical_leontief * rev_z + p.cyclical_eta * (eta - 1.0) + p.cyclical_invest * inv_z
    ol = 1.0 / np.maximum(1.0 - _vec(cfg, "fixed_cost"), 0.25)
    return p.da_tax_factor * ol * cyc * FACTOR_IMPULSE


def _oil_dln_ee(io, cfg: Config) -> np.ndarray:
    p = BetasParams()
    energy = io.index["ENERGY"]
    cost = io.G[:, energy]
    pt = _vec(cfg, "pass_through")
    va = np.maximum(1.0 - io.mu, 0.15)
    dln = -p.oil_cost_scale * p.da_tax_factor * cost * (1.0 - 0.40 * pt) / va
    dln = dln.copy()
    dln[energy] += p.oil_own
    return dln * FACTOR_IMPULSE


def _credit_dln_ee(cfg: Config) -> np.ndarray:
    p = BetasParams()
    nd = _vec(cfg, "nd_ebitda")
    return -p.credit_scale * _credit_el(cfg) * np.maximum(nd, 0.2) * FACTOR_IMPULSE


def _rate_dln(prov: AssetPriceProvider, fin, cfg: Config) -> np.ndarray:
    """Discount + financials + Layer-1 refi/revenue earnings channels."""
    p = BetasParams()
    nd = _vec(cfg, "nd_ebitda")
    dem = _vec(cfg, "dem_rate_semi")
    dln_ee = (-p.refi_share * nd + p.revenue_link * dem) * RATE_IMPULSE
    ee1 = prov.ee * np.exp(dln_ee)
    v0 = prov.values(fin.r0, fin.pi_star, 0.0, apply_financials=True)
    v1 = prov.values(fin.r0 + RATE_IMPULSE, fin.pi_star, 0.0, ee=ee1, apply_financials=True)
    return np.log(np.maximum(v1, 1e-18) / np.maximum(v0, 1e-18))


@pytest.mark.validation
def test_gate1_betas_emerge(config_dir, io) -> None:
    cfg = load_config(config_dir, {"dynamics.banks.mode": "passthrough"})
    prov, _real, fin = _provider(cfg, io)
    published = derive_betas(io, cfg)
    dln_rate = _rate_dln(prov, fin, cfg)
    dln_g = np.log(
        prov.values(fin.r0, fin.pi_star, 0.0, ee=prov.ee * np.exp(_growth_dln_ee(io, cfg)))
        / prov.values(fin.r0, fin.pi_star, 0.0)
    )
    dln_oil = np.log(
        prov.values(fin.r0, fin.pi_star, 0.0, ee=prov.ee * np.exp(_oil_dln_ee(io, cfg)))
        / prov.values(fin.r0, fin.pi_star, 0.0)
    )
    dln_cr = np.log(
        prov.values(fin.r0, fin.pi_star, 0.0, ee=prov.ee * np.exp(_credit_dln_ee(cfg)))
        / prov.values(fin.r0, fin.pi_star, 0.0)
    )
    rho_r = float(spearmanr(dln_rate, published.beta_rate).statistic)
    rho_g = float(spearmanr(dln_g, published.beta_growth).statistic)
    rho_o = float(spearmanr(dln_oil, published.beta_oil).statistic)
    rho_c = float(spearmanr(dln_cr, published.beta_credit).statistic)
    assert rho_r >= SPEARMAN_RATE
    assert rho_g >= SPEARMAN_GROWTH
    assert rho_o >= SPEARMAN_OIL
    assert rho_c >= SPEARMAN_CREDIT

    codes = list(published.codes)
    banks = codes.index("BANKS")
    positive = [c for c, v in zip(codes, dln_rate, strict=True) if v > 0]
    assert positive == ["BANKS"], positive
    assert dln_rate[banks] > 0
    most_neg = {codes[i] for i in np.argsort(dln_rate)[:4]}
    assert {"REALESTATE", "SOFTWARE", "UTILITIES"} <= most_neg


@pytest.mark.validation
def test_q_stays_one_at_baseline_under_passthrough(config_dir, io) -> None:
    cfg = load_config(config_dir, {"dynamics.banks.mode": "passthrough"})
    prov, _real, fin = _provider(cfg, io)
    assert np.allclose(prov.q_tobin(fin.r0, fin.pi_star, 0.0), 1.0, atol=1e-12)
