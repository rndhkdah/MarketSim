from __future__ import annotations

import numpy as np
import pytest

from marketsim.ledger.opening import open_passthrough_books
from marketsim.real.households import (
    consumption_nominal,
    household_basket,
    income_index,
    wealth_from_ledger,
)
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def _setup(cfg, io, pi_star=0.0):
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=pi_star)
    demand = household_basket(real, cfg, fin)
    return real, fin, demand


def test_baseline_basket_matches_household_fd(cfg, io) -> None:
    real, fin, demand = _setup(cfg, io)
    led = open_passthrough_books(cfg, real, fin)
    w = wealth_from_ledger(led)
    assert w == pytest.approx(fin.W, abs=1e-8)
    c_nom = consumption_nominal(cfg.dynamics.households.alpha1, fin.YD0, fin.alpha2, w)
    assert c_nom == pytest.approx(float(real.flat(real.C0).sum()), abs=1e-9)
    y = income_index(fin.YD0, 1.0, fin.YD0)
    qty = demand.allocate(c_nom, np.ones(real.S), y, 0.0)
    assert np.allclose(qty, real.flat(real.C0), atol=1e-9)
    assert np.allclose(demand.theta, io.final_demand["HOUSEHOLD"] / io.final_demand["HOUSEHOLD"].sum(), atol=1e-12)


def test_shares_sum_to_one(cfg, io) -> None:
    real, fin, demand = _setup(cfg, io)
    qty = demand.allocate(100.0, np.ones(real.S), 1.0, 0.0)
    shares = qty / qty.sum()
    assert shares.sum() == pytest.approx(1.0, abs=1e-12)


def test_income_elasticity_of_share_follows_eta(cfg, io) -> None:
    real, fin, demand = _setup(cfg, io)
    p = np.ones(real.S)
    s0 = demand.allocate(1.0, p, 1.00, 0.0)
    s1 = demand.allocate(1.0, p, 1.05, 0.0)
    sh0, sh1 = s0 / s0.sum(), s1 / s1.sum()
    elas = (sh1 - sh0) / np.maximum(sh0, 1e-12) / 0.05
    eta = np.array([cfg.sectors.params(c).eta for c in real.codes])
    for code, e, el in zip(real.codes, eta, elas, strict=True):
        if abs(e - 1.0) < 1e-9:
            assert abs(el) < 0.05, code
        else:
            assert np.sign(el) == np.sign(e - 1.0), (code, e, el)


def test_autos_falls_most_for_rate_gap(cfg, io) -> None:
    real, fin, demand = _setup(cfg, io)
    p = np.ones(real.S)
    base = demand.allocate(1.0, p, 1.0, 0.0)
    shocked = demand.allocate(1.0, p, 1.0, 1.0)
    rel = shocked / np.maximum(base, 1e-12) - 1.0
    assert real.codes[int(np.argmin(rel))] == "AUTOS"
