"""T6.21 — Tier-1 return statistics (gate 4)."""

from __future__ import annotations

import numpy as np

from marketsim.core.rng import RngHub
from marketsim.market.background import BackgroundFlow
from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import FlowCfg
from marketsim.market.metrics import DAYS_PER_YEAR, acf, pearson_kurtosis
from marketsim.pricing.mispricing import Mispricing, cap_weighted_index

# §6.3 news term: sparse public headlines (≈ monthly), size in log-ξ.
NEWS_P = 12.0 / DAYS_PER_YEAR
NEWS_SIGMA = 0.03
# Clustered participation: 15-day meta-orders (§6.4 / gate 3) so |r| clusters.
BURST_P = 0.01
BURST_MULT = 10.0
BURST_LEN = 28


def _index_returns(seed: int, years: int = 16) -> np.ndarray:
    rng = RngHub(seed)
    n = 6
    idio = np.linspace(0.12, 0.18, n)
    w = np.linspace(0.25, 0.08, n)
    w = w / w.sum()
    book = Mispricing(idio, w)
    flow = BackgroundFlow(
        tuple(f"EQ:{i}" for i in range(n)),
        np.full(n, 100.0),
        FlowCfg(),
    )
    kn = ImpactKernel()
    prev = 1.0
    rets: list[float] = []
    n_live = years * DAYS_PER_YEAR
    flow_rng = rng.stream("flow")
    burst_left = 0
    for i in range(DAYS_PER_YEAR + n_live):
        q = flow.step(rng, book.xi)
        if burst_left <= 0 and flow_rng.random() < BURST_P:
            burst_left = BURST_LEN
        if burst_left > 0:
            # Same intensity cluster, fresh signs — |r| clusters without return momentum.
            q = np.sign(flow_rng.standard_normal(q.size)) * np.abs(q) * BURST_MULT
            burst_left -= 1
        news = 0.0
        if flow_rng.random() < NEWS_P:
            news = NEWS_SIGMA * float(flow_rng.standard_normal())
        xi_i = float(kn.step(float(q.mean()), 100.0, 0.02))
        xi = book.step(rng, impact=xi_i, news=news)
        level = cap_weighted_index(xi, book.weights)
        if i >= DAYS_PER_YEAR:
            rets.append(level / prev - 1.0)
        prev = level
    return np.asarray(rets, dtype=float)


def test_gate4_kurtosis_abs_acf_and_return_acf() -> None:
    r = _index_returns(3)
    assert pearson_kurtosis(r) > 4.0
    assert abs(acf(r, 1)) < 0.1
    abs_r = np.abs(r)
    for lag in range(1, 21):
        assert acf(abs_r, lag) > 0.0
