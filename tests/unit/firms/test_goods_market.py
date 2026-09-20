"""T5.05 — heterogeneous-seller goods market."""

from __future__ import annotations

import math

import pytest

from marketsim.firms.firm import FirmsFile
from marketsim.firms.goods_market import SellerQuote, allocate, one_seller_r6


def _cfg() -> FirmsFile:
    return FirmsFile()


def test_equal_prices_shares_proportional_to_capacity() -> None:
    cfg = _cfg()
    quotes = [
        SellerQuote("npc", 80.0, 1.0, 80.0, 0.8),
        SellerQuote("a", 20.0, 1.0, 20.0, 0.2),
    ]
    for _ in range(40):
        for q in quotes:
            q.avail = q.capacity
        allocate(quotes, 100.0, cfg)
    assert quotes[0].share == pytest.approx(0.8, abs=1e-6)
    assert quotes[1].share == pytest.approx(0.2, abs=1e-6)


def test_price_cut_gains_share_gradually() -> None:
    cfg = _cfg()
    quotes = [
        SellerQuote("hi", 50.0, 1.0, 50.0, 0.5),
        SellerQuote("lo", 50.0, 0.95, 50.0, 0.5),
    ]
    path = [quotes[1].share]
    for _ in range(12):
        for q in quotes:
            q.avail = q.capacity
        allocate(quotes, 80.0, cfg)
        path.append(quotes[1].share)
    assert path[1] > path[0]
    assert path[1] - path[0] < 0.05
    # τ=6: first-half move is larger than the second half (exponential smoother).
    assert path[6] - path[0] > path[12] - path[6]
    assert path[6] > 0.5


def test_no_knife_edge_under_price_noise() -> None:
    cfg = _cfg()
    quotes = [
        SellerQuote("a", 50.0, 1.0, 50.0, 0.5),
        SellerQuote("b", 50.0, 1.0, 50.0, 0.5),
    ]
    jumps = []
    for t in range(24):
        quotes[0].posted_price = 1.0 + 0.01 * math.sin(t)
        quotes[1].posted_price = 1.0 - 0.01 * math.sin(t)
        quotes[0].avail = quotes[0].capacity
        quotes[1].avail = quotes[1].capacity
        prev = quotes[0].share
        allocate(quotes, 90.0, cfg)
        jumps.append(abs(quotes[0].share - prev))
    assert max(jumps) < 0.03


def test_spillover_conserves_demand() -> None:
    cfg = _cfg()
    quotes = [
        SellerQuote("tight", 10.0, 1.0, 2.0, 0.5, backlog=0.0),
        SellerQuote("spare", 10.0, 1.0, 20.0, 0.5, backlog=0.0),
    ]
    res = allocate(quotes, 16.0, cfg, backlog_loss=0.10)
    assert res.delivered + sum(res.unmet.values()) == pytest.approx(16.0, abs=1e-9)
    assert res.spill > 0


def test_one_seller_matches_r6() -> None:
    cfg = _cfg()
    q = SellerQuote("only", 10.0, 1.0, 6.0, 1.0, backlog=2.0)
    res = allocate([q], 5.0, cfg, backlog_loss=0.10)
    sales, back = one_seller_r6(avail=6.0, demand=5.0, backlog=2.0, backlog_loss=0.10)
    assert res.sales["only"] == pytest.approx(sales)
    assert res.backlog["only"] == pytest.approx(back)
