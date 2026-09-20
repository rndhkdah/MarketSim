"""T6.11 — thin engine quote + queue-reactive-lite; book never empty."""

from __future__ import annotations

import pytest

from marketsim.core.config import Config
from marketsim.core.rng import RngHub
from marketsim.market.clob import CLOB, Side
from marketsim.market.clob_liquidity import (
    DAYS_PER_YEAR,
    ClobLiquidity,
    snap_tick,
)
from marketsim.pricing.firm_value import thin_quote_feed


def _pair(cfg: Config, *, float_shares: int, symbol: str = "EQ:FIRM:acme", ref: float = 10.0):
    assert cfg.markets is not None
    clob = CLOB.from_clob_cfg(cfg.markets.clob)
    clob.add_book(symbol, reference_price=ref)
    eng = ClobLiquidity.attach(clob, symbol, float_shares=float_shares, cfg=cfg.markets.clob)
    return clob, eng


def test_engine_quote_tracks_fundamental_within_w(cfg: Config) -> None:
    assert cfg.markets is not None
    clob, eng = _pair(cfg, float_shares=1_000_000)
    v = 10.0
    eng.refresh_quote(v)
    book = clob.book(eng.symbol)
    lo, hi = thin_quote_feed(v, w=cfg.markets.clob.thin_quote_w)
    assert book.best_bid is not None and book.best_ask is not None
    assert book.best_bid == pytest.approx(snap_tick(lo, cfg.markets.clob.tick, side=Side.BUY))
    assert book.best_ask == pytest.approx(snap_tick(hi, cfg.markets.clob.tick, side=Side.SELL))
    assert eng.quote_inside_band(v)
    eng.refresh_quote(12.0)
    assert eng.quote_inside_band(12.0)


def test_depth_proportional_to_float(cfg: Config) -> None:
    _, small = _pair(cfg, float_shares=500_000, symbol="EQ:FIRM:s")
    _, big = _pair(cfg, float_shares=1_000_000, symbol="EQ:FIRM:b")
    small.refresh_quote(10.0)
    big.refresh_quote(10.0)
    assert big.depth_shares() == 2 * small.depth_shares()
    assert big.quote_qty == 2 * small.quote_qty


def test_book_never_empty_over_10_years(cfg: Config) -> None:
    clob, eng = _pair(cfg, float_shares=200_000)
    rng = RngHub(4)
    v = 10.0
    eng.refresh_quote(v)
    n = 10 * DAYS_PER_YEAR
    for t in range(n):
        v = 10.0 * (1.0 + 0.0001 * ((t % 21) - 10))
        eng.refresh_quote(v)
        eng.step_qr(rng, v)
        clob.end_tick()
        book = clob.book(eng.symbol)
        assert book.best_bid is not None
        assert book.best_ask is not None
        assert book.best_bid < book.best_ask
