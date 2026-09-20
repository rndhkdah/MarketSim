"""T6.23 — gate 8: 25-name market book, no agents, ≥ 3,000 ticks/s."""

from __future__ import annotations

from bench import market_bench, market_symbols


def test_gate8_25_names_3000_ticks_per_s_and_deterministic() -> None:
    assert len(market_symbols()) == 25
    a = market_bench(ticks=2520, warmup=42, seed=0)
    b = market_bench(ticks=2520, warmup=42, seed=0)
    c = market_bench(ticks=2520, warmup=42, seed=1)
    assert a["ticks_per_s"] >= 3000.0
    assert a["hash"] == b["hash"]
    assert a["hash"] != c["hash"]
    assert a["n_instruments"] == 25.0
