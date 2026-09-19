from __future__ import annotations

import numpy as np
import pytest

from marketsim.ledger.opening import open_passthrough_books
from marketsim.ledger.sfc import assert_consistent
from marketsim.real.row import exports, import_bill, post_trade, row_nfa, trade_balance
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_trade_balanced_at_baseline(cfg, io, pi_star: float) -> None:
    real = compute_real_baseline(io, cfg)
    g = float(np.exp(pi_star / 12.0))
    p = np.full(real.S, g)
    p_imp = g
    ex = exports(real.flat(real.X0), p, p_imp, cfg.dynamics.row.export_price_elasticity)
    imp = import_bill(p_imp, real.flat(real.m), real.flat(real.x0))
    assert trade_balance(p, ex, imp) == pytest.approx(0.0, abs=1e-9)


def test_ten_percent_prices_cut_exports_7_3(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    p = np.full(real.S, 1.10)
    ex = exports(real.flat(real.X0), p, 1.0, 0.8)
    rel = ex / real.flat(real.X0)
    assert np.allclose(rel, 1.10 ** (-0.8), atol=1e-12)
    assert float(rel.mean()) == pytest.approx(1.10 ** (-0.8), abs=1e-12)
    assert float(rel.mean() - 1.0) == pytest.approx(-0.073, abs=0.001)


def test_row_position_equals_minus_cum_trade(cfg, io) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0)
    led = open_passthrough_books(cfg, real, fin)
    p = np.ones(real.S)
    cum = 0.0
    for t, scale in enumerate((1.0, 0.9, 1.1), start=1):
        ex = exports(real.flat(real.X0) * scale, p, 1.0, 0.8)
        imp = import_bill(1.0, real.flat(real.m), real.flat(real.x0))
        tb = post_trade(led, real, p=p, ex_real=ex, imp_nom=imp, tick=t)
        cum += tb
        assert_consistent(led, t)
        assert row_nfa(led) == pytest.approx(-cum, abs=1e-8)
