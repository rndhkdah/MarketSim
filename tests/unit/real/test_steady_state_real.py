from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.steady_state import compute_real_baseline


def test_zero_leak_matches_leontief(cfg, io) -> None:
    b = compute_real_baseline(io, cfg, leak_override=np.zeros(io.n))
    x0 = b.flat(b.x0)
    d0 = b.flat(b.d0)
    assert np.allclose(x0, io.L @ d0, atol=1e-9)
    assert x0.sum() == pytest.approx(209.3, abs=0.1)


def test_trade_balanced(cfg, io) -> None:
    b = compute_real_baseline(io, cfg)
    m, x0, x_fd = b.flat(b.m), b.flat(b.x0), b.flat(b.X0)
    assert abs(float((m * x0).sum()) - float(x_fd.sum())) < 1e-9


def test_capex_identity(cfg, io) -> None:
    b = compute_real_baseline(io, cfg)
    delta_m = cfg.dynamics.capex.delta_annual / 12.0
    spent = float((delta_m * b.flat(b.v) * b.flat(b.K0)).sum())
    assert spent == pytest.approx(b.I_bus0, abs=1e-9)


def test_markup_unit_cost(cfg, io) -> None:
    b = compute_real_baseline(io, cfg)
    unit = io.A.sum(axis=0) + b.flat(b.ell) + b.flat(b.m)
    assert np.allclose(b.flat(b.markup) * unit, 1.0, atol=1e-9)


def test_wage_share_before_and_after_import_carveout(cfg, io) -> None:
    b = compute_real_baseline(io, cfg, leak_override=np.zeros(io.n))
    before = b.wages_pre_import / b.va_pre_import
    after = b.wages_post_import / b.va_post_import
    assert before == pytest.approx(0.518, abs=0.01)
    # After carve-out wages and VA both shrink by the import bill; share stays ws-weighted.
    assert after == pytest.approx(before, abs=0.02)
    # Documented after-value for the card (not a golden).
    assert b.wages_post_import < b.wages_pre_import
