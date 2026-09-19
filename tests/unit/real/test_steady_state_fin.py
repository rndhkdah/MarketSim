from __future__ import annotations

import pytest

from marketsim.ledger.opening import open_passthrough_books, opening_wealth
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_payout_alpha2_and_sfc(cfg, io, pi_star: float) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=pi_star)
    assert fin.alpha2 > 0
    assert ((fin.payout > 0.3) & (fin.payout < 0.8)).all(), fin.payout
    led = open_passthrough_books(cfg, real, fin)
    assert opening_wealth(led) == pytest.approx(fin.W, abs=1e-8)
    hh_saving = fin.grow * fin.W
    deficit = fin.grow * fin.B
    firm_borrow = fin.grow * float(fin.debt.sum())
    trade = 0.0
    assert hh_saving == pytest.approx(deficit + firm_borrow + trade, abs=1e-9)


@pytest.mark.parametrize("vat", [0.0, 0.1])
def test_vat_solvable(cfg, io, vat: float) -> None:
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io, pi_star=0.0, vat=vat)
    assert fin.alpha2 > 0
    assert 0.0 < fin.tau_y < 0.6
    assert ((fin.payout > 0.3) & (fin.payout < 0.8)).all()
