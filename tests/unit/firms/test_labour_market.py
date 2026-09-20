"""T5.08 — vacancy matching, wage ranking, firing cost, pool cap."""

from __future__ import annotations

import pytest

from marketsim.firms.firm import Firm, FirmsFile
from marketsim.firms.labour_market import fire, match


def _cfg() -> FirmsFile:
    return FirmsFile()


def test_hires_cannot_exceed_pool() -> None:
    cfg = _cfg()
    firms = [
        Firm(id="a", operator="x", vacancies=100.0, wage_offer=1.0, employees=5.0),
        Firm(id="b", operator="y", vacancies=100.0, wage_offer=1.0, employees=5.0),
    ]
    res = match(firms, cfg, unemployed=10.0, wage_bar=1.0, region="INDUSTRIAL")
    assert sum(res.hires.values()) == pytest.approx(min(10.0, 0.10 * 10.0 * 2), abs=1e-9)
    assert sum(res.hires.values()) <= 10.0 + 1e-12
    assert res.pool_left >= -1e-12


def test_higher_wage_fills_faster() -> None:
    cfg = _cfg()
    firms = [
        Firm(id="lo", operator="x", vacancies=10.0, wage_offer=1.0, employees=5.0),
        Firm(id="hi", operator="y", vacancies=10.0, wage_offer=1.2, employees=5.0),
    ]
    res = match(firms, cfg, unemployed=200.0, wage_bar=1.0, region="INDUSTRIAL")
    assert res.hires["hi"] > res.hires["lo"]


def test_firing_cost_charged() -> None:
    firm = Firm(id="a", operator="x", employees=10.0, wage_offer=2.0)
    cost = fire(firm, _cfg(), 3.0)
    assert cost == pytest.approx(3.0 * 2.0 * 2.0)
    assert firm.employees == pytest.approx(7.0)


def test_unbounded_vacancies_capped() -> None:
    cfg = _cfg()
    firms = [Firm(id="a", operator="x", vacancies=1e9, wage_offer=1.0, employees=1.0)]
    res = match(firms, cfg, unemployed=20.0, wage_bar=1.0, region="INDUSTRIAL")
    assert res.hires["a"] <= 0.10 * 20.0 + 1e-12
    assert res.hires["a"] <= 20.0
