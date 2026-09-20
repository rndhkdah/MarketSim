"""T5.06 — FirmDecision limits, partial autopilot, control check."""

from __future__ import annotations

import math

import pytest

from marketsim.core.errors import ConfigError
from marketsim.firms.firm import Firm, FirmsFile, Plant
from marketsim.firms.levers import FirmDecision, validate_decision


def _firm() -> Firm:
    return Firm(
        id="acme",
        operator="alice",
        plants=(Plant("INDUSTRIAL", "AUTOS", 10.0),),
        employees=8.0,
        posted_price={("INDUSTRIAL", "AUTOS"): 1.0},
        credit_limit=20.0,
        credit_drawn=5.0,
    )


def _cfg() -> FirmsFile:
    return FirmsFile()


def test_price_step_and_floor() -> None:
    firm, cfg = _firm(), _cfg()
    cell = ("INDUSTRIAL", "AUTOS")
    raw = FirmDecision("acme", "alice", posted_price={cell: 2.0})
    out = validate_decision(raw, firm, cfg, p_ref={cell: 1.0})
    hi = math.exp(cfg.pricing.step_max_month)
    assert out.decision.posted_price[cell] == pytest.approx(hi)
    assert out.adjustments[0].reason == "step_or_floor"
    lo = math.exp(-cfg.pricing.step_max_month)
    raw = FirmDecision("acme", "alice", posted_price={cell: 0.0})
    out = validate_decision(raw, firm, cfg, p_ref={cell: 1.0})
    assert out.decision.posted_price[cell] == pytest.approx(lo)
    firm.posted_price[cell] = cfg.pricing.floor_of_pref * 1.05
    out = validate_decision(FirmDecision("acme", "alice", posted_price={cell: 0.0}), firm, cfg, p_ref={cell: 1.0})
    assert out.decision.posted_price[cell] == pytest.approx(cfg.pricing.floor_of_pref)


def test_overtime_and_hire_and_premium_and_leverage() -> None:
    firm, cfg = _firm(), _cfg()
    cell = ("INDUSTRIAL", "AUTOS")
    raw = FirmDecision(
        "acme",
        "alice",
        target_output={cell: 50.0},
        vacancies=100.0,
        shortage_premium=0.9,
        borrow=100.0,
        expand_capacity={cell: 100.0},
        input_cover_m=12.0,
    )
    out = validate_decision(raw, firm, cfg, unemployed=20.0)
    assert out.decision.target_output[cell] == pytest.approx(10.0 * cfg.production.overtime_cap)
    assert out.decision.vacancies == pytest.approx(0.10 * 20.0)
    assert out.decision.shortage_premium == pytest.approx(0.5)
    assert out.decision.borrow == pytest.approx(15.0)
    assert out.decision.expand_capacity[cell] == pytest.approx(0.50 * 10.0)
    assert out.decision.input_cover_m == pytest.approx(6.0)
    assert {a.lever for a in out.adjustments} >= {"production", "labour", "procurement", "financing", "capex"}


def test_partial_leaves_other_levers_on_autopilot() -> None:
    out = validate_decision(FirmDecision("acme", "alice", posted_price={("INDUSTRIAL", "AUTOS"): 1.02}), _firm(), _cfg())
    assert out.autopilot["pricing"] is False
    assert out.autopilot["production"] is True
    assert out.autopilot["financing"] is True
    assert out.decision.target_output is None


def test_foreign_operator_rejected() -> None:
    with pytest.raises(ConfigError, match="does not control"):
        validate_decision(
            FirmDecision("acme", "alice", posted_price={("INDUSTRIAL", "AUTOS"): 1.0}),
            _firm(),
            _cfg(),
            controller="mallory",
        )
