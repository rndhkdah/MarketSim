from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.clock import EventQueue
from marketsim.real.cenbank import (
    CentralBank,
    is_quarter_end_month,
    seed_price_history,
    taylor_target,
)


def test_taylor_by_hand_two_cases(cfg) -> None:
    tay = cfg.edges.policy.taylor
    t1 = taylor_target(tay.r_neutral, 0.02, tay.phi_inflation, tay.phi_output_gap, 0.02, 0.0)
    assert t1 == pytest.approx(tay.r_neutral + 0.02)
    t2 = taylor_target(tay.r_neutral, 0.02, tay.phi_inflation, tay.phi_output_gap, 0.04, -0.01)
    assert t2 == pytest.approx(tay.r_neutral + 0.02 + 1.5 * 0.02 + 0.5 * (-0.01))


def test_pi12_seeded_from_month_one(cfg) -> None:
    cb = CentralBank.from_config(cfg, pi_star=0.02)
    assert cb.pi12() == pytest.approx(0.02, abs=1e-12)
    g = float(np.exp(0.02 / 12.0))
    cb.observe_prices(g, g)
    assert cb.pi12() == pytest.approx(0.02, abs=1e-12)
    assert seed_price_history(g)[-1] == pytest.approx(1.0)


def test_rate_changes_only_at_quarter_ends(cfg) -> None:
    cb = CentralBank.from_config(cfg, pi_star=0.02)
    rates = []
    for m in range(12):
        cb.maybe_meet(gap=0.05, z_mon=0.0)
        rates.append(cb.r)
        assert is_quarter_end_month(m) == ((m + 1) % 3 == 0)
    changes = [i for i in range(1, 12) if abs(rates[i] - rates[i - 1]) > 1e-15]
    assert changes == [2, 5, 8, 11]


def test_elb_binds(cfg) -> None:
    cb = CentralBank.from_config(cfg, pi_star=0.0)
    cb.elb = 0.0
    cb.r_rule = 0.01
    cb.month = 2  # force a meeting
    cb.maybe_meet(gap=-2.0, z_mon=0.0)
    assert cb.r == pytest.approx(0.0)
    assert cb.r_rule < 0.0


def test_meetings_on_event_queue(cfg) -> None:
    cb = CentralBank.from_config(cfg, pi_star=0.0)
    q = EventQueue()
    cb.schedule_meetings(q, horizon_months=12)
    due = q.pop_due(21 * 12)
    months = [p["month"] for p in due]
    assert months == [2, 5, 8, 11]
