"""T5.19 — collusion flag vs competitive autopilots."""

from __future__ import annotations

from marketsim.scenarios.evaluation import evaluate_margins


def test_collusion_flagged() -> None:
    # two firms holding a 15 % margin vs 3 % competitive for a year
    collude = [0.15] * 18
    rep = evaluate_margins(collude, competitive_margin=0.03, persist_months=12, excess=0.05)
    assert rep.flagged is True


def test_competitive_autopilot_not_flagged() -> None:
    competitive = [0.03 + 0.01 * ((i % 3) - 1) for i in range(24)]
    rep = evaluate_margins(competitive, competitive_margin=0.03, persist_months=12, excess=0.05)
    assert rep.flagged is False
