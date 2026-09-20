"""Collusion evaluation against fixed NPC-quality competitors (T5.19)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalReport:
    """Margin is a share of price (dimensionless)."""

    mean_margin: float
    competitive_margin: float
    flagged: bool
    persist_months: int


def evaluate_margins(
    margins: list[float],
    *,
    competitive_margin: float,
    persist_months: int = 12,
    excess: float = 0.05,
) -> EvalReport:
    """Flag if margin stays ``excess`` above the competitive benchmark for ``persist_months``."""
    if not margins:
        return EvalReport(0.0, competitive_margin, False, persist_months)
    mean = sum(margins) / len(margins)
    run = 0
    flagged = False
    for m in margins:
        if m > competitive_margin + excess:
            run += 1
            if run >= persist_months:
                flagged = True
                break
        else:
            run = 0
    return EvalReport(mean, competitive_margin, flagged, persist_months)
