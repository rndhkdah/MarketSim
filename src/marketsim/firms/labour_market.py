"""Regional vacancy matching (T5.08 / §5.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.firms.firm import Firm, FirmsFile


@dataclass
class MatchResult:
    """Persons and cr. ``hires`` / ``quits`` / ``layoffs`` are persons / month."""

    hires: dict[str, float]
    quits: dict[str, float]
    layoffs: dict[str, float]
    firing_cost: dict[str, float]
    pool_left: float


def match(
    firms: list[Firm],
    cfg: FirmsFile,
    *,
    unemployed: float,
    wage_bar: float,
    region: str,
) -> MatchResult:
    """Fill vacancies; ``Σ hires ≤`` unemployed and the per-firm hire-share cap."""
    lab = cfg.labour
    ids = [f.id for f in firms]
    v = np.array([max(f.vacancies, 0.0) for f in firms], dtype=float)
    w = np.array([max(f.wage_offer, 1e-12) for f in firms], dtype=float)
    v_tot = float(v.sum())
    u = max(unemployed, 0.0)
    if v_tot <= 0.0 or u <= 0.0:
        return MatchResult({i: 0.0 for i in ids}, {i: 0.0 for i in ids}, {i: 0.0 for i in ids}, {i: 0.0 for i in ids}, u)
    fill = min(1.0, lab.matching_m0 * (u / v_tot) ** 0.5)
    wage_term = (w / max(wage_bar, 1e-12)) ** lab.epsilon_w
    raw = v * fill * wage_term
    hire_cap = lab.hire_share_of_unemployed * u
    raw = np.minimum(raw, hire_cap)
    tot = float(raw.sum())
    if tot > u:
        raw = raw * (u / tot)
    hires = {i: float(h) for i, h in zip(ids, raw, strict=True)}
    quits = {}
    for f, wi in zip(firms, w, strict=True):
        gap = max(wage_bar / wi - 1.0, 0.0)
        quits[f.id] = float(f.employees * gap / 12.0)  # annual gap → monthly flow
    return MatchResult(hires, quits, {i: 0.0 for i in ids}, {i: 0.0 for i in ids}, u - float(sum(hires.values())))


def fire(firm: Firm, cfg: FirmsFile, n: float) -> float:
    """Lay off ``n`` persons; returns firing cost in cr."""
    laid = min(max(n, 0.0), firm.employees)
    firm.employees -= laid
    return laid * firm.wage_offer * cfg.labour.firing_cost_months
