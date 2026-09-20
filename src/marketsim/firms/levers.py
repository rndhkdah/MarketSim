"""FirmDecision validation and clipping (T5.06 / §5.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from marketsim.core.errors import ConfigError
from marketsim.firms.firm import LEVERS, Firm, FirmsFile

Timing = Literal["month_boundary", "next_tick"]

LEVER_TIMING: dict[str, Timing] = {
    "pricing": "month_boundary",
    "production": "month_boundary",
    "labour": "month_boundary",
    "procurement": "month_boundary",
    "capex": "month_boundary",
    "financing": "next_tick",
    "treasury": "next_tick",
    "exit": "month_boundary",
}


@dataclass
class Adjustment:
    """One clip / reject. ``requested`` and ``applied`` keep the lever's native units."""

    lever: str
    field: str
    requested: Any
    applied: Any
    reason: str


@dataclass
class FirmDecision:
    """Every economic field is optional; omitted levers stay on autopilot."""

    firm_id: str
    operator: str
    posted_price: dict[tuple[str, str], float] | None = None  # index
    target_output: dict[tuple[str, str], float] | None = None  # units / month
    vacancies: float | None = None  # persons
    wage_offer: float | None = None  # cr / person / month
    layoffs: float | None = None  # persons
    input_cover_m: float | None = None  # months
    order_mult: float | None = None  # dimensionless
    shortage_premium: float | None = None  # 0–0.5
    expand_capacity: dict[tuple[str, str], float] | None = None  # units / year
    new_plant: tuple[str, str, float] | None = None  # region, sector, capacity
    rnd_spend: float | None = None  # cr / month
    borrow: float | None = None  # cr
    repay: float | None = None  # cr
    dividend: float | None = None  # cr
    treasury_enabled: bool | None = None
    liquidate: bool | None = None


@dataclass
class ValidatedDecision:
    decision: FirmDecision
    adjustments: list[Adjustment] = field(default_factory=list)
    autopilot: dict[str, bool] = field(default_factory=dict)


def _clip(lo: float, hi: float, value: float) -> float:
    return min(hi, max(lo, value))


def validate_decision(
    decision: FirmDecision,
    firm: Firm,
    cfg: FirmsFile,
    *,
    p_ref: dict[tuple[str, str], float] | None = None,
    cell_capacity: dict[tuple[str, str], float] | None = None,
    unemployed: float = 0.0,
    controller: str | None = None,
) -> ValidatedDecision:
    """Clip to §5.4 limits. Rejects decisions the operator does not control."""
    if controller is not None and controller != firm.operator:
        raise ConfigError(f"agent {controller!r} does not control firm {firm.id}")
    if decision.firm_id != firm.id:
        raise ConfigError(f"decision firm {decision.firm_id!r} != {firm.id}")
    if decision.operator != firm.operator:
        raise ConfigError(f"decision operator {decision.operator!r} != {firm.operator}")

    adj: list[Adjustment] = []
    d = FirmDecision(firm_id=decision.firm_id, operator=decision.operator)
    auto = {k: True for k in LEVERS}

    if decision.posted_price is not None:
        auto["pricing"] = False
        applied: dict[tuple[str, str], float] = {}
        for cell, price in decision.posted_price.items():
            ref = (p_ref or {}).get(cell, firm.posted_price.get(cell, 1.0))
            floor = cfg.pricing.floor_of_pref * ref
            prev = firm.posted_price.get(cell, ref)
            lo = prev * float(np_exp(-cfg.pricing.step_max_month))
            hi = prev * float(np_exp(cfg.pricing.step_max_month))
            clipped = _clip(max(floor, lo), hi, float(price))
            if clipped != float(price):
                adj.append(Adjustment("pricing", f"posted_price{cell}", price, clipped, "step_or_floor"))
            applied[cell] = clipped
        d.posted_price = applied

    if decision.target_output is not None:
        auto["production"] = False
        applied_x: dict[tuple[str, str], float] = {}
        cap_by = {p.cell: p.capacity for p in firm.plants}
        for cell, x in decision.target_output.items():
            cap = cap_by.get(cell, 0.0) * cfg.production.overtime_cap
            clipped = _clip(0.0, cap, float(x))
            if clipped != float(x):
                adj.append(Adjustment("production", f"target_output{cell}", x, clipped, "overtime_cap"))
            applied_x[cell] = clipped
        d.target_output = applied_x

    if decision.vacancies is not None or decision.wage_offer is not None or decision.layoffs is not None:
        auto["labour"] = False
        hire_cap = cfg.labour.hire_share_of_unemployed * max(unemployed, 0.0)
        if decision.vacancies is not None:
            clipped = _clip(0.0, hire_cap if hire_cap > 0 else decision.vacancies, float(decision.vacancies))
            if hire_cap > 0 and clipped != float(decision.vacancies):
                adj.append(Adjustment("labour", "vacancies", decision.vacancies, clipped, "hire_share"))
            d.vacancies = clipped
        if decision.wage_offer is not None:
            d.wage_offer = max(float(decision.wage_offer), 0.0)
            if d.wage_offer != float(decision.wage_offer):
                adj.append(Adjustment("labour", "wage_offer", decision.wage_offer, d.wage_offer, "nonneg"))
        if decision.layoffs is not None:
            clipped = _clip(0.0, firm.employees, float(decision.layoffs))
            if clipped != float(decision.layoffs):
                adj.append(Adjustment("labour", "layoffs", decision.layoffs, clipped, "headcount"))
            d.layoffs = clipped

    if (
        decision.input_cover_m is not None
        or decision.order_mult is not None
        or decision.shortage_premium is not None
    ):
        auto["procurement"] = False
        if decision.input_cover_m is not None:
            clipped = _clip(0.0, cfg.procurement.input_cover_max_m, float(decision.input_cover_m))
            if clipped != float(decision.input_cover_m):
                adj.append(Adjustment("procurement", "input_cover_m", decision.input_cover_m, clipped, "cover_max"))
            d.input_cover_m = clipped
        if decision.order_mult is not None:
            cap = cfg.controls.decision_size_cap
            clipped = _clip(0.0, cap, float(decision.order_mult))
            if clipped != float(decision.order_mult):
                adj.append(Adjustment("procurement", "order_mult", decision.order_mult, clipped, "order_size"))
            d.order_mult = clipped
        if decision.shortage_premium is not None:
            clipped = _clip(0.0, cfg.procurement.premium_max, float(decision.shortage_premium))
            if clipped != float(decision.shortage_premium):
                adj.append(Adjustment("procurement", "shortage_premium", decision.shortage_premium, clipped, "premium_max"))
            d.shortage_premium = clipped

    if decision.expand_capacity is not None or decision.new_plant is not None or decision.rnd_spend is not None:
        auto["capex"] = False
        if decision.expand_capacity is not None:
            applied_k: dict[tuple[str, str], float] = {}
            own = {p.cell: p.capacity for p in firm.plants}
            for cell, dk in decision.expand_capacity.items():
                cap = cfg.capex.capacity_growth_cap_year * own.get(cell, 0.0)
                clipped = _clip(0.0, cap if cap > 0 else float(dk), float(dk))
                if cap > 0 and clipped != float(dk):
                    adj.append(Adjustment("capex", f"expand{cell}", dk, clipped, "growth_cap"))
                applied_k[cell] = clipped
            d.expand_capacity = applied_k
        d.new_plant = decision.new_plant
        if decision.rnd_spend is not None:
            d.rnd_spend = max(float(decision.rnd_spend), 0.0)

    if decision.borrow is not None or decision.repay is not None or decision.dividend is not None:
        auto["financing"] = False
        if decision.borrow is not None:
            room = max(firm.credit_limit - firm.credit_drawn, 0.0)
            clipped = _clip(0.0, room, float(decision.borrow))
            if clipped != float(decision.borrow):
                adj.append(Adjustment("financing", "borrow", decision.borrow, clipped, "credit_line"))
            d.borrow = clipped
        if decision.repay is not None:
            clipped = _clip(0.0, firm.credit_drawn, float(decision.repay))
            if clipped != float(decision.repay):
                adj.append(Adjustment("financing", "repay", decision.repay, clipped, "drawn"))
            d.repay = clipped
        if decision.dividend is not None:
            d.dividend = max(float(decision.dividend), 0.0)

    if decision.treasury_enabled is not None:
        auto["treasury"] = False
        d.treasury_enabled = bool(decision.treasury_enabled)

    if decision.liquidate is not None:
        auto["exit"] = False
        d.liquidate = bool(decision.liquidate)

    return ValidatedDecision(decision=d, adjustments=adj, autopilot=auto)


def np_exp(x: float) -> float:
    """local exp so levers.py does not depend on numpy in the public surface."""
    import math

    return math.exp(x)
