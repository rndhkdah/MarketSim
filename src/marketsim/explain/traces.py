"""Exact explainability traces (T8.06 / §8.2).

A firm's realised demand is the product
``qty = cell_demand × share × rationing``.
``cell_demand`` itself is the product of a baseline and the named shifters
(income, relative price, rate, typed-edge, want, events). ``share`` is the
product of own-price-vs-reference, availability, and stickiness.

An asset log-return is
``Δ ln P = Δ ln V + Δξ`` with ``ln V = ln E^e + ln PE0 − D·Δρ + D·Δg_lr``
and ``ξ = I + s + n`` (impact, sentiment, noise). Components are constructed
so they sum to the total within 1e-9.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Identity tolerance (dimensionless log-points / relative qty). T8.06.
TRACE_ATOL = 1e-9


@dataclass(frozen=True)
class DemandTrace:
    """One firm/cell demand identity. ``qty`` is units / month; factors are dimensionless."""

    baseline: float
    income: float
    relative_price: float
    rate: float
    typed_edge: float
    want_shifter: float
    events: float
    cell_demand: float
    own_price: float
    availability: float
    stickiness: float
    share: float
    rationing: float
    qty: float

    def to_state(self) -> dict[str, float]:
        """JSON-safe factors. ``qty`` is units / month; others are dimensionless."""
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class ReturnTrace:
    """One-asset log-return identity. Every field is a log-point (dimensionless)."""

    earnings: float
    pe0: float
    discount: float
    growth: float
    dln_v: float
    impact: float
    sentiment: float
    noise: float
    dxi: float
    dln_p: float

    def to_state(self) -> dict[str, float]:
        """JSON-safe log-points (dimensionless)."""
        return {k: float(v) for k, v in asdict(self).items()}


def _prod(parts: tuple[float, ...]) -> float:
    out = 1.0
    for value in parts:
        out *= float(value)
    return out


def demand_trace(
    *,
    baseline: float,
    income: float = 1.0,
    relative_price: float = 1.0,
    rate: float = 1.0,
    typed_edge: float = 1.0,
    want_shifter: float = 1.0,
    events: float = 1.0,
    own_price: float = 1.0,
    availability: float = 1.0,
    stickiness: float = 1.0,
    rationing: float = 1.0,
) -> DemandTrace:
    """Build the demand identity. Factors are dimensionless; ``baseline`` is units / month."""
    cell = _prod((baseline, income, relative_price, rate, typed_edge, want_shifter, events))
    share = _prod((own_price, availability, stickiness))
    qty = cell * share * float(rationing)
    return DemandTrace(
        baseline=float(baseline),
        income=float(income),
        relative_price=float(relative_price),
        rate=float(rate),
        typed_edge=float(typed_edge),
        want_shifter=float(want_shifter),
        events=float(events),
        cell_demand=cell,
        own_price=float(own_price),
        availability=float(availability),
        stickiness=float(stickiness),
        share=share,
        rationing=float(rationing),
        qty=qty,
    )


def demand_change(
    before: DemandTrace,
    after: DemandTrace,
) -> dict[str, float]:
    """Log-change of each factor. Sum of log-factors equals ``Δ log qty`` within ``TRACE_ATOL``."""
    keys = (
        "baseline",
        "income",
        "relative_price",
        "rate",
        "typed_edge",
        "want_shifter",
        "events",
        "own_price",
        "availability",
        "stickiness",
        "rationing",
    )
    out = {key: float(_dlog(getattr(after, key), getattr(before, key))) for key in keys}
    out["dlog_cell"] = float(_dlog(after.cell_demand, before.cell_demand))
    out["dlog_share"] = float(_dlog(after.share, before.share))
    out["dlog_qty"] = float(_dlog(after.qty, before.qty))
    return out


def asset_return_trace(
    *,
    ee0: float,
    ee1: float,
    pe0_0: float,
    pe0_1: float,
    duration: float,
    rho0: float,
    rho1: float,
    g0: float = 0.0,
    g1: float = 0.0,
    impact0: float = 0.0,
    impact1: float = 0.0,
    sentiment0: float = 0.0,
    sentiment1: float = 0.0,
    noise0: float = 0.0,
    noise1: float = 0.0,
) -> ReturnTrace:
    """``Δ ln P`` from fundamentals + ξ. ``ee`` is cr / year; ``duration`` is years; ρ, g annual."""
    import math

    earnings = math.log(float(ee1) / float(ee0))
    pe0 = math.log(float(pe0_1) / float(pe0_0))
    discount = -float(duration) * (float(rho1) - float(rho0))
    growth = float(duration) * (float(g1) - float(g0))
    dln_v = earnings + pe0 + discount + growth
    impact = float(impact1) - float(impact0)
    sentiment = float(sentiment1) - float(sentiment0)
    noise = float(noise1) - float(noise0)
    dxi = impact + sentiment + noise
    return ReturnTrace(
        earnings=earnings,
        pe0=pe0,
        discount=discount,
        growth=growth,
        dln_v=dln_v,
        impact=impact,
        sentiment=sentiment,
        noise=noise,
        dxi=dxi,
        dln_p=dln_v + dxi,
    )


def check_demand_identity(trace: DemandTrace, *, atol: float = TRACE_ATOL) -> None:
    """Raise ``AssertionError`` if the product identity misses ``atol``."""
    cell = _prod(
        (
            trace.baseline,
            trace.income,
            trace.relative_price,
            trace.rate,
            trace.typed_edge,
            trace.want_shifter,
            trace.events,
        )
    )
    share = _prod((trace.own_price, trace.availability, trace.stickiness))
    qty = cell * share * trace.rationing
    if abs(cell - trace.cell_demand) > atol:
        raise AssertionError(f"cell_demand {trace.cell_demand} != {cell}")
    if abs(share - trace.share) > atol:
        raise AssertionError(f"share {trace.share} != {share}")
    if abs(qty - trace.qty) > atol:
        raise AssertionError(f"qty {trace.qty} != {qty}")


def check_return_identity(trace: ReturnTrace, *, atol: float = TRACE_ATOL) -> None:
    """Raise ``AssertionError`` if log-return parts miss ``atol``."""
    dln_v = trace.earnings + trace.pe0 + trace.discount + trace.growth
    dxi = trace.impact + trace.sentiment + trace.noise
    dln_p = dln_v + dxi
    if abs(dln_v - trace.dln_v) > atol:
        raise AssertionError(f"dln_v {trace.dln_v} != {dln_v}")
    if abs(dxi - trace.dxi) > atol:
        raise AssertionError(f"dxi {trace.dxi} != {dxi}")
    if abs(dln_p - trace.dln_p) > atol:
        raise AssertionError(f"dln_p {trace.dln_p} != {dln_p}")


def check_demand_change(before: DemandTrace, after: DemandTrace, *, atol: float = TRACE_ATOL) -> None:
    """Log-factor sum equals ``Δ log qty``."""
    parts = demand_change(before, after)
    keys = (
        "baseline",
        "income",
        "relative_price",
        "rate",
        "typed_edge",
        "want_shifter",
        "events",
        "own_price",
        "availability",
        "stickiness",
        "rationing",
    )
    total = sum(parts[key] for key in keys)
    if abs(total - parts["dlog_qty"]) > atol:
        raise AssertionError(f"sum {total} != dlog_qty {parts['dlog_qty']}")
    cell_keys = ("baseline", "income", "relative_price", "rate", "typed_edge", "want_shifter", "events")
    if abs(sum(parts[k] for k in cell_keys) - parts["dlog_cell"]) > atol:
        raise AssertionError("cell log-factors do not sum")
    share_keys = ("own_price", "availability", "stickiness")
    if abs(sum(parts[k] for k in share_keys) - parts["dlog_share"]) > atol:
        raise AssertionError("share log-factors do not sum")


def _dlog(new: float, old: float) -> float:
    import math

    if old <= 0.0 or new <= 0.0:
        return 0.0
    return math.log(float(new) / float(old))


def default_demand_trace() -> DemandTrace:
    """Baseline identity (1 unit / month, all factors 1)."""
    return demand_trace(baseline=1.0)


def demand_identity_gap(trace: DemandTrace) -> float:
    """Max absolute miss of the product identity (dimensionless / units)."""
    cell = _prod(
        (
            trace.baseline,
            trace.income,
            trace.relative_price,
            trace.rate,
            trace.typed_edge,
            trace.want_shifter,
            trace.events,
        )
    )
    share = _prod((trace.own_price, trace.availability, trace.stickiness))
    qty = cell * share * trace.rationing
    return max(
        abs(cell - trace.cell_demand),
        abs(share - trace.share),
        abs(qty - trace.qty),
    )


def return_identity_gap(trace: ReturnTrace) -> float:
    """Max absolute miss of the log-sum identity (log-points)."""
    dln_v = trace.earnings + trace.pe0 + trace.discount + trace.growth
    dxi = trace.impact + trace.sentiment + trace.noise
    dln_p = dln_v + dxi
    return max(abs(dln_v - trace.dln_v), abs(dxi - trace.dxi), abs(dln_p - trace.dln_p))


def traces_payload(demand: DemandTrace, ret: ReturnTrace) -> dict[str, Any]:
    """Wire body for GET …/explain. Units: see the dataclasses."""
    return {"demand": demand.to_state(), "asset_return": ret.to_state()}
