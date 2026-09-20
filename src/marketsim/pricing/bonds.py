"""Bond-index holding return and extra corporate-pool tenors (§6.3 / T8.01)."""

from __future__ import annotations

from marketsim.firms.financing import pool_funding_rate
from marketsim.market.corp_pool import BOND_DURATION_Y, NOTE_DURATION_Y, POOL_DURATION_Y

# config/sectors.yaml bond_index.duration — published bond-index / GB_BOND duration (years).
BOND_INDEX_DURATION = 7.0
# Annual yield → month (existing bond_index_return carry).
MONTHS_PER_YEAR = 12.0
# T8.01 extra pool tenors besides 4.2y CORP_POOL (years). Same NOTE / BOND
# anchors as ``y_match`` — imported from corp_pool.py, not retyped.
EXTRA_POOL_TENORS_Y: tuple[float, ...] = (NOTE_DURATION_Y, BOND_DURATION_Y)
# Full curve: extra tenors + the existing 4.2y CORP_POOL point (years).
POOL_TENORS_Y: tuple[float, ...] = (NOTE_DURATION_Y, POOL_DURATION_Y, BOND_DURATION_Y)


def bond_index_return(
    y_prev: float,
    y: float,
    *,
    duration: float = BOND_INDEX_DURATION,
    carry: float | None = None,
) -> float:
    """Bond-index holding return ``= −D · Δy + carry``.

    Units: ``y_prev`` and ``y`` are annual-decimal yields; ``duration`` is years
    (default 7.0 = ``bond_index.duration``). ``carry`` is a monthly decimal
    return; default is the previous yield / 12 (annual yield → month).
    Returns a monthly holding return (decimal).
    """
    dy = y - y_prev
    if carry is None:
        carry = y_prev / MONTHS_PER_YEAR  # annual yield → month
    return float(-duration * dy + carry)


def interpolate_govt_yield(y_note: float, y_bond: float, tenor_y: float) -> float:
    """Government yield at ``tenor_y`` years, linear in NOTE / BOND (same as ``y_match``).

    ``y_note``, ``y_bond`` and the return are annual decimals. Anchors are
    ``NOTE_DURATION_Y`` = 2.7y and ``BOND_DURATION_Y`` = 7.1y (§6.11 / T8.01).
    At ``tenor_y = POOL_DURATION_Y`` (4.2y) this equals ``y_match``.
    """
    span = BOND_DURATION_Y - NOTE_DURATION_Y
    weight = (float(tenor_y) - NOTE_DURATION_Y) / span
    return float(y_note) + weight * (float(y_bond) - float(y_note))


def govt_yield_at_tenor(tenor_y: float, y_note: float, y_bond: float) -> float:
    """Alias of :func:`interpolate_govt_yield` with tenor first (T8.01). Annual decimal."""
    return interpolate_govt_yield(y_note, y_bond, tenor_y)


def pool_funding_at_tenor(
    y_note: float,
    y_bond: float,
    spread: float,
    tenor_y: float,
) -> float:
    """Pool funding rate at ``tenor_y``: ``y_tenor + s_t`` (annual decimal).

    No firm / sector / region / rating term (D14 / T8.01).
    """
    y_tenor = interpolate_govt_yield(y_note, y_bond, tenor_y)
    return pool_funding_rate(y_tenor, spread)


def pool_rate_at_tenor(
    tenor_y: float,
    y_note: float,
    y_bond: float,
    spread: float,
) -> float:
    """Alias of :func:`pool_funding_at_tenor` with tenor first (T8.01). Annual decimal."""
    return pool_funding_at_tenor(y_note, y_bond, spread, tenor_y)


def pool_maturity_curve(
    y_note: float,
    y_bond: float,
    spread: float,
) -> tuple[tuple[float, float], ...]:
    """``((tenor_years, pool_rate_annual), ...)`` at extra tenors plus 4.2y CORP_POOL.

    Ordered by tenor. The same global spread ``s_t`` applies at every point (D14).
    """
    return tuple((t, pool_funding_at_tenor(y_note, y_bond, spread, t)) for t in POOL_TENORS_Y)
