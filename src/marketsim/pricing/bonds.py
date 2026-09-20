"""Bond-index holding return (§6.3: ``−D · Δy + carry``)."""

from __future__ import annotations

# config/sectors.yaml bond_index.duration — published bond-index / GB_BOND duration (years).
BOND_INDEX_DURATION = 7.0
# §6.11 NOTE / BOND duration anchors (years) for extra CORP_POOL tenors (T8.01).
NOTE_DURATION_Y = 2.7
BOND_DURATION_Y = 7.1
POOL_DURATION_Y = 4.2
# Extra pool tenors besides the 4.2y CORP_POOL (years). T8.01.
POOL_TENORS_Y: tuple[float, ...] = (2.0, 4.2, 7.0)


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
        carry = y_prev / 12.0  # annual yield → month
    return float(-duration * dy + carry)


def interpolate_govt_yield(y_note: float, y_bond: float, tenor_y: float) -> float:
    """Linear interpolate NOTE/BOND government yields (annual decimal). ``tenor_y`` is years."""
    span = BOND_DURATION_Y - NOTE_DURATION_Y
    weight = (float(tenor_y) - NOTE_DURATION_Y) / span
    return float(y_note) + weight * (float(y_bond) - float(y_note))


def pool_funding_at_tenor(
    y_note: float,
    y_bond: float,
    spread: float,
    tenor_y: float,
) -> float:
    """Common pool rate at ``tenor_y`` years: interpolated govt yield + ``s_t`` (D14)."""
    return interpolate_govt_yield(y_note, y_bond, tenor_y) + float(spread)
