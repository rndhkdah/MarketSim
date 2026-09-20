"""Bond-index holding return (§6.3: ``−D · Δy + carry``)."""

from __future__ import annotations

# config/sectors.yaml bond_index.duration — published bond-index / GB_BOND duration (years).
BOND_INDEX_DURATION = 7.0


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
