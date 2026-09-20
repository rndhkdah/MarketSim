"""Cash-settled commodity prices: national sector reference + carry (T6.18 / §6.2–§6.3).

``OIL``, ``METALS``, ``GRAINS`` settle on the national reference price of
ENERGY, MATERIALS, AGRIFOOD. Carry is storage net of convenience (§6.3).
The price *level* is never capped (AGENTS rule 7): a 3–5× cost-push on the
reference is transmitted one-for-one. The real layer already bounds the
*step* of the goods price (``dynamics.prices.step_max_month``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from marketsim.layer1.build_io import CODES
from marketsim.market.instruments import COMMODITY_SECTORS

# Simulated calendar: 21 days / month × 12 months (core.calendar).
DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
# One market tick and one real-economy month, as years (for ``carry * dt``).
DT_DAY = 1.0 / DAYS_PER_YEAR
DT_MONTH = 1.0 / MONTHS_PER_YEAR
# §6.3 net storage / convenience carry (annual decimal). Default 0 so the
# no-shock settlement price equals the reference (stationary at π*). Not a
# markets.yaml key on this card; pass explicitly to apply a non-zero carry.
CARRY = 0.0

COMMODITY_SYMBOLS: tuple[str, ...] = tuple(COMMODITY_SECTORS)


def carry_return(carry: float, dt: float) -> float:
    """One-period carry as a decimal return.

    ``carry`` is an annual decimal (storage / convenience, §6.3). ``dt`` is
    years per period (``DT_DAY`` = 1/252, ``DT_MONTH`` = 1/12).
    """
    return float(carry) * float(dt)


def national_reference_price(
    prices: Mapping[str, float] | np.ndarray | float,
    sector: str,
    *,
    codes: Sequence[str] | None = None,
    weights: np.ndarray | None = None,
) -> float:
    """National cell reference price for ``sector`` (price index).

    ``prices`` is a scalar, a sector→price map, ``(S,)``, or ``(R, S)``.
    Regional rows are sales-weighted when ``weights`` is ``(R,)`` or
    ``(R, S)``; otherwise equal-weighted. Units: price index.
    """
    if isinstance(prices, Mapping):
        return float(prices[sector])
    arr = np.asarray(prices, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    names = tuple(CODES if codes is None else codes)
    j = names.index(sector)
    if arr.ndim == 1:
        return float(arr[j])
    if arr.ndim != 2:
        raise ValueError(f"prices must be scalar, (S,), or (R, S); got {arr.shape}")
    col = arr[:, j]
    if weights is None:
        return float(col.mean())
    w = np.asarray(weights, dtype=float)
    if w.ndim == 2:
        w = w[:, j]
    wsum = float(w.sum())
    if wsum <= 0.0:
        return float(col.mean())
    return float((col * w).sum() / wsum)


def commodity_price(
    p_ref: float,
    *,
    carry: float = CARRY,
    dt: float = DT_DAY,
) -> float:
    """Cash-settlement price ``p_ref · (1 + carry · dt)`` (price index).

    ``p_ref`` is the national sector reference (price index). ``carry`` is
    annual decimal (§6.3). ``dt`` is years. No level cap: a 3–5× ``p_ref``
    is returned as a 3–5× settlement (times the carry factor).
    """
    return float(p_ref) * (1.0 + carry_return(carry, dt))


def commodity_prices(
    prices: Mapping[str, float] | np.ndarray,
    *,
    carry: float = CARRY,
    dt: float = DT_DAY,
    codes: Sequence[str] | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Settlement prices for ``OIL``, ``METALS``, ``GRAINS`` (price index).

    ``prices`` is a sector→price map or a ``(S,)`` / ``(R, S)`` array in
    ``codes`` order (default ``CODES``). Carry is the same on every name.
    """
    names = tuple(CODES if codes is None else codes)
    out: dict[str, float] = {}
    for symbol, sector in COMMODITY_SECTORS.items():
        ref = national_reference_price(prices, sector, codes=names, weights=weights)
        out[symbol] = commodity_price(ref, carry=carry, dt=dt)
    return out
