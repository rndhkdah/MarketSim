"""Regional property index tied to REALESTATE value (T8.09).

``P_r = V_RE · exp(ξ_r)``. The national index is the population-share average
of ``P_r``, so ``ξ = 0`` ⇒ national = ``V_RE`` and collateral matches
``credit.collateral_index``. Symbols are ``RE:<REGION>``. ``V_RE`` is cr.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from marketsim.real.credit import collateral_index

RE_PREFIX = "RE:"
# Shipped regions.yaml order (T3.01). Weights = population_share.
DEFAULT_REGIONS: tuple[str, ...] = ("CAPITAL", "INDUSTRIAL", "RESOURCE")
DEFAULT_WEIGHTS: tuple[float, ...] = (0.45, 0.35, 0.20)


def re_symbol(region: str) -> str:
    """Tradable regional property symbol. ``region`` is a unitless code."""
    return f"{RE_PREFIX}{region}"


def symbols(regions: Sequence[str] = DEFAULT_REGIONS) -> tuple[str, ...]:
    """``RE:<REGION>`` names in ``regions`` order."""
    return tuple(re_symbol(r) for r in regions)


def _weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    if weights is None:
        w = np.asarray(DEFAULT_WEIGHTS[:n], dtype=float)
        if w.size < n:
            w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(list(weights), dtype=float)
    tot = float(w.sum())
    if tot <= 0.0:
        return np.full(n, 1.0 / max(n, 1))
    return w / tot


def regional_prices(
    v_re: float,
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
    weights: Sequence[float] | None = None,
    xi: Mapping[str, float] | Sequence[float] | None = None,
) -> dict[str, float]:
    """``P_r = V_RE · exp(ξ_r)`` (index). ``v_re`` is cr; ``xi`` is dimensionless.

    National index is the ``weights`` average of ``P_r``. ``ξ = 0`` ⇒ every
    region prints ``V_RE`` and the national index equals ``V_RE``.
    """
    del weights  # weights enter national_index / traded_national, not the level split
    if xi is None:
        x = np.zeros(len(regions))
    elif isinstance(xi, Mapping):
        x = np.array([float(xi.get(re_symbol(r), xi.get(str(r), 0.0))) for r in regions], dtype=float)
    else:
        x = np.asarray(list(xi), dtype=float)
    prices = float(v_re) * np.exp(x)
    return {re_symbol(r): float(p) for r, p in zip(regions, prices, strict=True)}


def national_index(prices: Mapping[str, float], weights: Sequence[float] | None = None) -> float:
    """Cap-weighted national property index (same units as ``P_r``)."""
    keys = tuple(sorted(prices))
    vals = np.array([float(prices[k]) for k in keys], dtype=float)
    w = _weights(weights, len(keys))
    # Caller usually passes unsorted dict; use insertion order if it matches regions.
    if set(prices) == set(symbols()):
        ordered = [prices[re_symbol(r)] for r in DEFAULT_REGIONS]
        w = _weights(weights, len(DEFAULT_REGIONS))
        return float(np.dot(np.asarray(ordered, dtype=float), w))
    return float(np.dot(vals, w))


def traded_national(
    v_re: float,
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
    weights: Sequence[float] | None = None,
    xi: Mapping[str, float] | Sequence[float] | None = None,
) -> float:
    """National traded index. ``ξ = 0`` ⇒ equals ``v_re`` (cr)."""
    prices = regional_prices(v_re, regions=regions, weights=weights, xi=xi)
    w = _weights(weights, len(regions))
    ordered = [prices[re_symbol(r)] for r in regions]
    return float(np.dot(np.asarray(ordered, dtype=float), w))


def collateral_from_price(p_traded: float, p_trend: float) -> float:
    """``Λ_coll`` from traded vs trend index. Same gate as ``credit.collateral_index``."""
    rel = max(float(p_traded), 1e-12) / max(float(p_trend), 1e-12)
    gap = float(np.log(rel))
    return float(collateral_index(gap))
