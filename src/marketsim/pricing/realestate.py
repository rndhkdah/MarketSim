"""Regional property index tied to REALESTATE value and collateral (T8.09).

This is the asset-price meeting point for REALESTATE collateral (two-graphs
rule): the real layer supplies ``V_RE`` (cr); optional log-mispricing ``ξ_r``
is the financial-layer residual. No hand-set sector beta.

Quoted regional index (same scale as ``V_RE``)::

    P_r = V_RE · exp(ξ_r)     (ξ omitted ⇒ P_r = V_RE)

National traded index (``w_r`` = loc_weight, default ``population_share``)::

    P_national = Σ_r w_r · P_r

Default weights sum to 1, so ``ξ = 0`` ⇒ ``P_national = V_RE`` and collateral
that reads the traded national price matches the existing ``V_RE`` channel.
Symbols are ``RE:<REGION>`` (e.g. ``RE:CAPITAL``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from marketsim.real.credit import LAM_SNAP, collateral_index

RE_PREFIX = "RE:"
DEFAULT_VENUE = "engine_mm"
# config/regions.yaml order and population_share (CAPITAL, INDUSTRIAL, RESOURCE).
DEFAULT_REGIONS: tuple[str, ...] = ("CAPITAL", "INDUSTRIAL", "RESOURCE")
DEFAULT_WEIGHTS: tuple[float, ...] = (0.45, 0.35, 0.20)


def re_symbol(region: str, prefix: str = RE_PREFIX) -> str:
    """Tradable regional property symbol ``RE:<REGION>``. ``region`` is a code."""
    return f"{prefix}{region}"


def symbols(regions: Sequence[str] = DEFAULT_REGIONS, *, prefix: str = RE_PREFIX) -> tuple[str, ...]:
    """``RE:<REGION>`` names in ``regions`` order."""
    return tuple(re_symbol(r, prefix) for r in regions)


def loc_weights(
    weights: Sequence[float] | Mapping[str, float] | None = None,
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
    prefix: str = RE_PREFIX,
) -> np.ndarray:
    """Location weights summing to 1 (dimensionless), aligned with ``regions``.

    ``weights`` is a sequence in region order or a map keyed by region code or
    ``RE:<REGION>``. ``None`` uses shipped ``population_share`` when ``regions``
    is the default triple; otherwise equal weights.
    """
    n = len(regions)
    if n == 0:
        raise ValueError("regions must be non-empty")
    if weights is None:
        if tuple(regions) == DEFAULT_REGIONS:
            w = np.asarray(DEFAULT_WEIGHTS, dtype=float)
        else:
            w = np.full(n, 1.0 / n)
    elif isinstance(weights, Mapping):
        w = np.empty(n, dtype=float)
        for i, region in enumerate(regions):
            if region in weights:
                w[i] = float(weights[region])
            else:
                w[i] = float(weights[re_symbol(region, prefix)])
    else:
        w = np.asarray(list(weights), dtype=float)
        if w.size != n:
            raise ValueError(f"weights length {w.size} != n_regions {n}")
    if np.any(w < 0.0):
        raise ValueError("loc weights must be >= 0")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("loc weights must sum to a positive total")
    return w / total


def regional_prices(
    v_re: float,
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
    weights: Sequence[float] | Mapping[str, float] | None = None,
    xi: Mapping[str, float] | Sequence[float] | None = None,
    prefix: str = RE_PREFIX,
) -> dict[str, float]:
    """Regional quoted prices ``P_r = V_RE · exp(ξ_r)`` (index, cr-scale of ``V_RE``).

    ``v_re`` is cr. ``xi`` is dimensionless log-mispricing, keyed by symbol or
    region, or a sequence in region order. Omitted ``ξ`` (or ``ξ = 0``) ⇒
    ``P_r = V_RE``. ``weights`` are unused here: they aggregate the national
    index, they do not rescale the quoted regional index (so ``ξ = 0`` keeps
    every ``RE:<REGION>`` print equal to ``V_RE`` and collateral-consistent).
    """
    del weights
    regs = tuple(regions)
    n = len(regs)
    if xi is None:
        x = np.zeros(n, dtype=float)
    elif isinstance(xi, Mapping):
        x = np.array(
            [float(xi.get(re_symbol(r, prefix), xi.get(str(r), 0.0))) for r in regs],
            dtype=float,
        )
    else:
        x = np.asarray(list(xi), dtype=float)
        if x.size != n:
            raise ValueError(f"xi length {x.size} != n_regions {n}")
    prices = float(v_re) * np.exp(x)
    return {re_symbol(r, prefix): float(p) for r, p in zip(regs, prices, strict=True)}


def national_index(
    prices: Mapping[str, float],
    weights: Sequence[float] | Mapping[str, float] | None = None,
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
    prefix: str = RE_PREFIX,
) -> float:
    """National traded index ``Σ_r w_r · P_r`` (index, same units as ``P_r``).

    ``prices`` is keyed by ``RE:<REGION>`` or region code. ``weights`` sum to 1.
    """
    w = loc_weights(weights, regions=regions, prefix=prefix)
    ordered = np.array(
        [
            float(prices[re_symbol(r, prefix)] if re_symbol(r, prefix) in prices else prices[r])
            for r in regions
        ],
        dtype=float,
    )
    return float(np.dot(ordered, w))


def traded_national(
    v_re: float,
    *,
    regions: Sequence[str] = DEFAULT_REGIONS,
    weights: Sequence[float] | Mapping[str, float] | None = None,
    xi: Mapping[str, float] | Sequence[float] | None = None,
    prefix: str = RE_PREFIX,
) -> float:
    """National traded index from ``V_RE`` and ``ξ``. ``ξ = 0`` ⇒ equals ``v_re`` (cr)."""
    prices = regional_prices(v_re, regions=regions, xi=xi, prefix=prefix)
    return national_index(prices, weights, regions=regions, prefix=prefix)


def collateral_from_price(p_traded: float, p_trend: float) -> float:
    """``Λ_coll`` from the traded national index vs its trend. Dimensionless.

    Same ``ln(P / P_trend)`` → ``collateral_index`` path as ``CreditBlock``
    (imported from ``marketsim.real.credit``, not a second formula). Instantaneous;
    Erlang / LTV / event multipliers stay on the credit block.
    """
    rel = max(float(p_traded), 1e-12) / max(float(p_trend), 1e-12)
    gap = 0.0 if abs(rel - 1.0) < LAM_SNAP else float(np.log(rel))
    return float(collateral_index(gap))


@dataclass(frozen=True)
class RealEstateSpec:
    """Loaded ``realestate:`` block. ``weights`` sum to 1 and align with ``regions``."""

    prefix: str = RE_PREFIX
    venue: str = DEFAULT_VENUE
    regions: tuple[str, ...] = DEFAULT_REGIONS
    weights: tuple[float, ...] = DEFAULT_WEIGHTS

    def to_state(self) -> dict[str, Any]:
        """Serialise prefix/venue/regions/weights. No kernel state."""
        return {
            "prefix": self.prefix,
            "venue": self.venue,
            "regions": list(self.regions),
            "weights": list(self.weights),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> RealEstateSpec:
        regions = tuple(str(r) for r in state["regions"])
        raw_w = state["weights"]
        w = loc_weights(tuple(float(x) for x in raw_w), regions=regions)
        return cls(
            prefix=str(state["prefix"]),
            venue=str(state["venue"]),
            regions=regions,
            weights=tuple(float(x) for x in w),
        )


def load_realestate_spec(
    markets_path: str | Path | None = None,
    *,
    weights: Sequence[float] | Mapping[str, float] | None = None,
) -> RealEstateSpec:
    """Read ``realestate:`` from ``markets.yaml`` (yaml only; not ``config.py``).

    Explicit ``weights`` override the file. Missing block → shipped defaults.
    ``markets_path`` defaults to ``config/markets.yaml`` next to the package.
    """
    path = (
        Path(__file__).resolve().parents[3] / "config" / "markets.yaml"
        if markets_path is None
        else Path(markets_path)
    )
    loaded = yaml.safe_load(path.read_text()) if path.is_file() else {}
    raw = loaded if isinstance(loaded, dict) else {}
    block = raw.get("realestate") if isinstance(raw.get("realestate"), dict) else {}
    prefix = str(block.get("prefix", RE_PREFIX))
    venue = str(block.get("venue", DEFAULT_VENUE))
    regions = tuple(str(r) for r in block.get("regions", DEFAULT_REGIONS))
    file_w = block.get("weights")
    if weights is not None:
        w = loc_weights(weights, regions=regions, prefix=prefix)
    elif file_w is not None:
        w = loc_weights(tuple(float(x) for x in file_w), regions=regions, prefix=prefix)
    else:
        w = loc_weights(None, regions=regions, prefix=prefix)
    return RealEstateSpec(
        prefix=prefix,
        venue=venue,
        regions=regions,
        weights=tuple(float(x) for x in w),
    )
