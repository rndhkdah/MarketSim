"""Want layer and within-want allocation (§3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from marketsim.layer1.build_io import CODES


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WantSpec(FrozenModel):
    shape: str
    sectors: dict[str, float]


class WantsConfig(FrozenModel):
    sigma_default: float = 0.8
    sigma: dict[str, float] = Field(default_factory=dict)
    min_share: float = 0.01
    max_share: float = 0.95
    availability_kappa: float = 1.0
    wants: dict[str, WantSpec]


@dataclass
class WantLayer:
    """Dense want → sector maps. ``M`` is ``(Q, S)`` prior/RAS weights (rows sum to 1)."""

    names: tuple[str, ...]
    shapes: tuple[str, ...]
    m: np.ndarray  # (Q, S)
    sigma: np.ndarray  # (Q,)
    min_share: float
    max_share: float
    kappa: float
    shift: np.ndarray  # (Q,) multiplicative; 1 at rest

    @property
    def n_wants(self) -> int:
        return len(self.names)


def load_wants(raw: dict, codes: tuple[str, ...] = CODES) -> WantLayer:
    """Build a ``WantLayer`` from ``wants.yaml`` (or a calibrator overlay)."""
    cfg = WantsConfig.model_validate(raw)
    names = tuple(cfg.wants.keys())
    shapes = tuple(cfg.wants[n].shape for n in names)
    s_idx = {c: i for i, c in enumerate(codes)}
    m = np.zeros((len(names), len(codes)))
    for qi, name in enumerate(names):
        for code, w in cfg.wants[name].sectors.items():
            if code not in s_idx:
                raise ValueError(f"want {name} unknown sector {code}")
            m[qi, s_idx[code]] = float(w)
        tot = float(m[qi].sum())
        if tot <= 0:
            raise ValueError(f"want {name} has empty membership")
        m[qi] /= tot
    sigma = np.array([cfg.sigma.get(n, cfg.sigma_default) for n in names], dtype=float)
    return WantLayer(
        names=names,
        shapes=shapes,
        m=m,
        sigma=sigma,
        min_share=cfg.min_share,
        max_share=cfg.max_share,
        kappa=cfg.availability_kappa,
        shift=np.ones(len(names)),
    )


def allocate_within_want(
    layer: WantLayer,
    prices: np.ndarray,
    avail: np.ndarray,
) -> np.ndarray:
    """``share[q,i] ∝ M[q,i]·(p_i/P_q)^{−σ_q}·avail_i^κ``, clipped and renormalised.

    Returns ``(Q, S)``. Rows of active wants (row-sum of M > 0) sum to 1.
    """
    p = np.asarray(prices, dtype=float)
    av = np.maximum(np.asarray(avail, dtype=float), 1e-12)
    m = layer.m
    p_q = np.maximum((m * p[None, :]).sum(axis=1), 1e-12)
    rel = p[None, :] / p_q[:, None]
    raw = m * np.power(rel, -layer.sigma[:, None]) * np.power(av[None, :], layer.kappa)
    raw = np.where(m > 0, raw, 0.0)
    raw = np.clip(raw, 0.0, None)
    # clip positive members into [min, max] then renormalise
    members = m > 0
    raw = np.where(members, np.clip(raw, layer.min_share, layer.max_share), 0.0)
    denom = raw.sum(axis=1, keepdims=True)
    return np.divide(raw, denom, out=np.zeros_like(raw), where=denom > 0)


def want_price(layer: WantLayer, shares: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Spending-weighted want price ``P_q`` (index)."""
    return np.maximum((shares * np.asarray(prices, dtype=float)[None, :]).sum(axis=1), 1e-12)
