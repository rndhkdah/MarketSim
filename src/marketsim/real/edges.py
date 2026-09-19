"""Typed substitution / complement edges on final demand (§2.8)."""

from __future__ import annotations

from typing import Any

import numpy as np

from marketsim.core.config import Config, TypedEdge
from marketsim.core.erlang import ErlangSmoother

DEFAULT_SATURATION = 0.3


def _mean_m(edge: TypedEdge) -> float:
    return 3.0 * float(edge.lag_q)


def _saturation(edge: TypedEdge) -> float:
    return DEFAULT_SATURATION if edge.saturation is None else float(edge.saturation)


class TypedEdgeBlock:
    """``shifter_dst = exp(Σ clip(elasticity · Smooth[ln(p_src/Pc)], ±sat))``."""

    def __init__(self, cfg: Config, codes: tuple[str, ...]) -> None:
        self.codes = codes
        self.idx = {c: i for i, c in enumerate(codes)}
        self.edges: list[TypedEdge] = list(cfg.edges.substitution) + list(cfg.edges.complement)
        self.smooth = [ErlangSmoother(e.erlang_k, _mean_m(e), 0.0) for e in self.edges]

    def shifters(self, prices: np.ndarray, pc: float) -> np.ndarray:
        """Per-sector FD shifters (dimensionless). Intermediate orders are never scaled."""
        p = np.asarray(prices, dtype=float)
        log_shift = np.zeros(len(self.codes))
        pc_safe = max(float(pc), 1e-12)
        for e, sm in zip(self.edges, self.smooth, strict=True):
            if e.src not in self.idx or e.dst not in self.idx:
                continue
            ln_rel = float(np.log(max(p[self.idx[e.src]], 1e-12) / pc_safe))
            smoothed = float(sm.push(ln_rel))
            sat = _saturation(e)
            log_shift[self.idx[e.dst]] += float(np.clip(e.elasticity * smoothed, -sat, sat))
        return np.exp(log_shift)

    def to_state(self) -> dict[str, Any]:
        return {"smooth": [s.to_state() for s in self.smooth]}

    def from_state(self, state: dict[str, Any]) -> None:
        self.smooth = [ErlangSmoother.from_state(s) for s in state["smooth"]]
