"""Published macro aggregates (real and nominal)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Aggregates:
    """One-month snapshot. Real quantities at current-month prices unless noted."""

    gdp_prod_real: float
    gdp_exp_real: float
    gdp_prod_nom: float
    gdp_exp_nom: float
    cpi: float
    core_cpi: float
    u: float
    utilisation: float
    x: np.ndarray
    govt_balance: float
    debt_gdp: float
    trade_balance: float
    saving_rate: float
    leverage: float
    pi12: float

    def to_state(self) -> dict[str, Any]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def production_gdp(x: np.ndarray, leak: np.ndarray, inv_prev: np.ndarray, mu: np.ndarray, m: np.ndarray) -> float:
    """Real production-side GDP, net of spoilage: ``Σ(x − λ inv_prev − (μ+m)x)``."""
    return float((x - leak * inv_prev - (mu + m) * x).sum())


def expenditure_gdp(
    c: np.ndarray,
    g: np.ndarray,
    i_fixed: float,
    ex: np.ndarray,
    m_real: np.ndarray,
    d_inv: np.ndarray,
) -> float:
    """C + I_fixed + G + X − M + Δ finished-goods inventories."""
    return float(c.sum() + g.sum() + i_fixed + ex.sum() - m_real.sum() + d_inv.sum())


@dataclass
class Published:
    """Lagged published series. ``published(name, lag)`` is the value ``lag`` months ago."""

    hist: dict[str, list[float]] = field(default_factory=dict)

    def push(self, **series: float) -> None:
        for name, value in series.items():
            self.hist.setdefault(name, []).append(float(value))

    def published(self, series: str, lag: int = 0) -> float:
        h = self.hist[series]
        idx = -1 - lag
        if idx < -len(h):
            raise KeyError(f"{series} has no vintage at lag {lag}")
        return h[idx]

    def to_state(self) -> dict[str, Any]:
        return {k: list(v) for k, v in self.hist.items()}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Published:
        p = cls()
        p.hist = {k: [float(x) for x in v] for k, v in state.items()}
        return p
