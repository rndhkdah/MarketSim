"""Demand-system protocol and the scalar-η allocation (R2)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DemandSystem(Protocol):
    def allocate(
        self,
        budget: float,
        prices: np.ndarray,
        income_index: float,
        rate_gap: float,
        shifters: np.ndarray | None = None,
    ) -> np.ndarray:
        """Real quantities (cr/month at current prices) summing in value to the budget share rule."""
        ...


class ScalarEtaDemand:
    """``z_i = θ_i y^{η_i−1} (p_i/P_c)^{1+ε_i} exp(semi_i/100 · rate_gap) · shifter_i``."""

    def __init__(
        self,
        theta: np.ndarray,
        eta: np.ndarray,
        eps: np.ndarray,
        dem_rate_semi: np.ndarray,
        zeta: float,
        vat: float = 0.0,
        excise: np.ndarray | None = None,
    ) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.eta = np.asarray(eta, dtype=float)
        self.eps = np.asarray(eps, dtype=float)
        self.dem_rate_semi = np.asarray(dem_rate_semi, dtype=float)
        self.zeta = float(zeta)
        self.vat = float(vat)
        self.excise = np.zeros_like(self.theta) if excise is None else np.asarray(excise, dtype=float)

    def allocate(
        self,
        budget: float,
        prices: np.ndarray,
        income_index: float,
        rate_gap: float,
        shifters: np.ndarray | None = None,
    ) -> np.ndarray:
        p = np.asarray(prices, dtype=float)
        pc = float((self.theta * p).sum())
        y = float(income_index)
        shift = np.ones_like(p) if shifters is None else np.asarray(shifters, dtype=float)
        z = (
            self.theta
            * np.power(y, self.eta - 1.0)
            * np.power(p / pc, 1.0 + self.eps)
            * np.exp(self.dem_rate_semi / 100.0 * rate_gap)
            * shift
        )
        z = np.maximum(z, 0.0)
        zsum = float(z.sum())
        if zsum <= 0:
            return np.zeros_like(p)
        hh_shift = zsum / float(self.theta.sum())
        wedge = 1.0 + self.vat + self.excise
        return budget * (hh_shift**self.zeta) * (z / zsum) / (p * wedge)
