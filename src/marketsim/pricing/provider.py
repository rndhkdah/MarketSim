"""Gordon-style AssetPriceProvider stub (§2.10). Household equity wealth effect is off."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.config import Config
from marketsim.real.steady_state import FinancialBaseline, RealBaseline

RHO_RATE_WEIGHT = 0.35  # §2.10: Δρ = 0.35·(r − π_e − r_n) + z_risk


@dataclass
class StubAssetPriceProvider:
    """``ln V = ln E^e + ln PE0 − cf_duration·Δρ`` with ``Q = V / (v K) = 1`` at baseline."""

    ee: np.ndarray
    pe0: np.ndarray
    v_k: np.ndarray
    duration: np.ndarray
    r_n: float

    @classmethod
    def from_baseline(cls, real: RealBaseline, fin: FinancialBaseline, cfg: Config) -> StubAssetPriceProvider:
        ee = 12.0 * np.maximum(fin.ebitda0 - fin.int0 - fin.tax0, 1e-12)
        v_k = real.flat(real.v) * real.flat(real.K0)
        pe0 = v_k / ee
        duration = np.array([cfg.sectors.params(c).cf_duration for c in real.codes], dtype=float)
        return cls(ee=ee, pe0=pe0, v_k=v_k, duration=duration, r_n=cfg.edges.policy.taylor.r_neutral)

    def delta_rho(self, r: float, pi_e: float, z_risk: float = 0.0) -> float:
        return RHO_RATE_WEIGHT * (r - pi_e - self.r_n) + z_risk

    def values(self, r: float, pi_e: float, z_risk: float = 0.0) -> np.ndarray:
        """Fundamental values ``V_j`` (cr)."""
        drho = self.delta_rho(r, pi_e, z_risk)
        return np.exp(np.log(self.ee) + np.log(self.pe0) - self.duration * drho)

    def q_tobin(self, r: float, pi_e: float, z_risk: float = 0.0) -> np.ndarray:
        """Tobin's Q. Dimensionless; 1 at baseline."""
        return self.values(r, pi_e, z_risk) / self.v_k

    def v_re(self, r: float, pi_e: float, z_risk: float = 0.0, codes: tuple[str, ...] | None = None) -> float:
        """REALESTATE value used as the collateral index."""
        v = self.values(r, pi_e, z_risk)
        if codes is None:
            return float(v[-1])
        return float(v[list(codes).index("REALESTATE")])
