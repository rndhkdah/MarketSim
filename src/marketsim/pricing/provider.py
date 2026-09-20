"""Asset-price provider: structural fundamentals behind the Phase-2 interface (T6.04 / §6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from marketsim.core.config import Config
from marketsim.pricing.curve import y10
from marketsim.pricing.discount import delta_rho as curve_delta_rho
from marketsim.pricing.fundamentals import financials_dlnv, fundamental_values
from marketsim.real.steady_state import FinancialBaseline, RealBaseline

PricingMode = Literal["structural", "factor_lite"]
RHO_RATE_WEIGHT = 0.35  # §2.10 DISC_PASS; economy `values(r,π_e,z)` stays on this


@dataclass
class AssetPriceProvider:
    """``ln V = ln E^e + ln PE0 − D·Δρ + D·Δg_lr`` plus optional financials add-on.

    ``Q = V / (v K) = 1`` at baseline. The Phase-2 call ``values(r, π_e, z_risk)``
    keeps the stub ``Δρ = 0.35·(r − π_e − r_n) + z_risk`` so credit / capex
    IRFs stay put. The curve-based ``Δρ`` and financials override are used when
    ``ee`` / ``apply_financials`` are passed, or in ``factor_lite``.
    """

    ee: np.ndarray
    pe0: np.ndarray
    v_k: np.ndarray
    duration: np.ndarray
    r_n: float
    codes: tuple[str, ...]
    mode: PricingMode = "structural"
    beta_rate: np.ndarray | None = None
    cfg: Config | None = None
    r0: float | None = None

    @classmethod
    def from_baseline(cls, real: RealBaseline, fin: FinancialBaseline, cfg: Config) -> AssetPriceProvider:
        ee = 12.0 * np.maximum(fin.ebitda0 - fin.int0 - fin.tax0, 1e-12)
        v_k = real.flat(real.v) * real.flat(real.K0)
        pe0 = v_k / ee
        duration = np.array([cfg.sectors.params(c).cf_duration for c in real.codes], dtype=float)
        mode: PricingMode = "structural"
        if cfg.markets is not None:
            mode = cfg.markets.pricing.mode
        beta_rate = None
        if mode == "factor_lite":
            from marketsim.layer1.betas import derive_betas
            from marketsim.layer1.io import load_io, resolve_io_path

            betas = derive_betas(load_io(resolve_io_path(cfg)), cfg)
            beta_rate = np.asarray(betas.beta_rate, dtype=float)
        return cls(
            ee=ee,
            pe0=pe0,
            v_k=v_k,
            duration=duration,
            r_n=cfg.edges.policy.taylor.r_neutral,
            codes=tuple(real.codes),
            mode=mode,
            beta_rate=beta_rate,
            cfg=cfg,
            r0=float(fin.r0),
        )

    def delta_rho(self, r: float, pi_e: float, z_risk: float = 0.0) -> float:
        """Phase-2 ``Δρ`` (annual decimal). Exact stub: ``0.35·(r − π_e − r_n) + z_risk``."""
        return RHO_RATE_WEIGHT * (r - pi_e - self.r_n) + z_risk

    def values(
        self,
        r: float,
        pi_e: float,
        z_risk: float = 0.0,
        *,
        ee: np.ndarray | None = None,
        delta_g_lr: float | np.ndarray = 0.0,
        sentiment: float = 0.0,
        apply_financials: bool = False,
    ) -> np.ndarray:
        """Fundamental values ``V_j`` (cr)."""
        ee_use = self.ee if ee is None else np.asarray(ee, dtype=float)
        r_base = self.r_n + pi_e if self.r0 is None else float(self.r0)
        if self.mode == "factor_lite" and self.beta_rate is not None:
            dln = self.beta_rate * (r - r_base) - z_risk * self.duration
            return self.v_k * np.exp(dln)
        if ee is None and not apply_financials and np.all(np.asarray(delta_g_lr) == 0.0) and sentiment == 0.0:
            drho = self.delta_rho(r, pi_e, z_risk)
            return fundamental_values(ee_use, self.pe0, self.duration, drho, 0.0)
        drho = curve_delta_rho(r, self.r_n, pi_e, z_risk=z_risk, sentiment=sentiment)
        v = fundamental_values(ee_use, self.pe0, self.duration, drho, delta_g_lr)
        if apply_financials and self.cfg is not None:
            y_t = y10(r, self.r_n, pi_e)
            y_0 = y10(r_base, self.r_n, pi_e)
            v = v * np.exp(financials_dlnv(self.codes, self.cfg, r=r, r0=r_base, y10_t=y_t, y10_0=y_0))
        return v

    def q_tobin(self, r: float, pi_e: float, z_risk: float = 0.0) -> np.ndarray:
        """Tobin's Q. Dimensionless; 1 at baseline."""
        return self.values(r, pi_e, z_risk) / self.v_k

    def v_re(self, r: float, pi_e: float, z_risk: float = 0.0, codes: tuple[str, ...] | None = None) -> float:
        """REALESTATE value used as the collateral index."""
        v = self.values(r, pi_e, z_risk)
        names = self.codes if codes is None else codes
        return float(v[list(names).index("REALESTATE")])


# T2.21 name. Default ``values(r, π_e, z)`` is the stub formula.
StubAssetPriceProvider = AssetPriceProvider
