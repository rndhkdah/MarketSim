"""Demand-system protocol, scalar-η allocation (R2), and tiers × wants (T3.12)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import yaml

from marketsim.core.config import Config
from marketsim.demand.packages import PackageSet, load_packages, want_shares
from marketsim.demand.tiers import Tiers, build_tiers
from marketsim.demand.wants import WantLayer, WantShifts, allocate_within_want, load_wants, want_price
from marketsim.layer1.build_io import CODES


class DemandSystem(Protocol):
    def allocate(
        self,
        budget: float,
        prices: np.ndarray,
        income_index: float,
        rate_gap: float,
        shifters: np.ndarray | None = None,
        avail: np.ndarray | None = None,
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
        avail: np.ndarray | None = None,
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
        _ = avail  # scalar-η has no availability channel
        return budget * (hh_shift**self.zeta) * (z / zsum) / (p * wedge)


class TiersWantsDemand:
    """Wealth-tier packages over the want layer. Macro C is unchanged; this is composition."""

    def __init__(
        self,
        layer: WantLayer,
        packages: PackageSet,
        tiers: Tiers,
        theta: np.ndarray,
        eps: np.ndarray,
        dem_rate_semi: np.ndarray,
        vat: float = 0.0,
        excise: np.ndarray | None = None,
    ) -> None:
        self.layer = layer
        self.packages = packages
        self.tiers = tiers
        self.theta = np.asarray(theta, dtype=float)
        self.eps = np.asarray(eps, dtype=float)
        self.dem_rate_semi = np.asarray(dem_rate_semi, dtype=float)
        self.vat = float(vat)
        self.excise = np.zeros_like(self.theta) if excise is None else np.asarray(excise, dtype=float)
        self.shifts: WantShifts | None = None

    def _want_shift(self) -> np.ndarray:
        if self.shifts is not None:
            return self.shifts.national()
        return self.layer.shift

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        theta: np.ndarray,
        eps: np.ndarray,
        dem_rate_semi: np.ndarray,
        vat: float = 0.0,
        excise: np.ndarray | None = None,
        codes: tuple[str, ...] = CODES,
    ) -> TiersWantsDemand:
        """Load RAS-fitted ``wants.yaml`` / ``buy_packages.yaml`` from ``cfg.config_dir``."""
        root = cfg.config_dir
        wants_raw = yaml.safe_load((root / "wants.yaml").read_text(encoding="utf-8"))
        pkgs_raw = yaml.safe_load((root / "buy_packages.yaml").read_text(encoding="utf-8"))
        layer = load_wants(wants_raw, codes)
        packages = load_packages(pkgs_raw, layer.names)
        tiers = build_tiers(float(pkgs_raw.get("sigma", 0.75)))
        return cls(
            layer=layer,
            packages=packages,
            tiers=tiers,
            theta=theta,
            eps=eps,
            dem_rate_semi=dem_rate_semi,
            vat=vat,
            excise=excise,
        )

    def want_index(self, name: str) -> int:
        return self.layer.names.index(name)

    def sector_shares(
        self,
        income_index: float,
        prices: np.ndarray,
        rate_gap: float = 0.0,
        shifters: np.ndarray | None = None,
        avail: np.ndarray | None = None,
    ) -> np.ndarray:
        """Nominal sector spend shares (simplex) at real income ``income_index``."""
        p = np.asarray(prices, dtype=float)
        y = float(income_index)
        av = np.ones_like(p) if avail is None else np.maximum(np.asarray(avail, dtype=float), 1e-12)
        shift = np.ones_like(p) if shifters is None else np.asarray(shifters, dtype=float)
        v = want_shares(y, self.packages, self.tiers) * self._want_shift()
        v = v / max(float(v.sum()), 1e-12)
        within = allocate_within_want(self.layer, p, av, clip_bounds=False)
        p_q = want_price(self.layer, within, p)
        w_m = self.layer.m / np.maximum(self.layer.m.sum(axis=1, keepdims=True), 1e-12)
        eps_q = w_m @ self.eps
        pc = max(float(v @ p_q), 1e-12)
        v = v * np.power(p_q / pc, 1.0 + eps_q)
        v = v / max(float(v.sum()), 1e-12)
        s = np.maximum(v @ within, 0.0)
        s = s * np.exp(self.dem_rate_semi / 100.0 * rate_gap) * shift
        tot = float(s.sum())
        if tot <= 0:
            return np.zeros_like(p)
        return s / tot

    def allocate(
        self,
        budget: float,
        prices: np.ndarray,
        income_index: float,
        rate_gap: float,
        shifters: np.ndarray | None = None,
        avail: np.ndarray | None = None,
    ) -> np.ndarray:
        """Real quantities (cr/month). Nominal spend is ``budget`` after the tax wedge."""
        p = np.asarray(prices, dtype=float)
        s = self.sector_shares(income_index, p, rate_gap, shifters, avail)
        wedge = 1.0 + self.vat + self.excise
        return float(budget) * s / (p * wedge)
