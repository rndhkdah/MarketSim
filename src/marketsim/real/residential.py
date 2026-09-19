"""Household residential investment, routed to CONSTRUCT (R3)."""

from __future__ import annotations

import numpy as np

from marketsim.core.config import Config
from marketsim.core.erlang import ErlangSmoother
from marketsim.real.steady_state import RealBaseline


def residential_desired(
    res0: float,
    y: float,
    rate_gap: float,
    income_el: float,
    rate_semi: float,
    z_dem: float = 0.0,
) -> float:
    """Desired real residential starts (cr/month) before the Erlang lag."""
    return float(res0 * (y**income_el) * np.exp(rate_semi / 100.0 * rate_gap) * np.exp(z_dem))


class ResidentialBlock:
    """Erlang(2, 3m) smoother on residential starts. Units: cr/month real."""

    def __init__(self, cfg: Config, real: RealBaseline) -> None:
        assert cfg.dynamics is not None
        lag = cfg.dynamics.residential.lag
        self.res0 = float(real.res0)
        self.income_el = cfg.dynamics.residential.income_elasticity
        self.rate_semi = cfg.dynamics.residential.rate_semi
        self.share = cfg.dynamics.residential.share_of_investment
        self.smoother = ErlangSmoother(lag.k, lag.mean_m, self.res0)
        self.construct = real.codes.index("CONSTRUCT")

    def step(self, y: float, rate_gap: float, z_dem: float = 0.0) -> float:
        desired = residential_desired(self.res0, y, rate_gap, self.income_el, self.rate_semi, z_dem)
        return float(self.smoother.push(desired))

    def add_to_final(self, inv_goods: np.ndarray, res_real: float) -> np.ndarray:
        out = np.asarray(inv_goods, dtype=float).copy()
        out[self.construct] += res_real
        return out
