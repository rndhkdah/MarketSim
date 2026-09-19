"""Impulse-response harness (§2.12)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy

DEFAULT_PERSISTENCE = {
    "demand": 6.0,
    "monetary": 4.0,
    "cost_push": 8.0,
    "supply": 12.0,
    "fiscal": 8.0,
    "row": 6.0,
    "risk_appetite": 3.0,
    "catastrophe": 0.0,
}

CAT_TARGETS = ("CONSTRUCT", "REALESTATE", "AUTOS")


@dataclass
class IRFResult:
    """One shocked path. Index 0 is month 1 after the impulse."""

    gap: np.ndarray
    lvl: np.ndarray
    x: np.ndarray
    x0: np.ndarray
    cpi: np.ndarray
    u: np.ndarray
    r: np.ndarray
    codes: tuple[str, ...]
    gdp0: float
    pi_star: float

    def sector_gap(self, code: str) -> np.ndarray:
        """``x_{t,s} / x0_s − 1``. Dimensionless."""
        i = self.codes.index(code)
        return self.x[:, i] / self.x0[i] - 1.0


def make_economy(
    config_dir: Path,
    *,
    pi_star: float = 0.0,
    overrides: dict[str, Any] | None = None,
    check_sfc: bool = False,
) -> RealEconomy:
    cfg = load_config(config_dir, overrides or {})
    io = load_io(resolve_io_path(cfg))
    return RealEconomy(cfg, io, pi_star=pi_star, check_sfc=check_sfc)


def run_irf(
    kind: str,
    size: float,
    persistence_q: float | None = None,
    months: int = 240,
    *,
    config_dir: Path,
    pi_star: float = 0.0,
    overrides: dict[str, Any] | None = None,
    check_sfc: bool = False,
) -> IRFResult:
    """Shock at month 1, AR(1) decay, ``months`` of history vs the π* path."""
    eco = make_economy(config_dir, pi_star=pi_star, overrides=overrides, check_sfc=check_sfc)
    pq = DEFAULT_PERSISTENCE[kind] if persistence_q is None else float(persistence_q)
    x0 = eco.x.copy()
    gdp0 = eco.fin.gdp0
    gaps = np.zeros(months)
    lvls = np.zeros(months)
    xs = np.zeros((months, eco.real.S))
    cpi = np.zeros(months)
    u = np.zeros(months)
    r = np.zeros(months)
    for t in range(months):
        if t == 0:
            if kind == "catastrophe":
                eco.inject(kind, size, targets=list(CAT_TARGETS), tick=1)
            else:
                eco.inject(kind, size, persistence_q=pq)
        rec = eco.step_month()
        gaps[t] = rec["gdp"] / gdp0 - 1.0
        cpi[t] = rec["cpi"]
        lvls[t] = float(np.log(max(rec["cpi"], 1e-12))) - pi_star * (t + 1) / 12.0
        xs[t] = rec["x"]
        u[t] = rec["U"]
        r[t] = rec["r"]
    return IRFResult(
        gap=gaps,
        lvl=lvls,
        x=xs,
        x0=x0,
        cpi=cpi,
        u=u,
        r=r,
        codes=eco.codes,
        gdp0=gdp0,
        pi_star=pi_star,
    )
