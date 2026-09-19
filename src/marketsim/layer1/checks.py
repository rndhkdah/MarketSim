"""Layer-1 structural and rotation checks. Always go through `load_io`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.config import Config
from marketsim.layer1.betas import Betas, derive_betas
from marketsim.layer1.io import IOTable

GOLDEN_NNZ = 219
GOLDEN_RHO = 0.536
GOLDEN_MULT = (1.58, 2.77)
GOLDEN_GO = 209.3
GOLDEN_WAGES = 51.82
GOLDEN_GOS = 48.18
EARLY_CYCLE = ("AUTOS", "CONSTRUCT", "CAPGOODS", "TRANSPORT", "SOFTWARE")
RECESSION = ("STAPLES", "HEALTH", "REALESTATE", "UTILITIES", "TELECOM")


@dataclass(frozen=True)
class StructureReport:
    n: int
    nnz: int
    rho: float
    mult_min: float
    mult_max: float
    go: float
    wages: float
    gos: float
    oil_top3: tuple[str, ...]


def structure_report(io: IOTable, cfg: Config) -> StructureReport:
    d = io.baseline_final_demand(100.0)
    x = io.L @ d
    va = (1.0 - io.mu) * x
    ws = np.array([cfg.edges.labour.wage_share[c] for c in io.codes])
    wages = float((ws * va).sum())
    gos = float(((1.0 - ws) * va).sum())
    energy = io.index["ENERGY"]
    cost = io.G[:, energy]
    ranking = sorted(
        ((io.codes[i], float(cost[i])) for i in range(io.n) if i != energy),
        key=lambda t: -t[1],
    )
    mult = io.output_multipliers()
    return StructureReport(
        n=io.n,
        nnz=int(np.count_nonzero(io.A > 0)),
        rho=io.spectral_radius(),
        mult_min=float(mult.min()),
        mult_max=float(mult.max()),
        go=float(x.sum()),
        wages=wages,
        gos=gos,
        oil_top3=tuple(c for c, _ in ranking[:3]),
    )


def assert_structure(io: IOTable, cfg: Config) -> StructureReport:
    r = structure_report(io, cfg)
    assert r.n == 18, r.n
    assert r.nnz == GOLDEN_NNZ, r.nnz
    assert abs(r.rho - GOLDEN_RHO) < 5e-4, r.rho
    assert GOLDEN_MULT[0] <= r.mult_min <= r.mult_max <= GOLDEN_MULT[1], (r.mult_min, r.mult_max)
    assert abs(r.go - GOLDEN_GO) < 0.15, r.go
    assert abs(r.wages - GOLDEN_WAGES) < 0.05, r.wages
    assert abs(r.gos - GOLDEN_GOS) < 0.05, r.gos
    assert set(r.oil_top3) == {"UTILITIES", "MATERIALS", "TRANSPORT"}, r.oil_top3
    assert np.all(io.L > -1e-9)
    return r


def rotation_sets(betas: Betas) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Early-cycle = highest growth betas; recession = lowest."""
    order = [code for _, code in sorted(zip(betas.beta_growth, betas.codes, strict=True), reverse=True)]
    return tuple(order[:5]), tuple(reversed(order[-5:]))


def assert_rotation(io: IOTable, cfg: Config, betas: Betas | None = None) -> None:
    b = betas or derive_betas(io, cfg)
    early, late = rotation_sets(b)
    assert set(early) == set(EARLY_CYCLE), early
    assert set(late) == set(RECESSION), late
