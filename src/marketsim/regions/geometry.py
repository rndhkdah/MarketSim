"""Regional geometry: capacity-share targets, link matrices, cell masks (T3.01 / §3.2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketsim.core.errors import ConfigError


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RegionSpec(FrozenModel):
    """One region. ``population_share`` is dimensionless; ``wage_level`` is an index (1 at INDUSTRIAL)."""

    code: str
    population_share: float
    wage_level: float = 1.0

    @field_validator("population_share", "wage_level")
    @classmethod
    def _pos(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("population_share and wage_level must be > 0")
        return v


class LinkSpec(FrozenModel):
    """Undirected preference-friction link. ``cost`` is a log-index; ``capacity_mult`` scales baseline flow."""

    a: str
    b: str
    cost: float
    capacity_mult: float = 1.0

    @field_validator("cost")
    @classmethod
    def _cost(cls, v: float) -> float:
        if v < 0:
            raise ValueError("link cost must be >= 0")
        return v

    @field_validator("capacity_mult")
    @classmethod
    def _cap(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("capacity_mult must be > 0")
        return v

    @model_validator(mode="after")
    def _ends(self) -> LinkSpec:
        if self.a == self.b:
            raise ValueError("link endpoints must differ")
        return self


class TradeCfg(FrozenModel):
    tau_share_m: float = 6.0  # months; Armington share smoother
    home_bias: float = 1.5  # dimensionless gravity weight on s = d
    gravity_theta: float = 8.0  # 1/cost units in exp(−θ·cost)
    availability_kappa: float = 1.0  # exponent on fill-rate availability

    @field_validator("tau_share_m", "home_bias", "gravity_theta", "availability_kappa")
    @classmethod
    def _pos(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("trade parameters must be > 0")
        return v


class MigrationCfg(FrozenModel):
    enabled: bool = True
    rate_per_pp_per_year: float = 0.001  # share of LF per year per pp U-gap

    @field_validator("rate_per_pp_per_year")
    @classmethod
    def _rate(cls, v: float) -> float:
        if v < 0:
            raise ValueError("migration rate must be >= 0")
        return v


class RegionsConfig(FrozenModel):
    """Pydantic schema for ``config/regions.yaml``."""

    regions: list[RegionSpec]
    location_quotient: dict[str, dict[str, float]] = Field(default_factory=dict)
    tradability: dict[str, float] = Field(default_factory=dict)
    armington_sigma: dict[str, float] = Field(default_factory=dict)
    links: list[LinkSpec] = Field(default_factory=list)
    trade: TradeCfg = Field(default_factory=TradeCfg)
    migration: MigrationCfg = Field(default_factory=MigrationCfg)

    @model_validator(mode="after")
    def _shares(self) -> RegionsConfig:
        if not self.regions:
            raise ValueError("regions must list at least one region")
        codes = [r.code for r in self.regions]
        if len(set(codes)) != len(codes):
            raise ValueError("region codes must be unique")
        total = sum(r.population_share for r in self.regions)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("population_share must sum to 1")
        known = set(codes)
        for src, dst in ((lnk.a, lnk.b) for lnk in self.links):
            if src not in known or dst not in known:
                raise ValueError(f"link endpoint not a region: {src}–{dst}")
        for rcode in self.location_quotient:
            if rcode not in known:
                raise ValueError(f"location_quotient region not recognised: {rcode}")
        return self


@dataclass(frozen=True)
class Geometry:
    """Dense regional arrays. Capacity shares are (R, S); link matrices are (R, R)."""

    codes: tuple[str, ...]
    sector_codes: tuple[str, ...]
    population_share: np.ndarray  # (R,) dimensionless
    wage_level: np.ndarray  # (R,) index
    location_quotient: np.ndarray  # (R, S) dimensionless; omitted YAML entries = 1
    tradability: np.ndarray  # (S,) in [0, 1]
    armington_sigma: np.ndarray  # (S,) substitution elasticity
    cost: np.ndarray  # (R, R) preference friction; 0 on the diagonal
    capacity_mult: np.ndarray  # (R, R) link capacity vs baseline flow; 0 if no link
    capacity_share: np.ndarray  # (R, S) target K shares; columns sum to 1
    trade: TradeCfg
    migration: MigrationCfg

    @property
    def n_regions(self) -> int:
        return len(self.codes)

    @property
    def n_sectors(self) -> int:
        return len(self.sector_codes)

    def mask(
        self,
        regions: Sequence[str] | None = None,
        sectors: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Boolean ``(R, S)`` mask. ``None`` means every region / sector."""
        r_ok = np.ones(self.n_regions, dtype=bool)
        s_ok = np.ones(self.n_sectors, dtype=bool)
        if regions is not None:
            want = set(regions)
            r_ok = np.array([c in want for c in self.codes], dtype=bool)
        if sectors is not None:
            want_s = set(sectors)
            s_ok = np.array([c in want_s for c in self.sector_codes], dtype=bool)
        return r_ok[:, None] & s_ok[None, :]


def target_capacity_share(
    population_share: np.ndarray,
    location_quotient: np.ndarray,
) -> np.ndarray:
    """``share[s, i] ∝ pop_s × LQ[s, i]``, columns sum to 1. Dimensionless."""
    raw = np.asarray(population_share, dtype=float)[:, None] * np.asarray(location_quotient, dtype=float)
    col = raw.sum(axis=0, keepdims=True)
    if np.any(col <= 0):
        raise ConfigError("capacity-share column is empty")
    return raw / col


def build_geometry(cfg: RegionsConfig, sector_codes: Sequence[str]) -> Geometry:
    """Stack YAML into dense arrays in ``sector_codes`` / region-list order."""
    sectors = tuple(sector_codes)
    regions = tuple(r.code for r in cfg.regions)
    r_idx = {c: i for i, c in enumerate(regions)}
    s_idx = {c: i for i, c in enumerate(sectors)}
    n_r = len(regions)
    n_s = len(sectors)

    pop = np.array([r.population_share for r in cfg.regions], dtype=float)
    wage = np.array([r.wage_level for r in cfg.regions], dtype=float)

    lq = np.ones((n_r, n_s), dtype=float)
    for rcode, mapping in cfg.location_quotient.items():
        ri = r_idx[rcode]
        for scode, val in mapping.items():
            if scode not in s_idx:
                raise ConfigError(f"location_quotient sector not recognised: {scode}")
            if val <= 0:
                raise ConfigError(f"location_quotient must be > 0, got {rcode}.{scode}={val}")
            lq[ri, s_idx[scode]] = float(val)

    trad = np.ones(n_s, dtype=float)
    for scode, val in cfg.tradability.items():
        if scode not in s_idx:
            raise ConfigError(f"tradability sector not recognised: {scode}")
        if not 0.0 <= val <= 1.0:
            raise ConfigError(f"tradability must be in [0, 1], got {scode}={val}")
        trad[s_idx[scode]] = float(val)

    sigma = np.full(n_s, float(cfg.armington_sigma.get("default", 2.0)), dtype=float)
    for scode, val in cfg.armington_sigma.items():
        if scode == "default":
            continue
        if scode not in s_idx:
            raise ConfigError(f"armington_sigma sector not recognised: {scode}")
        if val <= 0:
            raise ConfigError(f"armington_sigma must be > 0, got {scode}={val}")
        sigma[s_idx[scode]] = float(val)

    cost = np.zeros((n_r, n_r), dtype=float)
    cap = np.zeros((n_r, n_r), dtype=float)
    seen: set[tuple[int, int]] = set()
    for link in cfg.links:
        i, j = r_idx[link.a], r_idx[link.b]
        pair = (min(i, j), max(i, j))
        if pair in seen:
            raise ConfigError(f"duplicate link {link.a}–{link.b}")
        seen.add(pair)
        cost[i, j] = cost[j, i] = float(link.cost)
        cap[i, j] = cap[j, i] = float(link.capacity_mult)

    shares = target_capacity_share(pop, lq)
    return Geometry(
        codes=regions,
        sector_codes=sectors,
        population_share=pop,
        wage_level=wage,
        location_quotient=lq,
        tradability=trad,
        armington_sigma=sigma,
        cost=cost,
        capacity_mult=cap,
        capacity_share=shares,
        trade=cfg.trade,
        migration=cfg.migration,
    )
