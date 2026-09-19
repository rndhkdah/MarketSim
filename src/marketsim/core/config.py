"""Typed, frozen config. One pydantic model per YAML file; overrides by dotted path."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marketsim.core.errors import ConfigError

ERLANG_RE = re.compile(r"^erlang\((\d+)\)$")

INSTITUTIONS = ("HOUSEHOLD", "GOVT", "CENBANK", "ROW", "LABOUR")
EXTRA_ENDPOINTS = ("ALL_SECTORS", "INVESTMENT", "DISCOUNT_RATE", "BANKSYS", "MM")
PRODUCTION_MODES = ("stock", "order", "flow")


def parse_erlang(value: str | int) -> int:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"erlang k must be >= 1, got {value}")
        return value
    m = ERLANG_RE.match(str(value).strip())
    if not m:
        raise ValueError(f"expected erlang(k), got {value!r}")
    return int(m.group(1))


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SectorParams(FrozenModel):
    eta: float
    eps_own: float
    pass_through: float
    pass_lag_q: float
    inv_lag_q: float
    build_lag_q: float
    dem_rate_semi: float
    nd_ebitda: float
    cf_duration: float
    fixed_cost: float
    util_target: float
    asset_leverage: float = 2.5

    @field_validator("eps_own")
    @classmethod
    def _eps_own_neg(cls, v: float) -> float:
        if v >= 0:
            raise ValueError("eps_own must be < 0")
        return v

    @field_validator("pass_through")
    @classmethod
    def _pt(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("pass_through must satisfy 0 < pass_through <= 1")
        return v

    @field_validator("pass_lag_q", "inv_lag_q", "build_lag_q")
    @classmethod
    def _lags(cls, v: float) -> float:
        if v < 0:
            raise ValueError("lags must be >= 0")
        return v


class FinancialsBlock(FrozenModel):
    nim_rate_beta: float = 0.0
    float_rate_beta: float = 0.0
    curve_beta: float = 0.0
    bond_mtm_duration: float = 0.0


class BondIndex(FrozenModel):
    duration: float = 7.0
    name: str = "GBOND10"


class SectorsConfig(FrozenModel):
    defaults: dict[str, float]
    sectors: dict[str, SectorParams]
    financials: dict[str, FinancialsBlock] = Field(default_factory=dict)
    market_cap_weights: dict[str, float]
    bond_index: BondIndex = Field(default_factory=BondIndex)

    def params(self, code: str) -> SectorParams:
        return self.sectors[code]


class CapexCoefficients(FrozenModel):
    phi_accelerator: float
    psi_utilisation: float
    chi_cost_of_capital: float
    q_tobin: float


class CapexConfig(FrozenModel):
    routing: dict[str, float]
    coefficients: CapexCoefficients
    lag_q: float
    unit: str = "pct_pts_of_K_per_year"
    unit_scale: float = 0.01
    q_scale: float = 0.1
    supply_line_weight: float = 0.85


class BankCapitalGate(FrozenModel):
    midpoint: float = 0.085
    steepness: float = 90.0
    ratio: Literal["equity_over_rwa"] = "equity_over_rwa"
    baseline_capital_ratio: float = 0.125
    normalise_at_baseline: bool = True


class TypedEdge(FrozenModel):
    src: str
    dst: str
    channel: str
    elasticity: float
    lag: str | int = "erlang(2)"
    lag_q: float = 0.0
    sign: str = "+"
    saturation: float | None = None
    up: float | None = None
    down: float | None = None

    @field_validator("lag")
    @classmethod
    def _lag(cls, v: str | int) -> str | int:
        parse_erlang(v)
        return v

    @property
    def erlang_k(self) -> int:
        return parse_erlang(self.lag)


class CreditConfig(FrozenModel):
    bank_capital_gate: BankCapitalGate
    edges: list[TypedEdge] = Field(default_factory=list)


class LabourConfig(FrozenModel):
    wage_share: dict[str, float]
    phillips_slope: float
    wage_stickiness_q: float


class TaylorRule(FrozenModel):
    r_neutral: float
    phi_inflation: float
    phi_output_gap: float
    smoothing: float


class PolicyConfig(FrozenModel):
    taylor: TaylorRule
    automatic_stabiliser: float = -0.35
    discretionary_lag_q: float = 2.0


class ShockSpec(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    persistence_q: float = 0.0


class EdgesConfig(FrozenModel):
    capex: CapexConfig
    credit: CreditConfig
    collateral: list[TypedEdge] = Field(default_factory=list)
    substitution: list[TypedEdge] = Field(default_factory=list)
    complement: list[TypedEdge] = Field(default_factory=list)
    labour: LabourConfig
    policy: PolicyConfig
    shocks: dict[str, ShockSpec]


class WorldSettings(FrozenModel):
    scale: float = 1.0
    seed: int = 0
    mode: Literal["game", "professional"] = "professional"
    run_mode: Literal["lockstep", "realtime"] = "lockstep"
    io_source: Literal["seed", "bea"] = "seed"
    modules: list[str] = Field(default_factory=list)


class DynamicsProduction(FrozenModel):
    mode: dict[str, Literal["stock", "order", "flow"]]


class DynamicsConfig(FrozenModel):
    production: DynamicsProduction


class Config(FrozenModel):
    config_dir: Path
    sectors: SectorsConfig
    edges: EdgesConfig
    world: WorldSettings
    dynamics: DynamicsConfig | None = None

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(self.sectors.sectors.keys())


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return data


def _merge_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(raw.get("defaults") or {})
    merged: dict[str, Any] = {}
    for code, block in (raw.get("sectors") or {}).items():
        row = {**defaults, **(block or {})}
        merged[code] = row
    out = dict(raw)
    out["sectors"] = merged
    return out


def _set_dotted(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = mapping
    for p in parts[:-1]:
        nxt = cur.get(p)
        if nxt is None:
            cur[p] = {}
            nxt = cur[p]
        if not isinstance(nxt, dict):
            raise ConfigError(f"cannot override {path}: {p} is not a mapping")
        cur = nxt
    cur[parts[-1]] = value


def _apply_overrides(raw: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return raw
    out = json.loads(json.dumps(raw, default=str))
    for path, value in overrides.items():
        _set_dotted(out, path, value)
    return out


def _validate_codes(cfg: Config) -> None:
    codes = set(cfg.codes)
    if len(codes) != 18:
        raise ConfigError(f"expected 18 sectors, got {len(codes)}")
    for name, mapping in (
        ("market_cap_weights", cfg.sectors.market_cap_weights),
        ("wage_share", cfg.edges.labour.wage_share),
        ("production.mode", cfg.dynamics.production.mode if cfg.dynamics else {}),
    ):
        if mapping and set(mapping) != codes:
            raise ConfigError(f"{name} must contain every sector code")
    if abs(sum(cfg.sectors.market_cap_weights.values()) - 1.0) > 1e-6:
        raise ConfigError("market_cap_weights must sum to 1")
    if abs(sum(cfg.edges.capex.routing.values()) - 1.0) > 1e-6:
        raise ConfigError("capex.routing must sum to 1")
    allowed = codes | set(INSTITUTIONS) | set(EXTRA_ENDPOINTS)
    edges: list[TypedEdge] = [
        *cfg.edges.credit.edges,
        *cfg.edges.collateral,
        *cfg.edges.substitution,
        *cfg.edges.complement,
    ]
    for e in edges:
        if e.src not in allowed or e.dst not in allowed:
            raise ConfigError(f"edge endpoint not recognised: {e.src} -> {e.dst}")


def load_config(config_dir: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load `sectors.yaml`, `edges.yaml`, `world.yaml`, optional `dynamics.yaml`."""
    root = Path(config_dir).resolve()
    if not root.is_dir():
        raise ConfigError(f"config dir not found: {root}")

    sectors_raw = _merge_defaults(_read_yaml(root / "sectors.yaml"))
    edges_raw = _read_yaml(root / "edges.yaml")
    world_raw = _read_yaml(root / "world.yaml")
    dyn_path = root / "dynamics.yaml"
    dynamics_raw = _read_yaml(dyn_path) if dyn_path.exists() else None

    bundle = {
        "sectors": sectors_raw,
        "edges": edges_raw,
        "world": world_raw,
        "dynamics": dynamics_raw,
    }
    # overrides use dotted paths from the bundle root, e.g. world.seed or dynamics.prices.kappa_util
    if overrides:
        for path, value in overrides.items():
            _set_dotted(bundle, path, value)

    try:
        sectors = SectorsConfig.model_validate(bundle["sectors"])
        edges = EdgesConfig.model_validate(bundle["edges"])
        world = WorldSettings.model_validate(bundle["world"])
        dynamics = (
            DynamicsConfig.model_validate(bundle["dynamics"]) if bundle["dynamics"] is not None else None
        )
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(str(exc)) from exc

    cfg = Config(config_dir=root, sectors=sectors, edges=edges, world=world, dynamics=dynamics)
    _validate_codes(cfg)
    return cfg


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config"
