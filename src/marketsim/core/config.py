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


class ErlangLag(FrozenModel):
    k: int
    mean_m: float

    @field_validator("k")
    @classmethod
    def _k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("erlang k must be >= 1")
        return v

    @field_validator("mean_m")
    @classmethod
    def _mean(cls, v: float) -> float:
        if v < 0:
            raise ValueError("erlang mean_m must be >= 0")
        return v


def _positive(name: str, v: float) -> float:
    if v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v


def _unit_interval(name: str, v: float) -> float:
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be in [0, 1]")
    return v


class ExpectationsCfg(FrozenModel):
    tau_sales_m: float
    tau_growth_m: float
    anchor_growth: float
    infl_anchor: float
    tau_infl_m: float

    @field_validator("tau_sales_m", "tau_growth_m", "tau_infl_m")
    @classmethod
    def _tau(cls, v: float) -> float:
        return _positive("tau", v)

    @field_validator("anchor_growth", "infl_anchor")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("anchor", v)


class CriticalInputCfg(FrozenModel):
    min_share: float
    suppliers: list[str]

    @field_validator("min_share")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("min_share", v)


class DynamicsProduction(FrozenModel):
    mode: dict[str, Literal["stock", "order", "flow"]]
    cover_scale: float
    tau_inv_mult: float
    order_book_scale: float
    input_cover_m: float
    tau_input_m: float
    backlog_loss: float
    tau_backlog_m: float
    overtime_cap: float
    critical_input: CriticalInputCfg
    leak_per_month: dict[str, float] = Field(default_factory=dict)

    @field_validator("tau_inv_mult", "tau_input_m", "tau_backlog_m", "overtime_cap", "cover_scale", "order_book_scale")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("tau/scale", v)

    @field_validator("input_cover_m")
    @classmethod
    def _cover(cls, v: float) -> float:
        if v < 0:
            raise ValueError("input_cover_m must be >= 0")
        return v

    @field_validator("backlog_loss")
    @classmethod
    def _loss(cls, v: float) -> float:
        return _unit_interval("backlog_loss", v)

    @field_validator("leak_per_month")
    @classmethod
    def _leak(cls, v: dict[str, float]) -> dict[str, float]:
        for code, leak in v.items():
            if leak < 0:
                raise ValueError(f"leak for {code} must be >= 0")
        return v


class PricesCfg(FrozenModel):
    kappa_util: float
    gamma_cover: float
    cover_floor: float
    step_max_month: float
    fast_mean_m: float

    @field_validator("fast_mean_m", "step_max_month", "cover_floor")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("price param", v)


class DynamicsCapexCfg(FrozenModel):
    delta_annual: float
    start_rate_cap_mult: float
    tau_util_m: float
    q_clip: float

    @field_validator("delta_annual", "start_rate_cap_mult", "tau_util_m", "q_clip")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("capex param", v)


class ResidentialCfg(FrozenModel):
    share_of_investment: float
    rate_semi: float
    income_elasticity: float
    lag: ErlangLag

    @field_validator("share_of_investment")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("share_of_investment", v)


class HouseholdsCfg(FrozenModel):
    alpha1: float
    tau_income_m: float
    rate_budget_passthrough: float
    rate_lag: ErlangLag

    @field_validator("alpha1", "rate_budget_passthrough")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("household share", v)

    @field_validator("tau_income_m")
    @classmethod
    def _tau(cls, v: float) -> float:
        return _positive("tau_income_m", v)


class DynamicsLabourCfg(FrozenModel):
    tau_hire_m: float
    tau_fire_m: float
    u_star: float
    lf_cap: float

    @field_validator("tau_hire_m", "tau_fire_m")
    @classmethod
    def _tau(cls, v: float) -> float:
        return _positive("labour tau", v)

    @field_validator("u_star", "lf_cap")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("labour share", v)


class FiscalCfg(FrozenModel):
    debt_to_gdp: float
    kappa_debt: float
    tau_tax_m: float
    tax_rate_bounds: tuple[float, float]
    benefit_replacement: float
    corp_tax: float
    vat: float

    @field_validator("debt_to_gdp", "benefit_replacement", "corp_tax", "vat")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("fiscal share", v)

    @field_validator("tau_tax_m")
    @classmethod
    def _tau(cls, v: float) -> float:
        return _positive("tau_tax_m", v)


class FirmsCfg(FrozenModel):
    kappa_leverage: float
    tau_profit_m: float
    tau_ebitda_m: float
    base_spread: float
    spread_leverage_floor: float

    @field_validator("tau_profit_m", "tau_ebitda_m")
    @classmethod
    def _tau(cls, v: float) -> float:
        return _positive("firm tau", v)


class RowCfg(FrozenModel):
    export_price_elasticity: float
    import_prior: dict[str, float]


class DynamicsPolicyCfg(FrozenModel):
    elb: float
    core_weight: float

    @field_validator("core_weight")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("core_weight", v)


class BanksCfg(FrozenModel):
    mode: Literal["passthrough", "full"]
    dep_margin: float
    ll0: float
    kappa_ll: float
    ll_cap_mult: float
    capital_target: float
    reserves_to_deposits: float

    @field_validator("capital_target", "reserves_to_deposits")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("bank share", v)


class DynamicsCreditCfg(FrozenModel):
    pricing: Literal["uniform", "risk_based"]
    s0: float
    corp_funding_mix: float
    gate_enabled: bool
    s_gate: float
    s_loss: float
    collateral_trend_years: float

    @field_validator("corp_funding_mix")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("corp_funding_mix", v)


class DynamicsShocksCfg(FrozenModel):
    animal_spirits_weight: float
    cost_push_targets: dict[str, float]


class DynamicsConfig(FrozenModel):
    expectations: ExpectationsCfg
    production: DynamicsProduction
    prices: PricesCfg
    capex: DynamicsCapexCfg
    residential: ResidentialCfg
    households: HouseholdsCfg
    labour: DynamicsLabourCfg
    fiscal: FiscalCfg
    firms: FirmsCfg
    row: RowCfg
    policy: DynamicsPolicyCfg
    banks: BanksCfg
    credit: DynamicsCreditCfg
    shocks: DynamicsShocksCfg

    @field_validator("production")
    @classmethod
    def _leak_stock_only(cls, prod: DynamicsProduction) -> DynamicsProduction:
        for code, leak in prod.leak_per_month.items():
            mode = prod.mode.get(code)
            if mode is None:
                raise ValueError(f"leak on unknown sector {code}")
            if leak > 0 and mode != "stock":
                raise ValueError(f"leak only on stock-mode sectors, got {code}={mode}")
        return prod


class TaxLimits(FrozenModel):
    tax_rate: tuple[float, float] = (0.0, 0.6)
    tariff: float = 0.5
    vat: tuple[float, float] = (0.0, 0.6)
    delta_g_of_gdp_per_quarter: float = 0.02


class GovtAuthorityCfg(FrozenModel):
    control: Literal["autopilot", "scripted", "agent"] = "autopilot"
    legislative_lag_m: int = 1
    discretionary_lag_q: float = 2.0
    limits: TaxLimits = Field(default_factory=TaxLimits)


class CenbankLimits(FrozenModel):
    rate_move: float = 0.02


class CenbankAuthorityCfg(FrozenModel):
    control: Literal["autopilot", "scripted", "agent"] = "autopilot"
    limits: CenbankLimits = Field(default_factory=CenbankLimits)


class MonetaryCalendarCfg(FrozenModel):
    meetings_per_year: int = 8
    rate_step: float = 0.0025
    deadband: float = 0.0010
    blackout_days: int = 10
    minutes_lag_days: int = 21


class MonetaryRuleCfg(FrozenModel):
    phi_inflation: float = 1.50
    phi_u: float = 1.0
    phi_y_mult: float = 0.0
    core_weight: float = 0.5
    smoothing: float = 0.80
    kappa_rstar: float = 1.0
    rstar_tau_m: int = 60


class MonetaryDataCfg(FrozenModel):
    cpi_lag_m: int = 1
    unemployment_lag_m: int = 1
    gdp_lag_q: int = 1
    use_published_vintages: bool = True


class MonetaryStrategyCfg(FrozenModel):
    makeup: float = 0.0
    makeup_decay: float = 0.98
    makeup_clip: float = 0.02


class RiskManagementCfg(FrozenModel):
    enabled: bool = False
    sahm_threshold: float = 0.005
    sahm_cut: float = 0.005
    decay_m: int = 12


class FciCfg(FrozenModel):
    phi_fci: float = 0.0
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "corp_spread": 0.4,
            "capital_gate": 0.3,
            "equity_drawdown": 0.2,
            "term_premium": 0.1,
        }
    )


class MonetaryCreditCfg(FrozenModel):
    phi_credit: float = 0.0


class ElbToolkitCfg(FrozenModel):
    rate: float = 0.0
    allow_negative: bool = False
    guidance_credibility: float = 0.7
    qe_per_gap_point: float = 0.02


class CommitteeCfg(FrozenModel):
    dispersion_bp: float = 0.0
    projection_noise_bp: float = 25.0


class MonetaryFrameworkCfg(FrozenModel):
    control: Literal["autopilot", "scripted", "agent"] = "autopilot"
    calendar: MonetaryCalendarCfg = Field(default_factory=MonetaryCalendarCfg)
    rule: MonetaryRuleCfg = Field(default_factory=MonetaryRuleCfg)
    data: MonetaryDataCfg = Field(default_factory=MonetaryDataCfg)
    strategy: MonetaryStrategyCfg = Field(default_factory=MonetaryStrategyCfg)
    risk_management: RiskManagementCfg = Field(default_factory=RiskManagementCfg)
    financial_conditions: FciCfg = Field(default_factory=FciCfg)
    credit: MonetaryCreditCfg = Field(default_factory=MonetaryCreditCfg)
    elb: ElbToolkitCfg = Field(default_factory=ElbToolkitCfg)
    committee: CommitteeCfg = Field(default_factory=CommitteeCfg)


class PolicyFile(FrozenModel):
    govt: GovtAuthorityCfg = Field(default_factory=GovtAuthorityCfg)
    cenbank: CenbankAuthorityCfg = Field(default_factory=CenbankAuthorityCfg)
    monetary: MonetaryFrameworkCfg = Field(default_factory=MonetaryFrameworkCfg)


class Config(FrozenModel):
    config_dir: Path
    sectors: SectorsConfig
    edges: EdgesConfig
    world: WorldSettings
    dynamics: DynamicsConfig | None = None
    policy: PolicyFile | None = None

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

    pol_path = root / "policy.yaml"
    policy_raw = _read_yaml(pol_path) if pol_path.exists() else None

    bundle = {
        "sectors": sectors_raw,
        "edges": edges_raw,
        "world": world_raw,
        "dynamics": dynamics_raw,
        "policy": policy_raw,
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
        policy = PolicyFile.model_validate(bundle["policy"]) if bundle["policy"] is not None else None
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(str(exc)) from exc

    cfg = Config(config_dir=root, sectors=sectors, edges=edges, world=world, dynamics=dynamics, policy=policy)
    _validate_codes(cfg)
    return cfg


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config"
