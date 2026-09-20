"""Firm / plant state, registry, and `firms.yaml` schema (T5.01 / §5.2–§5.7)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketsim.core.errors import ConfigError, StateError

LEVERS: tuple[str, ...] = (
    "pricing",
    "production",
    "labour",
    "procurement",
    "capex",
    "financing",
    "treasury",
    "exit",
)
STATUSES: tuple[str, ...] = ("active", "distressed", "bankrupt", "liquidated")
RATINGS: tuple[str, ...] = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC")

Status = Literal["active", "distressed", "bankrupt", "liquidated"]
Rating = Literal["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _unit_interval(name: str, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return v


def _positive(name: str, v: float) -> float:
    if v <= 0:
        raise ValueError(f"{name} must be > 0")
    return v


def _nonneg(name: str, v: float) -> float:
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


class GoodsMarketCfg(FrozenModel):
    epsilon_s: float = 4.0  # dimensionless; §5.3
    availability_kappa: float = 1.0  # dimensionless
    tau_loyalty_m: float = 6.0  # months

    @field_validator("epsilon_s", "availability_kappa", "tau_loyalty_m")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("goods_market param", v)


class PricingCfg(FrozenModel):
    step_max_month: float = 0.15  # log-points / month
    floor_of_pref: float = 0.01  # share of p_ref

    @field_validator("step_max_month")
    @classmethod
    def _step(cls, v: float) -> float:
        return _positive("step_max_month", v)

    @field_validator("floor_of_pref")
    @classmethod
    def _floor(cls, v: float) -> float:
        return _unit_interval("floor_of_pref", v)


class ProductionCfg(FrozenModel):
    overtime_cap: float = 1.10  # output ≤ capacity × this

    @field_validator("overtime_cap")
    @classmethod
    def _ot(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("overtime_cap must be >= 1")
        return v


class LabourCfg(FrozenModel):
    hire_share_of_unemployed: float = 0.10  # share of regional U / month
    firing_cost_months: float = 2.0  # months of wage
    notice_m: int = 1  # months
    matching_m0: float = 1.0  # matching efficiency (§5.5)
    epsilon_w: float = 1.0  # wage elasticity of fills

    @field_validator("hire_share_of_unemployed")
    @classmethod
    def _hire(cls, v: float) -> float:
        return _unit_interval("hire_share_of_unemployed", v)

    @field_validator("firing_cost_months", "matching_m0", "epsilon_w")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("labour param", v)

    @field_validator("notice_m")
    @classmethod
    def _notice(cls, v: int) -> int:
        if v < 0:
            raise ValueError("notice_m must be >= 0")
        return v


class ProcurementCfg(FrozenModel):
    input_cover_max_m: float = 6.0  # months
    premium_max: float = 0.5  # dimensionless
    beta_premium: float = 2.0  # 1 / premium units
    rho_relationship: float = 1.0  # 1 / relationship units
    order_size_cap_of_cell: float = 1.0  # share of cell supply

    @field_validator("input_cover_max_m", "beta_premium", "rho_relationship")
    @classmethod
    def _pos(cls, v: float) -> float:
        return _positive("procurement param", v)

    @field_validator("premium_max")
    @classmethod
    def _prem(cls, v: float) -> float:
        return _nonneg("premium_max", v)

    @field_validator("order_size_cap_of_cell")
    @classmethod
    def _cap(cls, v: float) -> float:
        return _positive("order_size_cap_of_cell", v)


class CapexCfg(FrozenModel):
    new_cell_setup_premium: float = 0.10  # share of plant cost
    capacity_growth_cap_year: float = 0.50  # share of own K / year
    payment_erlang_k: int = 3

    @field_validator("new_cell_setup_premium", "capacity_growth_cap_year")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("capex share", v)

    @field_validator("payment_erlang_k")
    @classmethod
    def _k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("payment_erlang_k must be >= 1")
        return v


class FinancingCfg(FrozenModel):
    nd_ebitda_soft: float = 4.0  # ND / EBITDA
    nd_ebitda_hard: float = 6.0  # ND / EBITDA
    credit_line_of_replacement: float = 0.50  # share of plant replacement value
    loss_carryforward_years: int = 5
    rating_multipliers: dict[str, float] = Field(
        default_factory=lambda: {
            "AAA": 1.2,
            "AA": 1.1,
            "A": 1.0,
            "BBB": 0.9,
            "BB": 0.7,
            "B": 0.5,
            "CCC": 0.25,
        }
    )

    @field_validator("nd_ebitda_soft", "nd_ebitda_hard")
    @classmethod
    def _lev(cls, v: float) -> float:
        return _positive("nd_ebitda", v)

    @field_validator("credit_line_of_replacement")
    @classmethod
    def _line(cls, v: float) -> float:
        return _unit_interval("credit_line_of_replacement", v)

    @field_validator("loss_carryforward_years")
    @classmethod
    def _years(cls, v: int) -> int:
        if v < 0:
            raise ValueError("loss_carryforward_years must be >= 0")
        return v

    @model_validator(mode="after")
    def _ratings_and_caps(self) -> FinancingCfg:
        if self.nd_ebitda_hard < self.nd_ebitda_soft:
            raise ValueError("nd_ebitda_hard must be >= nd_ebitda_soft")
        missing = set(RATINGS) - set(self.rating_multipliers)
        extra = set(self.rating_multipliers) - set(RATINGS)
        if missing or extra:
            raise ValueError(f"rating_multipliers must be exactly {RATINGS}")
        for name, mult in self.rating_multipliers.items():
            if mult <= 0:
                raise ValueError(f"rating multiplier {name} must be > 0")
        return self


class TreasuryCfg(FrozenModel):
    default_enabled: bool = False


class ExitCfg(FrozenModel):
    plant_npc_discount: float = 0.30  # share of replacement value

    @field_validator("plant_npc_discount")
    @classmethod
    def _disc(cls, v: float) -> float:
        return _unit_interval("plant_npc_discount", v)


class BankruptcyCfg(FrozenModel):
    equity_negative_months: int = 3
    plant_recovery: float = 0.70  # share of replacement
    inventory_recovery: float = 0.50  # share of book
    large_threshold_of_monthly_gdp: float = 0.01  # share of monthly GDP

    @field_validator("equity_negative_months")
    @classmethod
    def _months(cls, v: int) -> int:
        if v < 1:
            raise ValueError("equity_negative_months must be >= 1")
        return v

    @field_validator("plant_recovery", "inventory_recovery", "large_threshold_of_monthly_gdp")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("bankruptcy share", v)


class ReportsCfg(FrozenModel):
    monthly_lag_days: int = 10
    quarterly_lag_days: int = 21

    @field_validator("monthly_lag_days", "quarterly_lag_days")
    @classmethod
    def _days(cls, v: int) -> int:
        if v < 0:
            raise ValueError("report lag must be >= 0")
        return v


class ControlsCfg(FrozenModel):
    cell_share_cap: float = 1.0  # 1.0 = regulator off
    decision_size_cap: float = 1.0  # share of cell

    @field_validator("cell_share_cap", "decision_size_cap")
    @classmethod
    def _share(cls, v: float) -> float:
        return _unit_interval("control share", v)


class FirmsFile(FrozenModel):
    """Root of `config/firms.yaml`. Every §5.3–§5.7 limit lives here."""

    npc_initial_share: float = 1.0  # dimensionless
    goods_market: GoodsMarketCfg = Field(default_factory=GoodsMarketCfg)
    pricing: PricingCfg = Field(default_factory=PricingCfg)
    production: ProductionCfg = Field(default_factory=ProductionCfg)
    labour: LabourCfg = Field(default_factory=LabourCfg)
    procurement: ProcurementCfg = Field(default_factory=ProcurementCfg)
    capex: CapexCfg = Field(default_factory=CapexCfg)
    financing: FinancingCfg = Field(default_factory=FinancingCfg)
    treasury: TreasuryCfg = Field(default_factory=TreasuryCfg)
    exit: ExitCfg = Field(default_factory=ExitCfg)
    bankruptcy: BankruptcyCfg = Field(default_factory=BankruptcyCfg)
    reports: ReportsCfg = Field(default_factory=ReportsCfg)
    controls: ControlsCfg = Field(default_factory=ControlsCfg)

    @field_validator("npc_initial_share")
    @classmethod
    def _npc(cls, v: float) -> float:
        return _unit_interval("npc_initial_share", v)


def cells_from_lists(regions: Sequence[str], sectors: Sequence[str]) -> frozenset[tuple[str, str]]:
    """Cartesian product of region × sector codes (the legal cell set)."""
    return frozenset((str(r), str(s)) for r in regions for s in sectors)


def _cell_rows(mapping: Mapping[tuple[str, str], float]) -> list[list[Any]]:
    return [[r, s, float(v)] for (r, s), v in sorted(mapping.items())]


def _load_cell_rows(rows: Sequence[Sequence[Any]] | None) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for row in rows or ():
        if len(row) != 3:
            raise StateError(f"cell map row must be [region, sector, value], got {row!r}")
        out[(str(row[0]), str(row[1]))] = float(row[2])
    return out


def _default_autopilot() -> dict[str, bool]:
    return {lever: True for lever in LEVERS}


@dataclass
class Plant:
    """One plant. ``capacity`` is real output units / month; ``productivity`` is a factor."""

    region: str
    sector: str
    capacity: float
    productivity: float = 1.0
    vintage: int = 0  # month index
    construction_pipeline: tuple[tuple[int, float], ...] = ()  # (months remaining, capacity)

    def __post_init__(self) -> None:
        self.region = str(self.region)
        self.sector = str(self.sector)
        self.capacity = float(self.capacity)
        self.productivity = float(self.productivity)
        self.vintage = int(self.vintage)
        pipe: list[tuple[int, float]] = []
        for item in self.construction_pipeline:
            months, cap = int(item[0]), float(item[1])
            if months < 0:
                raise ConfigError("construction pipeline months must be >= 0")
            pipe.append((months, cap))
        self.construction_pipeline = tuple(pipe)
        if self.capacity < 0 or self.productivity < 0:
            raise ConfigError("plant capacity and productivity must be >= 0")

    @property
    def cell(self) -> tuple[str, str]:
        return (self.region, self.sector)

    def to_state(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "sector": self.sector,
            "capacity": self.capacity,
            "productivity": self.productivity,
            "vintage": self.vintage,
            "construction_pipeline": [list(item) for item in self.construction_pipeline],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Plant:
        pipe = tuple((int(a), float(b)) for a, b in state.get("construction_pipeline", ()))
        return cls(
            region=state["region"],
            sector=state["sector"],
            capacity=float(state["capacity"]),
            productivity=float(state.get("productivity", 1.0)),
            vintage=int(state.get("vintage", 0)),
            construction_pipeline=pipe,
        )


@dataclass
class Firm:
    """Agent-operated firm. Ledger entity name is ``FIRM:<id>`` (T5.02)."""

    id: str
    operator: str
    status: Status = "active"
    plants: tuple[Plant, ...] = ()
    employees: float = 0.0  # persons
    wage_offer: float = 0.0  # cr / person / month
    vacancies: float = 0.0  # persons
    finished_inventories: dict[tuple[str, str], float] = field(default_factory=dict)
    input_inventories: dict[tuple[str, str], float] = field(default_factory=dict)
    backlog: dict[tuple[str, str], float] = field(default_factory=dict)
    posted_price: dict[tuple[str, str], float] = field(default_factory=dict)
    customer_share: dict[tuple[str, str], float] = field(default_factory=dict)
    cash: float = 0.0  # cr
    credit_limit: float = 0.0  # cr
    credit_drawn: float = 0.0  # cr
    term_debt: float = 0.0  # cr
    shares_outstanding: float = 0.0  # shares
    rating: Rating = "BBB"
    tax_loss_carryforward: float = 0.0  # cr
    autopilot: dict[str, bool] = field(default_factory=_default_autopilot)
    treasury_enabled: bool = False
    equity_negative_months: int = 0

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.operator = str(self.operator)
        if self.status not in STATUSES:
            raise ConfigError(f"unknown firm status {self.status!r}")
        if self.rating not in RATINGS:
            raise ConfigError(f"unknown rating {self.rating!r}")
        self.plants = tuple(self.plants)
        flags = _default_autopilot()
        flags.update({str(k): bool(v) for k, v in self.autopilot.items()})
        extra = set(flags) - set(LEVERS)
        if extra:
            raise ConfigError(f"unknown autopilot levers {sorted(extra)}")
        self.autopilot = flags
        if self.employees < 0 or self.vacancies < 0 or self.wage_offer < 0:
            raise ConfigError("employees, vacancies and wage_offer must be >= 0")
        if self.shares_outstanding < 0:
            raise ConfigError("shares_outstanding must be >= 0")

    def to_state(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operator": self.operator,
            "status": self.status,
            "plants": [p.to_state() for p in self.plants],
            "employees": self.employees,
            "wage_offer": self.wage_offer,
            "vacancies": self.vacancies,
            "finished_inventories": _cell_rows(self.finished_inventories),
            "input_inventories": _cell_rows(self.input_inventories),
            "backlog": _cell_rows(self.backlog),
            "posted_price": _cell_rows(self.posted_price),
            "customer_share": _cell_rows(self.customer_share),
            "cash": self.cash,
            "credit_limit": self.credit_limit,
            "credit_drawn": self.credit_drawn,
            "term_debt": self.term_debt,
            "shares_outstanding": self.shares_outstanding,
            "rating": self.rating,
            "tax_loss_carryforward": self.tax_loss_carryforward,
            "autopilot": {k: self.autopilot[k] for k in LEVERS},
            "treasury_enabled": self.treasury_enabled,
            "equity_negative_months": self.equity_negative_months,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Firm:
        plants = tuple(Plant.from_state(p) for p in state.get("plants", ()))
        return cls(
            id=state["id"],
            operator=state["operator"],
            status=state.get("status", "active"),
            plants=plants,
            employees=float(state.get("employees", 0.0)),
            wage_offer=float(state.get("wage_offer", 0.0)),
            vacancies=float(state.get("vacancies", 0.0)),
            finished_inventories=_load_cell_rows(state.get("finished_inventories")),
            input_inventories=_load_cell_rows(state.get("input_inventories")),
            backlog=_load_cell_rows(state.get("backlog")),
            posted_price=_load_cell_rows(state.get("posted_price")),
            customer_share=_load_cell_rows(state.get("customer_share")),
            cash=float(state.get("cash", 0.0)),
            credit_limit=float(state.get("credit_limit", 0.0)),
            credit_drawn=float(state.get("credit_drawn", 0.0)),
            term_debt=float(state.get("term_debt", 0.0)),
            shares_outstanding=float(state.get("shares_outstanding", 0.0)),
            rating=state.get("rating", "BBB"),
            tax_loss_carryforward=float(state.get("tax_loss_carryforward", 0.0)),
            autopilot=dict(state.get("autopilot") or _default_autopilot()),
            treasury_enabled=bool(state.get("treasury_enabled", False)),
            equity_negative_months=int(state.get("equity_negative_months", 0)),
        )


class FirmRegistry:
    """Stable firm ids; iteration is always sorted by id (design rule 10)."""

    def __init__(
        self,
        firms: Iterable[Firm] = (),
        *,
        cells: Iterable[tuple[str, str]],
    ) -> None:
        self._cells = frozenset((str(r), str(s)) for r, s in cells)
        by_id: dict[str, Firm] = {}
        for firm in firms:
            if firm.id in by_id:
                raise ConfigError(f"duplicate firm id {firm.id!r}")
            self._check_cells(firm)
            by_id[firm.id] = firm
        self._ids = tuple(sorted(by_id))
        self._firms = {i: by_id[i] for i in self._ids}

    def _check_cells(self, firm: Firm) -> None:
        for plant in firm.plants:
            if plant.cell not in self._cells:
                raise ConfigError(f"unknown cell {plant.region}/{plant.sector}")

    def add(self, firm: Firm) -> None:
        """Register a new firm. Ids are immutable once assigned."""
        if firm.id in self._firms:
            raise ConfigError(f"duplicate firm id {firm.id!r}")
        self._check_cells(firm)
        self._firms[firm.id] = firm
        self._ids = tuple(sorted(self._firms))

    def __contains__(self, firm_id: object) -> bool:
        return isinstance(firm_id, str) and firm_id in self._firms

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[Firm]:
        for i in self._ids:
            yield self._firms[i]

    def __getitem__(self, firm_id: str) -> Firm:
        return self._firms[firm_id]

    @property
    def ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def cells(self) -> frozenset[tuple[str, str]]:
        return self._cells

    def to_state(self) -> dict[str, Any]:
        return {
            "cells": [list(c) for c in sorted(self._cells)],
            "firms": [firm.to_state() for firm in self],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FirmRegistry:
        cells = [(str(r), str(s)) for r, s in state["cells"]]
        firms = [Firm.from_state(row) for row in state.get("firms", ())]
        return cls(firms, cells=cells)
