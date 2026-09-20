"""Versioned (v1) JSON schemas for observe(), submit, and news (T7.01).

Wire types are closed pydantic v2 models. Extra keys are rejected so hidden
state (ξ, arb capital, unpublished vintages, …) cannot enter ``Observation``.
Cell maps are JSON arrays of ``{region, sector, value}`` — tuple keys are not
JSON-safe. A schema change requires a version bump and a new golden file.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketsim.events.news import SEVERITY_LEVELS
from marketsim.events.news import NewsItem as EngineNewsItem
from marketsim.firms.levers import FirmDecision as EngineFirmDecision
from marketsim.market.clob import Fill as EngineFill
from marketsim.market.clob import Order as EngineOrder
from marketsim.market.clob import OrderType, Side, SubmitResult, TimeInForce

API_SCHEMA_VERSION = "v1.1"

# Public observe() keys — same set as ``scenarios.randomise.OBSERVE_PUBLIC_KEYS``.
# Never add a T6.22 hidden key here.
OBSERVATION_FIELDS: frozenset[str] = frozenset(
    {
        "tick",
        "agent_id",
        "releases",
        "news",
        "fills",
        "orders",
        "portfolio",
        "quotes",
        "markets",
        "prices",
        "reports",
        "firm_decisions",
        "last_decision",
        "goods",
        "labour",
        "regions",
        "bonds",
    }
)

SideName = Literal["buy", "sell"]
OrderTypeName = Literal["market", "limit", "stop"]
TimeInForceName = Literal["ioc", "gtc", "day"]
OrderStatusName = Literal["resting", "partial", "filled", "cancelled", "rejected", "expired"]
WorldMode = Literal["game", "professional"]
RunMode = Literal["lockstep", "realtime"]
AgentRole = Literal["admin", "agent", "observer", "policymaker"]
AuthorityName = Literal["GOVT", "CENBANK"]
ListingKind = Literal["primary", "secondary"]
CorporateKind = Literal["issue", "buyback", "dividend"]
WorldRunStatus = Literal["created", "running", "paused", "stopped"]


class ApiModel(BaseModel):
    """Frozen, closed JSON object. Extra keys are a validation error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _nonempty(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _rows_to_cells(rows: list[CellQty] | None) -> dict[tuple[str, str], float] | None:
    if rows is None:
        return None
    return {(row.region, row.sector): float(row.value) for row in rows}


def _cells_to_rows(mapping: dict[tuple[str, str], float] | None) -> list[CellQty] | None:
    if mapping is None:
        return None
    return [CellQty(region=r, sector=s, value=float(v)) for (r, s), v in sorted(mapping.items())]


class CellQty(ApiModel):
    """One (region, sector) quantity. Units are those of the parent field."""

    region: str
    sector: str
    value: float

    @field_validator("region", "sector")
    @classmethod
    def _code(cls, v: str) -> str:
        return _nonempty("cell code", v)


class NewPlant(ApiModel):
    """Greenfield / new-cell plant. ``capacity`` is real units / month."""

    region: str
    sector: str
    capacity: float

    @field_validator("region", "sector")
    @classmethod
    def _code(cls, v: str) -> str:
        return _nonempty("cell code", v)


class NewsItem(ApiModel):
    """Published headline. ``tick`` is days; ``severity_hint`` is a noisy ordinal 0–3.

    Magnitudes are never a field — they stay on the internal event spec.
    """

    id: str
    tick: int
    category: str
    headline: str
    regions: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    severity_hint: int
    is_rumour: bool = False

    @field_validator("id", "category", "headline")
    @classmethod
    def _text(cls, v: str) -> str:
        return _nonempty("news text", v)

    @field_validator("tick")
    @classmethod
    def _tick(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tick must be >= 0 (days)")
        return v

    @field_validator("severity_hint")
    @classmethod
    def _sev(cls, v: int) -> int:
        if v not in SEVERITY_LEVELS:
            raise ValueError(f"severity_hint must be in {SEVERITY_LEVELS}")
        return v

    def to_engine(self) -> EngineNewsItem:
        """Map to the T4.06 dataclass (regions/sectors become tuples)."""
        return EngineNewsItem(
            id=self.id,
            tick=self.tick,
            category=self.category,
            headline=self.headline,
            regions=tuple(self.regions),
            sectors=tuple(self.sectors),
            severity_hint=self.severity_hint,
            is_rumour=self.is_rumour,
        )

    @classmethod
    def from_engine(cls, item: EngineNewsItem) -> NewsItem:
        return cls(
            id=item.id,
            tick=item.tick,
            category=item.category,
            headline=item.headline,
            regions=list(item.regions),
            sectors=list(item.sectors),
            severity_hint=item.severity_hint,
            is_rumour=item.is_rumour,
        )


class Fill(ApiModel):
    """One match. ``price`` is cr / share; ``qty`` is shares; ``notional`` is cr."""

    symbol: str
    price: float
    qty: int
    maker: str
    taker: str
    maker_order_id: int
    taker_order_id: int
    notional: float
    tick: int

    @field_validator("symbol", "maker", "taker")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("fill id", v)

    def to_engine(self) -> EngineFill:
        return EngineFill(
            symbol=self.symbol,
            price=self.price,
            qty=self.qty,
            maker=self.maker,
            taker=self.taker,
            maker_order_id=self.maker_order_id,
            taker_order_id=self.taker_order_id,
            notional=self.notional,
            tick=self.tick,
        )

    @classmethod
    def from_engine(cls, fill: EngineFill) -> Fill:
        return cls(
            symbol=fill.symbol,
            price=fill.price,
            qty=fill.qty,
            maker=fill.maker,
            taker=fill.taker,
            maker_order_id=fill.maker_order_id,
            taker_order_id=fill.taker_order_id,
            notional=fill.notional,
            tick=fill.tick,
        )


class OrderRequest(ApiModel):
    """Submit one order. ``qty`` is shares; ``price`` / ``stop_price`` are cr / share."""

    agent_id: str
    side: SideName
    qty: int
    symbol: str
    order_type: OrderTypeName = "limit"
    tif: TimeInForceName = "gtc"
    price: float | None = None
    stop_price: float | None = None
    sequence: int | None = None
    idempotency_key: str | None = None

    @field_validator("agent_id", "symbol")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("order id", v)

    @field_validator("qty")
    @classmethod
    def _qty(cls, v: int) -> int:
        if v < 1:
            raise ValueError("qty must be >= 1 (shares)")
        return v

    @model_validator(mode="after")
    def _type_fields(self) -> OrderRequest:
        if self.order_type == "limit" and self.price is None:
            raise ValueError("limit order requires price (cr / share)")
        if self.order_type == "stop" and self.stop_price is None:
            raise ValueError("stop order requires stop_price (cr / share)")
        return self

    def to_engine(self) -> EngineOrder:
        return EngineOrder(
            agent_id=self.agent_id,
            side=Side(self.side),
            qty=self.qty,
            symbol=self.symbol,
            order_type=OrderType(self.order_type),
            tif=TimeInForce(self.tif),
            price=self.price,
            stop_price=self.stop_price,
            sequence=self.sequence,
        )

    @classmethod
    def from_engine(cls, order: EngineOrder, *, idempotency_key: str | None = None) -> OrderRequest:
        return cls(
            agent_id=order.agent_id,
            side=order.side.value,
            qty=order.qty,
            symbol=order.symbol,
            order_type=order.order_type.value,
            tif=order.tif.value,
            price=order.price,
            stop_price=order.stop_price,
            sequence=order.sequence,
            idempotency_key=idempotency_key,
        )


class OrderAck(ApiModel):
    """Accept / reject. ``remaining`` is unfilled shares (0 if dead)."""

    order_id: int | None = None
    status: OrderStatusName
    remaining: int
    fills: list[Fill] = Field(default_factory=list)
    reason: str | None = None
    idempotency_key: str | None = None

    @classmethod
    def from_engine(cls, result: SubmitResult, *, idempotency_key: str | None = None) -> OrderAck:
        return cls(
            order_id=result.order_id,
            status=result.status.value,
            remaining=result.remaining,
            fills=[Fill.from_engine(fill) for fill in result.fills],
            reason=result.reason,
            idempotency_key=idempotency_key,
        )


class OpenOrder(ApiModel):
    """Own live order as published on observe(). ``qty`` / ``remaining`` are shares."""

    order_id: int
    symbol: str
    side: SideName
    qty: int
    remaining: int
    status: OrderStatusName
    price: float | None = None
    order_type: OrderTypeName = "limit"


class FirmDecision(ApiModel):
    """Operator levers. Omitted fields stay on autopilot (T5.06 / §5.4)."""

    firm_id: str
    operator: str
    posted_price: list[CellQty] | None = None  # index
    target_output: list[CellQty] | None = None  # units / month
    vacancies: float | None = None  # persons
    wage_offer: float | None = None  # cr / person / month
    layoffs: float | None = None  # persons
    input_cover_m: float | None = None  # months
    order_mult: float | None = None  # dimensionless
    shortage_premium: float | None = None  # 0–0.5
    expand_capacity: list[CellQty] | None = None  # units / year
    new_plant: NewPlant | None = None
    rnd_spend: float | None = None  # cr / month
    borrow: float | None = None  # cr
    repay: float | None = None  # cr
    dividend: float | None = None  # cr
    treasury_enabled: bool | None = None
    liquidate: bool | None = None

    @field_validator("firm_id", "operator")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("firm id", v)

    @model_validator(mode="after")
    def _unique_cells(self) -> FirmDecision:
        for name in ("posted_price", "target_output", "expand_capacity"):
            rows: list[CellQty] | None = getattr(self, name)
            if rows is None:
                continue
            seen: set[tuple[str, str]] = set()
            for row in rows:
                key = (row.region, row.sector)
                if key in seen:
                    raise ValueError(f"duplicate cell {key} in {name}")
                seen.add(key)
        return self

    def to_engine(self) -> EngineFirmDecision:
        plant = None
        if self.new_plant is not None:
            plant = (self.new_plant.region, self.new_plant.sector, float(self.new_plant.capacity))
        return EngineFirmDecision(
            firm_id=self.firm_id,
            operator=self.operator,
            posted_price=_rows_to_cells(self.posted_price),
            target_output=_rows_to_cells(self.target_output),
            vacancies=self.vacancies,
            wage_offer=self.wage_offer,
            layoffs=self.layoffs,
            input_cover_m=self.input_cover_m,
            order_mult=self.order_mult,
            shortage_premium=self.shortage_premium,
            expand_capacity=_rows_to_cells(self.expand_capacity),
            new_plant=plant,
            rnd_spend=self.rnd_spend,
            borrow=self.borrow,
            repay=self.repay,
            dividend=self.dividend,
            treasury_enabled=self.treasury_enabled,
            liquidate=self.liquidate,
        )

    @classmethod
    def from_engine(cls, decision: EngineFirmDecision) -> FirmDecision:
        plant = None
        if decision.new_plant is not None:
            region, sector, capacity = decision.new_plant
            plant = NewPlant(region=region, sector=sector, capacity=float(capacity))
        return cls(
            firm_id=decision.firm_id,
            operator=decision.operator,
            posted_price=_cells_to_rows(decision.posted_price),
            target_output=_cells_to_rows(decision.target_output),
            vacancies=decision.vacancies,
            wage_offer=decision.wage_offer,
            layoffs=decision.layoffs,
            input_cover_m=decision.input_cover_m,
            order_mult=decision.order_mult,
            shortage_premium=decision.shortage_premium,
            expand_capacity=_cells_to_rows(decision.expand_capacity),
            new_plant=plant,
            rnd_spend=decision.rnd_spend,
            borrow=decision.borrow,
            repay=decision.repay,
            dividend=decision.dividend,
            treasury_enabled=decision.treasury_enabled,
            liquidate=decision.liquidate,
        )


class FoundFirmRequest(ApiModel):
    """POST /firms. ``capital`` is cr; optional plant ``capacity`` is units / month."""

    firm_id: str
    operator: str
    capital: float
    source: str | None = None
    region: str | None = None
    sector: str | None = None
    capacity: float | None = None
    greenfield: bool = False

    @field_validator("firm_id", "operator")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("firm id", v)

    @field_validator("capital")
    @classmethod
    def _capital(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("capital must be > 0 (cr)")
        return v


class ListingRequest(ApiModel):
    """POST /firms/{fid}/listing. ``shares`` are units; ``reserve_price`` is cr / share."""

    firm_id: str
    kind: ListingKind = "primary"
    shares: int
    reserve_price: float

    @field_validator("firm_id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("firm_id", v)

    @field_validator("shares")
    @classmethod
    def _shares(cls, v: int) -> int:
        if v < 1:
            raise ValueError("shares must be >= 1")
        return v

    @field_validator("reserve_price")
    @classmethod
    def _px(cls, v: float) -> float:
        if v < 0:
            raise ValueError("reserve_price must be >= 0 (cr / share)")
        return v


class CorporateAction(ApiModel):
    """Issue, buyback, or dividend. ``shares`` are units; ``amount`` is cr."""

    firm_id: str
    kind: CorporateKind
    shares: int | None = None
    amount: float | None = None

    @field_validator("firm_id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("firm_id", v)

    @model_validator(mode="after")
    def _payload(self) -> CorporateAction:
        if self.kind in ("issue", "buyback"):
            if self.shares is None or self.shares < 1:
                raise ValueError(f"{self.kind} requires shares >= 1")
        if self.kind == "dividend":
            if self.amount is None or self.amount < 0:
                raise ValueError("dividend requires amount >= 0 (cr)")
        return self


class CompetitorPrice(ApiModel):
    """One seller's posted price. ``posted_price`` is an index (1.0 at baseline)."""

    seller_id: str
    region: str
    sector: str
    posted_price: float

    @field_validator("seller_id", "region", "sector")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("goods id", v)


class GoodsView(ApiModel):
    """GET /goods. Prices are indices; ``demand`` is units / month."""

    reference_prices: list[CellQty] = Field(default_factory=list)
    demand: list[CellQty] = Field(default_factory=list)
    competitor_prices: list[CompetitorPrice] = Field(default_factory=list)


class LabourView(ApiModel):
    """GET /labour/{region}. Headcount is persons; ``wage_bar`` is cr / person / month."""

    region: str
    labour_force: float = 0.0
    unemployed: float = 0.0
    wage_bar: float = 0.0
    vacancies: float = 0.0

    @field_validator("region")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("region", v)


class BondHoldings(ApiModel):
    """Published bond book. Face amounts are cr."""

    own: dict[str, float] = Field(default_factory=dict)
    published: dict[str, dict[str, float]] = Field(default_factory=dict)


class BondCashEvent(ApiModel):
    """One coupon + redemption. Money fields are cr; ``tick`` is days."""

    tick: int
    agent_id: str
    instrument: str
    coupon: float
    redemption: float
    face_next: float

    @field_validator("agent_id", "instrument")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("bond cashflow id", v)

    @field_validator("tick")
    @classmethod
    def _tick(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tick must be >= 0 (days)")
        return v


class BondsView(ApiModel):
    """GET …/bonds. Yields are annual decimals; prices are indices; face is cr."""

    curve: dict[str, float] = Field(default_factory=dict)
    bucket_prices: dict[str, float] = Field(default_factory=dict)
    outstanding: dict[str, float] = Field(default_factory=dict)
    holdings: BondHoldings = Field(default_factory=BondHoldings)
    cashflows: list[BondCashEvent] = Field(default_factory=list)


class BondAuctionSlot(ApiModel):
    """One DMO slot. ``size`` is face (cr); ``y_fair`` is an annual decimal; ticks are days."""

    auction_id: str
    month: int
    instrument: str
    announce_tick: int
    auction_tick: int
    size: float
    y_fair: float
    status: Literal["announced", "open", "cleared"] = "open"

    @field_validator("auction_id", "instrument")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("auction slot id", v)

    @field_validator("month", "announce_tick", "auction_tick")
    @classmethod
    def _nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("auction month and ticks must be >= 0")
        return v


class BondAuctionResultView(ApiModel):
    """Uniform-price outcome. Yields are annual decimals; qtys are face (cr)."""

    auction_id: str
    stop_out: float
    size: float
    filled: float
    npc_fill: float
    agent_fill: dict[str, float] = Field(default_factory=dict)
    bid_to_cover: float
    tail: float
    winners: list[str] = Field(default_factory=list)

    @field_validator("auction_id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("auction id", v)


class BondAuctionsView(ApiModel):
    """GET …/bonds/auctions. Calendar ticks are days; sizes are face (cr)."""

    calendar: list[BondAuctionSlot] = Field(default_factory=list)
    sizes: dict[str, float] = Field(default_factory=dict)
    results: list[BondAuctionResultView] = Field(default_factory=list)


class BondBidRequest(ApiModel):
    """POST …/bonds/auctions/{aid}/bids. ``qty`` is face (cr); ``yield_annual`` is an annual decimal."""

    yield_annual: float
    qty: float
    agent_id: str | None = None

    @field_validator("qty")
    @classmethod
    def _qty(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("qty must be > 0 (cr face)")
        return v


class BondBidAck(ApiModel):
    """Accepted competitive bid. ``qty`` is face (cr); ``yield_annual`` is an annual decimal."""

    auction_id: str
    agent_id: str
    yield_annual: float
    qty: float
    accepted: bool = True

    @field_validator("auction_id", "agent_id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("bond bid id", v)


class PortfolioPosition(ApiModel):
    """One holding. ``qty`` is shares or face; ``market_value`` is cr."""

    instrument: str
    qty: float
    market_value: float = 0.0

    @field_validator("instrument")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("instrument", v)


class PortfolioView(ApiModel):
    """Own book. Cash and ``net_worth`` are cr."""

    cash: float = 0.0
    positions: list[PortfolioPosition] = Field(default_factory=list)
    net_worth: float = 0.0


class Report(ApiModel):
    """Lagged (or live, if operator) firm books. ``tick`` is the publication day."""

    firm_id: str
    live: bool
    books: dict[str, Any] | None = None
    tick: int | None = None

    @field_validator("firm_id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("firm_id", v)


class ErrorModel(ApiModel):
    """API error body: ``{code, message, details}``."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("code", "message")
    @classmethod
    def _text(cls, v: str) -> str:
        return _nonempty("error text", v)


class WorldSpec(ApiModel):
    """POST /v1/worlds. ``overrides`` are dotted config paths (T0.03)."""

    seed: int
    scenario: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    mode: WorldMode = "professional"
    run_mode: RunMode = "lockstep"

    @field_validator("seed")
    @classmethod
    def _seed(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be >= 0")
        return v


class WorldStatus(ApiModel):
    """GET /v1/worlds/{wid}. ``tick`` is simulated days."""

    world_id: str
    tick: int
    seed: int
    mode: WorldMode
    run_mode: RunMode
    state_hash: str
    n_agents: int = 0
    status: WorldRunStatus = "created"

    @field_validator("world_id", "state_hash")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("world status id", v)

    @field_validator("tick", "n_agents")
    @classmethod
    def _nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tick and n_agents must be >= 0")
        return v


class AgentRegistration(ApiModel):
    """POST /agents. ``starting_capital`` is cr; policymaker binds one authority."""

    agent_id: str
    role: AgentRole = "agent"
    starting_capital: float = 0.0
    authority: AuthorityName | None = None

    @field_validator("agent_id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("agent_id", v)

    @field_validator("starting_capital")
    @classmethod
    def _cash(cls, v: float) -> float:
        if v < 0:
            raise ValueError("starting_capital must be >= 0 (cr)")
        return v


class Observation(ApiModel):
    """GET /observe. Published fields only — no ξ, arb capital, or unpublished vintages."""

    tick: int
    agent_id: str
    releases: dict[str, Any] = Field(default_factory=dict)
    news: list[NewsItem] = Field(default_factory=list)
    fills: list[Fill] = Field(default_factory=list)
    orders: list[OpenOrder] = Field(default_factory=list)
    portfolio: PortfolioView = Field(default_factory=PortfolioView)
    quotes: dict[str, float] = Field(default_factory=dict)
    markets: dict[str, Any] = Field(default_factory=dict)
    prices: dict[str, float] = Field(default_factory=dict)
    reports: list[Report] = Field(default_factory=list)
    firm_decisions: int | None = None
    last_decision: str | None = None
    goods: GoodsView = Field(default_factory=GoodsView)
    labour: list[LabourView] = Field(default_factory=list)
    regions: dict[str, Any] = Field(default_factory=dict)
    bonds: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _nonempty("agent_id", v)

    @field_validator("tick")
    @classmethod
    def _tick(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tick must be >= 0 (days)")
        return v


class SubmitPayload(ApiModel):
    """One agent's submit body. Empty lists are a valid lockstep no-op."""

    decisions: list[FirmDecision] = Field(default_factory=list)
    orders: list[OrderRequest] = Field(default_factory=list)
    client_seq: int | None = None


CARD_MODELS: tuple[type[ApiModel], ...] = (
    WorldSpec,
    WorldStatus,
    AgentRegistration,
    Observation,
    OrderRequest,
    OrderAck,
    Fill,
    FirmDecision,
    FoundFirmRequest,
    ListingRequest,
    CorporateAction,
    GoodsView,
    LabourView,
    NewsItem,
    Report,
    ErrorModel,
    BondsView,
    BondAuctionsView,
    BondBidRequest,
    BondBidAck,
)

SUPPORTING_MODELS: tuple[type[ApiModel], ...] = (
    CellQty,
    NewPlant,
    CompetitorPrice,
    OpenOrder,
    PortfolioPosition,
    PortfolioView,
    SubmitPayload,
    BondHoldings,
    BondCashEvent,
    BondAuctionSlot,
    BondAuctionResultView,
)

V1_MODELS: tuple[type[ApiModel], ...] = CARD_MODELS + SUPPORTING_MODELS


def observation_field_names() -> frozenset[str]:
    """Top-level ``Observation`` keys (units: see field docstrings)."""
    return frozenset(Observation.model_fields)


def api_json_schema() -> dict[str, Any]:
    """Frozen v1 JSON Schema document. Changing it requires a version bump."""
    models = {cls.__name__: cls.model_json_schema() for cls in V1_MODELS}
    return {"title": "marketsim API", "version": API_SCHEMA_VERSION, "models": models}


def dump_api_schema() -> str:
    """Canonical JSON text (sorted keys, trailing newline)."""
    return json.dumps(api_json_schema(), indent=2, sort_keys=True) + "\n"
