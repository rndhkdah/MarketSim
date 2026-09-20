"""REST surface for worlds, agents, observe, orders, firms, and policy (T7.04 / §7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, FastAPI, Header, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field, field_validator

from marketsim.api.schemas import (
    OBSERVATION_FIELDS,
    AgentRegistration,
    ApiModel,
    ErrorModel,
    Fill,
    FirmDecision,
    FoundFirmRequest,
    GoodsView,
    LabourView,
    Observation,
    OpenOrder,
    OrderAck,
    OrderRequest,
    Report,
    WorldSpec,
    WorldStatus,
)
from marketsim.api.sessions import AuthError, Session, WorldManager, issue_token
from marketsim.core.errors import ConfigError, MarketsimError
from marketsim.firms.accounts import agent_entity
from marketsim.scenarios.randomise import sanitise_observe

# Page size (unitless count). T7.04 cannot add yaml / config.py keys.
_DEFAULT_PAGE_LIMIT = 50
_MAX_PAGE_LIMIT = 200

_ADMIN_ID = "admin"


class ApiError(MarketsimError):
    """HTTP error with an ``ErrorModel`` body. ``status_code`` is the HTTP status."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.body = ErrorModel(code=code, message=message, details=details or {})


class StepRequest(ApiModel):
    """POST …/step. ``n`` is simulated days."""

    n: int = 1

    @field_validator("n")
    @classmethod
    def _n(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n must be >= 1 (ticks / days)")
        return v


class AgentLimitsView(ApiModel):
    """Per-agent caps (§7.2). Units: counts (per tick or resting orders)."""

    orders_per_tick: int
    decisions_per_tick: int
    max_open_orders: int


class AgentAccount(ApiModel):
    """POST …/agents. ``account`` is the ledger entity ``AGENT:<id>``."""

    token: str
    account: str
    limits: AgentLimitsView


class Page(ApiModel):
    """Cursor page. ``limit`` is a count; ``next_cursor`` is an opaque offset."""

    items: list[Any] = Field(default_factory=list)
    next_cursor: str | None = None
    limit: int = _DEFAULT_PAGE_LIMIT


class RegionRow(ApiModel):
    """One region from ``config/regions.yaml``. Shares are dimensionless; wage is an index."""

    code: str
    population_share: float
    wage_level: float


class RegionsView(ApiModel):
    """GET …/regions."""

    regions: list[RegionRow] = Field(default_factory=list)


class BondsView(ApiModel):
    """GET …/bonds. Stub until T7.17; prices are indices, face is cr."""

    curve: dict[str, Any] = Field(default_factory=dict)
    bucket_prices: dict[str, float] = Field(default_factory=dict)
    outstanding: dict[str, float] = Field(default_factory=dict)
    holdings: dict[str, Any] = Field(default_factory=dict)


class BondAuctionsView(ApiModel):
    """GET …/bonds/auctions. Stub until T7.17."""

    calendar: list[Any] = Field(default_factory=list)
    sizes: dict[str, Any] = Field(default_factory=dict)
    results: list[Any] = Field(default_factory=list)


class FirmStateView(ApiModel):
    """GET …/firms/{fid}/state (operator). ``live`` is a flag; books may be empty."""

    firm_id: str
    operator: str
    live: bool = True
    books: dict[str, Any] = Field(default_factory=dict)


class FirmFinancialsView(ApiModel):
    """GET …/firms/{fid}/financials. Money fields are cr."""

    firm_id: str
    cash: float = 0.0
    credit_drawn: float = 0.0
    term_debt: float = 0.0


class GovernmentState(ApiModel):
    """GET …/government/state. Stub until T7.16; money fields would be cr."""

    budget: dict[str, Any] = Field(default_factory=dict)
    debt: dict[str, Any] = Field(default_factory=dict)
    issuance_plan: dict[str, Any] = Field(default_factory=dict)


class CenbankState(ApiModel):
    """GET …/cenbank/state. Stub until T7.16. ``rate`` is an annual decimal."""

    rate: float | None = None
    pi_star: float | None = None
    effective: dict[str, Any] = Field(default_factory=dict)


class PolicyDecisionRequest(ApiModel):
    """POST policy decisions. Any subset of §2.13 levers; extra keys are a 422."""

    purchases_level: float | None = None
    purchases_mix: dict[str, float] | None = None
    tau_y: float | None = None
    tau_c: float | None = None
    vat: float | None = None
    tariff: float | None = None
    excise: dict[str, float] | None = None
    benefit_replacement: float | None = None
    transfer_oneoff: float | None = None
    capex_subsidy: dict[str, float] | None = None
    rescue_banksys: float | None = None
    debt_target: float | None = None
    kappa_debt: float | None = None
    fiscal_rule_on: bool | None = None
    rate: float | None = None
    pi_star: float | None = None
    phi_pi: float | None = None
    phi_y: float | None = None
    smoothing: float | None = None
    capital_requirement: float | None = None
    ltv_cap: float | None = None
    lolr: float | None = None
    guidance_path: list[float] | None = None


class PolicyDecisionAck(ApiModel):
    """Echo of accepted levers. Values keep the units of each lever."""

    authority: Literal["GOVT", "CENBANK"]
    levers: dict[str, Any] = Field(default_factory=dict)


@dataclass
class _FirmRec:
    firm_id: str
    operator: str
    capital: float  # cr


@dataclass
class _WorldRest:
    admin_token: str
    next_order_id: int = 1
    orders: dict[int, OpenOrder] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    idempotency: dict[str, OrderAck] = field(default_factory=dict)
    firms: dict[str, _FirmRec] = field(default_factory=dict)


@dataclass
class RestState:
    """Process-local REST bookkeeping. ``config_dir`` is a filesystem path."""

    manager: WorldManager
    config_dir: Path
    worlds: dict[str, _WorldRest] = field(default_factory=dict)


def world_admin_token(world_id: str, seed: int) -> str:
    """Deterministic bootstrap admin bearer for ``world_id``. Unitless hex digest."""
    return issue_token(world_id, _ADMIN_ID, int(seed), "admin")


def _default_config_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "config"


def _error_json(status_code: int, body: ErrorModel) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _raise(status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
    raise ApiError(status_code, code, message, details)


def _bearer(authorization: str | None) -> str:
    if authorization is None or not str(authorization).strip():
        _raise(401, "unauthorized", "missing bearer token")
    scheme, _, token = str(authorization).partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        _raise(401, "unauthorized", "invalid authorization header")
    return token.strip()


def _rest_state(request: Request) -> RestState:
    return request.app.state.api


def _world_rest(state: RestState, world_id: str) -> _WorldRest:
    rec = state.worlds.get(world_id)
    if rec is None or world_id not in state.manager._worlds:
        _raise(404, "not_found", f"unknown world {world_id!r}")
    return rec


def _session(state: RestState, world_id: str, authorization: str | None) -> Session:
    token = _bearer(authorization)
    rec = _world_rest(state, world_id)
    if token == rec.admin_token:
        return Session(
            world_id=world_id,
            agent_id=_ADMIN_ID,
            role="admin",
            entity=agent_entity(_ADMIN_ID),
            token=token,
        )
    try:
        return state.manager.authenticate(world_id, token)
    except AuthError as exc:
        _raise(401, "unauthorized", str(exc))
    raise AssertionError("unreachable")


def _require_policymaker(session: Session, authority: Literal["GOVT", "CENBANK"]) -> None:
    if session.role != "policymaker":
        _raise(403, "forbidden", "policymaker role required")
    if session.authority is not None and session.authority != authority:
        _raise(403, "forbidden", f"policymaker is bound to {session.authority}")


def _page(items: list[Any], *, cursor: str | None, limit: int) -> Page:
    if limit < 1 or limit > _MAX_PAGE_LIMIT:
        _raise(422, "validation_error", "limit out of range", {"limit": limit})
    offset = 0
    if cursor is not None and cursor != "":
        try:
            offset = int(cursor)
        except ValueError:
            _raise(422, "validation_error", "cursor must be an integer offset", {"cursor": cursor})
        if offset < 0:
            _raise(422, "validation_error", "cursor must be >= 0", {"cursor": cursor})
    chunk = items[offset : offset + limit]
    nxt = str(offset + limit) if offset + limit < len(items) else None
    return Page(items=chunk, next_cursor=nxt, limit=limit)


def _apply_since(obs: dict[str, Any], since: int) -> dict[str, Any]:
    """Drop published list items with ``tick`` ≤ ``since`` (days)."""
    out = dict(obs)
    for key in ("news", "fills", "reports"):
        rows = out.get(key)
        if not isinstance(rows, list):
            continue
        kept: list[Any] = []
        for row in rows:
            tick = row.get("tick") if isinstance(row, dict) else getattr(row, "tick", None)
            if tick is None or int(tick) > since:
                kept.append(row)
        out[key] = kept
    return out


def _observation(world: Any, agent_id: str, since: int | None) -> Observation:
    raw = sanitise_observe(world.observe(agent_id))
    if since is not None:
        raw = _apply_since(raw, since)
    payload = {key: raw[key] for key in raw if key in OBSERVATION_FIELDS}
    payload.setdefault("tick", int(world.clock.tick))
    payload.setdefault("agent_id", agent_id)
    payload.setdefault("reports", [])
    return Observation.model_validate(payload)


def _levers(body: PolicyDecisionRequest) -> dict[str, Any]:
    return {k: v for k, v in body.model_dump().items() if v is not None}


def world_router() -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["world"])

    @router.post("/worlds", response_model=WorldStatus)
    def create_world(request: Request, spec: WorldSpec) -> WorldStatus:
        state = _rest_state(request)
        try:
            status = state.manager.create_world(spec, state.config_dir)
        except ConfigError as exc:
            _raise(400, "config_error", str(exc))
        state.worlds[status.world_id] = _WorldRest(
            admin_token=world_admin_token(status.world_id, status.seed)
        )
        return status

    @router.get("/worlds/{wid}", response_model=WorldStatus)
    def get_world(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> WorldStatus:
        state = _rest_state(request)
        _session(state, wid, authorization)
        return state.manager.status(wid)

    @router.delete("/worlds/{wid}", response_model=ErrorModel, response_model_exclude_unset=True)
    def delete_world(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> JSONResponse:
        state = _rest_state(request)
        _session(state, wid, authorization)
        # WorldManager has no public delete (T7.02); this card cannot extend sessions.py.
        state.manager._worlds.pop(wid, None)
        state.worlds.pop(wid, None)
        return _error_json(200, ErrorModel(code="deleted", message=f"world {wid} deleted"))

    @router.post("/worlds/{wid}/reset", response_model=WorldStatus)
    def reset_world(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> WorldStatus:
        state = _rest_state(request)
        _session(state, wid, authorization)
        handle = state.manager._handle(wid)
        handle.world.reset()
        handle.status = "created"
        for agent_id in sorted(handle.agents):
            rec = handle.agents[agent_id]
            rec.orders_this_tick = 0
            rec.decisions_this_tick = 0
        rest = state.worlds[wid]
        rest.next_order_id = 1
        rest.orders.clear()
        rest.fills.clear()
        rest.idempotency.clear()
        return state.manager.status(wid)

    @router.post("/worlds/{wid}/step", response_model=WorldStatus)
    def step_world(
        request: Request,
        wid: str,
        body: StepRequest | None = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> WorldStatus:
        state = _rest_state(request)
        _session(state, wid, authorization)
        n = 1 if body is None else body.n
        return state.manager.step(wid, n)

    return router


def agents_router() -> APIRouter:
    router = APIRouter(prefix="/v1/worlds", tags=["agents"])

    @router.post("/{wid}/agents", response_model=AgentAccount)
    def register_agent(
        request: Request,
        wid: str,
        registration: AgentRegistration,
        authorization: Annotated[str | None, Header()] = None,
    ) -> AgentAccount:
        state = _rest_state(request)
        _session(state, wid, authorization)
        try:
            rec = state.manager.register_agent(wid, registration)
        except ConfigError as exc:
            _raise(400, "config_error", str(exc))
        limits = rec["limits"]
        return AgentAccount(
            token=str(rec["token"]),
            account=str(rec["entity"]),
            limits=AgentLimitsView(
                orders_per_tick=int(limits.orders_per_tick),
                decisions_per_tick=int(limits.decisions_per_tick),
                max_open_orders=int(limits.max_open_orders),
            ),
        )

    return router


def observe_router() -> APIRouter:
    router = APIRouter(prefix="/v1/worlds", tags=["observe"])

    @router.get("/{wid}/observe", response_model=Observation)
    def observe(
        request: Request,
        wid: str,
        since: Annotated[int | None, Query(ge=0)] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> Observation:
        state = _rest_state(request)
        session = _session(state, wid, authorization)
        return _observation(state.manager.world(wid), session.agent_id, since)

    return router


def orders_router() -> APIRouter:
    router = APIRouter(prefix="/v1/worlds", tags=["orders"])

    @router.post("/{wid}/orders", response_model=OrderAck)
    def post_order(
        request: Request,
        wid: str,
        order: OrderRequest,
        authorization: Annotated[str | None, Header()] = None,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> OrderAck:
        state = _rest_state(request)
        session = _session(state, wid, authorization)
        rest = state.worlds[wid]
        key = (idempotency_key or order.idempotency_key or "").strip() or None
        cache_key = f"{session.agent_id}:{key}" if key else None
        if cache_key is not None and cache_key in rest.idempotency:
            return rest.idempotency[cache_key]
        try:
            state.manager.submit_order(wid, session.token, order)
        except AuthError as exc:
            _raise(401, "unauthorized", str(exc))
        oid = rest.next_order_id
        rest.next_order_id += 1
        rest.orders[oid] = OpenOrder(
            order_id=oid,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            remaining=order.qty,
            status="resting",
            price=order.price,
            order_type=order.order_type,
        )
        ack = OrderAck(
            order_id=oid,
            status="resting",
            remaining=order.qty,
            fills=[],
            idempotency_key=key,
        )
        if cache_key is not None:
            rest.idempotency[cache_key] = ack
        return ack

    @router.delete("/{wid}/orders/{oid}", response_model=OrderAck)
    def cancel_order(
        request: Request,
        wid: str,
        oid: int,
        authorization: Annotated[str | None, Header()] = None,
    ) -> OrderAck:
        state = _rest_state(request)
        _session(state, wid, authorization)
        rest = state.worlds[wid]
        open_order = rest.orders.get(oid)
        if open_order is None:
            _raise(404, "not_found", f"unknown order {oid}")
        cancelled = OpenOrder(
            order_id=open_order.order_id,
            symbol=open_order.symbol,
            side=open_order.side,
            qty=open_order.qty,
            remaining=0,
            status="cancelled",
            price=open_order.price,
            order_type=open_order.order_type,
        )
        rest.orders[oid] = cancelled
        return OrderAck(order_id=oid, status="cancelled", remaining=0, fills=[])

    @router.get("/{wid}/orders", response_model=Page)
    def list_orders(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
        limit: Annotated[int, Query(ge=1, le=_MAX_PAGE_LIMIT)] = _DEFAULT_PAGE_LIMIT,
        cursor: str | None = None,
    ) -> Page:
        state = _rest_state(request)
        _session(state, wid, authorization)
        book = state.worlds[wid].orders
        items = [book[k] for k in sorted(book)]
        return _page(items, cursor=cursor, limit=limit)

    @router.get("/{wid}/fills", response_model=Page)
    def list_fills(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
        limit: Annotated[int, Query(ge=1, le=_MAX_PAGE_LIMIT)] = _DEFAULT_PAGE_LIMIT,
        cursor: str | None = None,
    ) -> Page:
        state = _rest_state(request)
        _session(state, wid, authorization)
        return _page(list(state.worlds[wid].fills), cursor=cursor, limit=limit)

    return router


def firms_router() -> APIRouter:
    router = APIRouter(prefix="/v1/worlds", tags=["firms"])

    @router.post("/{wid}/firms", response_model=FoundFirmRequest)
    def found_firm(
        request: Request,
        wid: str,
        body: FoundFirmRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> FoundFirmRequest:
        state = _rest_state(request)
        session = _session(state, wid, authorization)
        rest = state.worlds[wid]
        if body.firm_id in rest.firms:
            _raise(409, "conflict", f"firm {body.firm_id!r} already exists")
        operator = body.operator or session.agent_id
        rest.firms[body.firm_id] = _FirmRec(firm_id=body.firm_id, operator=operator, capital=body.capital)
        return body

    @router.post("/{wid}/firms/{fid}/decisions", response_model=FirmDecision)
    def firm_decisions(
        request: Request,
        wid: str,
        fid: str,
        body: FirmDecision,
        authorization: Annotated[str | None, Header()] = None,
    ) -> FirmDecision:
        state = _rest_state(request)
        session = _session(state, wid, authorization)
        firm = state.worlds[wid].firms.get(fid)
        if firm is None:
            _raise(404, "not_found", f"unknown firm {fid!r}")
        if session.agent_id != firm.operator:
            _raise(403, "forbidden", "operator role required")
        if body.firm_id != fid:
            _raise(422, "validation_error", "firm_id does not match path", {"firm_id": body.firm_id})
        try:
            state.manager.record_decision(wid, session.token, body)
        except AuthError as exc:
            _raise(401, "unauthorized", str(exc))
        return body

    @router.get("/{wid}/firms/{fid}/state", response_model=FirmStateView)
    def firm_state(
        request: Request,
        wid: str,
        fid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> FirmStateView:
        state = _rest_state(request)
        session = _session(state, wid, authorization)
        firm = state.worlds[wid].firms.get(fid)
        if firm is None:
            _raise(404, "not_found", f"unknown firm {fid!r}")
        if session.agent_id != firm.operator:
            _raise(403, "forbidden", "operator only")
        return FirmStateView(firm_id=firm.firm_id, operator=firm.operator, live=True, books={})

    @router.get("/{wid}/firms/{fid}/financials", response_model=FirmFinancialsView)
    def firm_financials(
        request: Request,
        wid: str,
        fid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> FirmFinancialsView:
        state = _rest_state(request)
        _session(state, wid, authorization)
        firm = state.worlds[wid].firms.get(fid)
        if firm is None:
            _raise(404, "not_found", f"unknown firm {fid!r}")
        return FirmFinancialsView(firm_id=firm.firm_id)

    @router.get("/{wid}/firms/{fid}/reports", response_model=Page)
    def firm_reports(
        request: Request,
        wid: str,
        fid: str,
        authorization: Annotated[str | None, Header()] = None,
        limit: Annotated[int, Query(ge=1, le=_MAX_PAGE_LIMIT)] = _DEFAULT_PAGE_LIMIT,
        cursor: str | None = None,
    ) -> Page:
        state = _rest_state(request)
        _session(state, wid, authorization)
        if fid not in state.worlds[wid].firms:
            _raise(404, "not_found", f"unknown firm {fid!r}")
        reports: list[Report] = []
        return _page(reports, cursor=cursor, limit=limit)

    return router


def goods_router() -> APIRouter:
    router = APIRouter(prefix="/v1/worlds", tags=["goods"])

    @router.get("/{wid}/goods", response_model=GoodsView)
    def goods(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> GoodsView:
        state = _rest_state(request)
        _session(state, wid, authorization)
        return GoodsView()

    @router.get("/{wid}/labour/{region}", response_model=LabourView)
    def labour(
        request: Request,
        wid: str,
        region: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> LabourView:
        state = _rest_state(request)
        _session(state, wid, authorization)
        cfg = state.manager.world(wid).cfg.regions
        known = {row.code for row in cfg.regions} if cfg is not None else set()
        if known and region not in known:
            _raise(404, "not_found", f"unknown region {region!r}")
        return LabourView(region=region)

    @router.get("/{wid}/regions", response_model=RegionsView)
    def regions(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> RegionsView:
        state = _rest_state(request)
        _session(state, wid, authorization)
        cfg = state.manager.world(wid).cfg.regions
        rows = []
        if cfg is not None:
            rows = [
                RegionRow(code=row.code, population_share=row.population_share, wage_level=row.wage_level)
                for row in cfg.regions
            ]
        return RegionsView(regions=rows)

    return router


def bonds_router() -> APIRouter:
    router = APIRouter(prefix="/v1/worlds", tags=["bonds"])

    @router.get("/{wid}/bonds", response_model=BondsView)
    def bonds(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> BondsView:
        state = _rest_state(request)
        _session(state, wid, authorization)
        return BondsView()

    @router.get("/{wid}/bonds/auctions", response_model=BondAuctionsView)
    def bond_auctions(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> BondAuctionsView:
        state = _rest_state(request)
        _session(state, wid, authorization)
        return BondAuctionsView()

    return router


def policy_router() -> APIRouter:
    router = APIRouter(prefix="/v1/worlds", tags=["policy"])

    @router.get("/{wid}/government/state", response_model=GovernmentState)
    def government_state(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> GovernmentState:
        state = _rest_state(request)
        _session(state, wid, authorization)
        return GovernmentState()

    @router.post("/{wid}/government/decisions", response_model=PolicyDecisionAck)
    def government_decisions(
        request: Request,
        wid: str,
        body: PolicyDecisionRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> PolicyDecisionAck:
        state = _rest_state(request)
        session = _session(state, wid, authorization)
        _require_policymaker(session, "GOVT")
        return PolicyDecisionAck(authority="GOVT", levers=_levers(body))

    @router.get("/{wid}/cenbank/state", response_model=CenbankState)
    def cenbank_state(
        request: Request,
        wid: str,
        authorization: Annotated[str | None, Header()] = None,
    ) -> CenbankState:
        state = _rest_state(request)
        _session(state, wid, authorization)
        return CenbankState()

    @router.post("/{wid}/cenbank/decisions", response_model=PolicyDecisionAck)
    def cenbank_decisions(
        request: Request,
        wid: str,
        body: PolicyDecisionRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> PolicyDecisionAck:
        state = _rest_state(request)
        session = _session(state, wid, authorization)
        _require_policymaker(session, "CENBANK")
        return PolicyDecisionAck(authority="CENBANK", levers=_levers(body))

    return router


def create_app(
    manager: WorldManager | None = None,
    config_dir: str | Path | None = None,
) -> FastAPI:
    """Build the v1 FastAPI app. ``config_dir`` is the YAML bundle path."""
    app = FastAPI(title="marketsim API", version="v1", openapi_url="/openapi.json")
    app.state.api = RestState(
        manager=manager or WorldManager(),
        config_dir=Path(config_dir) if config_dir is not None else _default_config_dir(),
    )
    for factory in (
        world_router,
        agents_router,
        observe_router,
        orders_router,
        firms_router,
        goods_router,
        bonds_router,
        policy_router,
    ):
        app.include_router(factory())

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return _error_json(exc.status_code, exc.body)

    @app.exception_handler(AuthError)
    async def _auth_error(_request: Request, exc: AuthError) -> JSONResponse:
        return _error_json(401, ErrorModel(code="unauthorized", message=str(exc)))

    @app.exception_handler(ConfigError)
    async def _config_error(_request: Request, exc: ConfigError) -> JSONResponse:
        message = str(exc)
        code = "not_found" if "unknown world" in message else "config_error"
        status = 404 if code == "not_found" else 400
        return _error_json(status, ErrorModel(code=code, message=message))

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_json(
            422,
            ErrorModel(
                code="validation_error",
                message="request validation failed",
                details={"errors": jsonable_encoder(exc.errors())},
            ),
        )

    return app
