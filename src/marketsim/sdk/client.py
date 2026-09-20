"""HTTP / WebSocket SDK wrapping the v1 REST surface (T7.07 / §7.2).

``HttpClient`` (sync) and ``AsyncHttpClient`` expose the ``LOCAL_METHODS`` names
plus REST helpers. Bearer token. Stream resume uses ``since`` = last received
``seq`` (unitless) so replay has no gaps or duplicates. Tick units are simulated
days; cash fields are cr.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, Self, TypeVar

import httpx
from pydantic import BaseModel

from marketsim.api.rest import AgentAccount, world_admin_token
from marketsim.api.schemas import (
    AgentRegistration,
    ErrorModel,
    Observation,
    OpenOrder,
    OrderAck,
    OrderRequest,
    SubmitPayload,
    WorldSpec,
    WorldStatus,
)
from marketsim.api.schemas import FirmDecision as ApiFirmDecision
from marketsim.api.ws import FIRST_EVENT_SEQ
from marketsim.core.errors import MarketsimError, StateError
from marketsim.firms.levers import FirmDecision as EngineFirmDecision

# Starlette TestClient / httpx ASGI default (unitless URL).
_DEFAULT_BASE_URL = "http://testserver"

TModel = TypeVar("TModel", bound=BaseModel)


class SdkError(MarketsimError):
    """HTTP error from the v1 API. ``status_code`` is the HTTP status."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.details = details or {}


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude_none=True)


def _error(status_code: int, payload: Any) -> SdkError:
    if isinstance(payload, dict) and "code" in payload and "message" in payload:
        body = ErrorModel.model_validate(payload)
        return SdkError(status_code, body.code, body.message, dict(body.details))
    return SdkError(status_code, "http_error", f"HTTP {status_code}", {})


def _payload(resp: httpx.Response) -> Any:
    data = resp.json() if resp.content else {}
    if resp.is_error:
        raise _error(resp.status_code, data)
    return data


def _parse(resp: httpx.Response, model: type[TModel]) -> TModel:
    return model.model_validate(_payload(resp))


def _order_batch(orders: Sequence[OrderRequest] | OrderRequest) -> tuple[OrderRequest, ...]:
    return (orders,) if isinstance(orders, OrderRequest) else tuple(orders)


def _decision_batch(
    decisions: Sequence[ApiFirmDecision | EngineFirmDecision] | ApiFirmDecision | EngineFirmDecision,
) -> tuple[ApiFirmDecision, ...]:
    if isinstance(decisions, EngineFirmDecision):
        return (ApiFirmDecision.from_engine(decisions),)
    if isinstance(decisions, ApiFirmDecision):
        return (decisions,)
    out: list[ApiFirmDecision] = []
    for item in decisions:
        if isinstance(item, EngineFirmDecision):
            out.append(ApiFirmDecision.from_engine(item))
        else:
            out.append(item)
    return tuple(out)


def _flatten_actions(actions: Any) -> tuple[list[ApiFirmDecision], list[OrderRequest]]:
    if isinstance(actions, SubmitPayload):
        return list(_decision_batch(actions.decisions)), list(_order_batch(actions.orders))
    if isinstance(actions, (list, tuple)):
        decisions: list[ApiFirmDecision] = []
        orders: list[OrderRequest] = []
        for item in actions:
            more_d, more_o = _flatten_actions(item)
            decisions.extend(more_d)
            orders.extend(more_o)
        return decisions, orders
    if isinstance(actions, OrderRequest):
        return [], [actions]
    if isinstance(actions, EngineFirmDecision):
        return [ApiFirmDecision.from_engine(actions)], []
    if isinstance(actions, ApiFirmDecision):
        return [actions], []
    raise StateError(f"unsupported action type {type(actions).__name__}")


class _BoundClient:
    """Shared world / bearer binding. ``world_id`` is unitless; token is a SHA-256 hex digest."""

    def __init__(self, *, token: str | None = None, world_id: str | None = None) -> None:
        self.token = token
        self.world_id = world_id

    def bind(self, *, world_id: str | None = None, token: str | None = None) -> Self:
        """Set the active world and/or bearer. Returns ``self``."""
        if world_id is not None:
            self.world_id = world_id
        if token is not None:
            self.token = token
        return self

    def _require_world(self) -> str:
        if self.world_id is None or self.world_id == "":
            raise StateError("world_id is required")
        return self.world_id

    def _headers(
        self,
        extra: Mapping[str, str] | None = None,
        *,
        token: str | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        bearer = self.token if token is None else token
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        if extra:
            headers.update(extra)
        return headers

    def _world_url(self, suffix: str = "") -> str:
        return f"/v1/worlds/{self._require_world()}{suffix}"

    def _adopt_created(self, status: WorldStatus) -> WorldStatus:
        self.world_id = status.world_id
        self.token = world_admin_token(status.world_id, status.seed)
        return status


class StreamSubscription:
    """One ``/stream`` connection. ``last_seq`` is the last received event seq (unitless)."""

    def __init__(self, opener: Callable[[int], AbstractContextManager[Any]], *, since: int = 0) -> None:
        self._opener = opener
        self.last_seq = int(since)
        self._cm: AbstractContextManager[Any] | None = None
        self._ws: Any = None
        self._pending: list[dict[str, Any]] = []
        self.open()

    def open(self) -> None:
        """Connect with ``since=last_seq``. The first control frame is ``subscribed``."""
        self.close()
        self._cm = self._opener(self.last_seq)
        self._ws = self._cm.__enter__()
        hello = self._ws.receive_json()
        if hello.get("type") != "subscribed":
            self._note(hello)
            self._pending.append(hello)

    def reconnect(self) -> None:
        """Resume using ``since=last_seq`` so replay has no gaps or duplicates."""
        self.open()

    def recv(self) -> dict[str, Any]:
        """Next JSON envelope. ``tick`` is days; ``seq`` is unitless."""
        if self._pending:
            msg = self._pending.pop(0)
        else:
            if self._ws is None:
                raise StateError("stream is closed")
            msg = self._ws.receive_json()
        self._note(msg)
        return msg

    def close(self) -> None:
        """Drop the socket. ``last_seq`` is kept for reconnect."""
        if self._cm is None:
            return
        try:
            self._cm.__exit__(None, None, None)
        finally:
            self._cm = None
            self._ws = None
            self._pending.clear()

    def __enter__(self) -> StreamSubscription:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _note(self, msg: Mapping[str, Any]) -> None:
        if msg.get("type") == "subscribed":
            return
        seq = int(msg.get("seq", 0))
        if seq >= FIRST_EVENT_SEQ:
            self.last_seq = seq


class HttpClient(_BoundClient):
    """Sync v1 client. Wraps httpx or a Starlette ``TestClient`` (live ASGI server)."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        *,
        token: str | None = None,
        world_id: str | None = None,
        client: httpx.Client | None = None,
        app: Any | None = None,
    ) -> None:
        super().__init__(token=token, world_id=world_id)
        if client is not None:
            self._http = client
            self._owns = False
        elif app is not None:
            from fastapi.testclient import TestClient

            self._http = TestClient(app, base_url=base_url)
            self._owns = True
        else:
            self._http = httpx.Client(base_url=base_url)
            self._owns = True

    def close(self) -> None:
        if self._owns:
            self._http.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def create_world(self, spec: WorldSpec) -> WorldStatus:
        """POST /v1/worlds. Binds ``world_id`` and the bootstrap admin bearer."""
        resp = self._http.post("/v1/worlds", json=_dump(spec))
        return self._adopt_created(_parse(resp, WorldStatus))

    def register_agent(self, registration: AgentRegistration) -> AgentAccount:
        """POST …/agents. ``starting_capital`` is cr; token is a SHA-256 hex digest."""
        resp = self._http.post(self._world_url("/agents"), json=_dump(registration), headers=self._headers())
        return _parse(resp, AgentAccount)

    def observe(self, agent_id: str | None = None, *, since: int | None = None) -> Observation:
        """GET …/observe. Agent is the bearer; ``since`` is a tick (days)."""
        del agent_id
        params: dict[str, int] = {}
        if since is not None:
            params["since"] = since
        resp = self._http.get(self._world_url("/observe"), params=params or None, headers=self._headers())
        return _parse(resp, Observation)

    def submit(self, agent_id: str, actions: Any) -> None:
        """POST orders and/or firm decisions for ``agent_id`` (token-bound)."""
        decisions, orders = _flatten_actions(actions)
        if decisions:
            self.submit_decisions(agent_id, decisions)
        if orders:
            self.submit_orders(agent_id, orders)

    def submit_orders(self, agent_id: str, orders: Sequence[OrderRequest] | OrderRequest) -> list[OrderAck]:
        """POST …/orders. ``qty`` is shares; prices are cr / share."""
        del agent_id
        acks: list[OrderAck] = []
        for order in _order_batch(orders):
            extra = {"Idempotency-Key": order.idempotency_key} if order.idempotency_key else None
            resp = self._http.post(
                self._world_url("/orders"),
                json=_dump(order),
                headers=self._headers(extra),
            )
            acks.append(_parse(resp, OrderAck))
        return acks

    def submit_decisions(
        self,
        agent_id: str,
        decisions: Sequence[ApiFirmDecision | EngineFirmDecision] | ApiFirmDecision | EngineFirmDecision,
    ) -> list[ApiFirmDecision]:
        """POST …/firms/{fid}/decisions. Lever units match ``FirmDecision`` field docs."""
        del agent_id
        echoed: list[ApiFirmDecision] = []
        for decision in _decision_batch(decisions):
            resp = self._http.post(
                self._world_url(f"/firms/{decision.firm_id}/decisions"),
                json=_dump(decision),
                headers=self._headers(),
            )
            echoed.append(_parse(resp, ApiFirmDecision))
        return echoed

    def step(self, n: int = 1) -> WorldStatus:
        """POST …/step. Advance ``n`` ticks (days)."""
        resp = self._http.post(self._world_url("/step"), json={"n": n}, headers=self._headers())
        return _parse(resp, WorldStatus)

    def reset(self) -> WorldStatus:
        """POST …/reset. Rewind to tick 0 (days) with the same seed."""
        resp = self._http.post(self._world_url("/reset"), headers=self._headers())
        return _parse(resp, WorldStatus)

    def state_hash(self) -> str:
        """Hex digest from GET … status (unitless)."""
        return self.status().state_hash

    def status(self) -> WorldStatus:
        """GET /v1/worlds/{wid}. ``tick`` is simulated days."""
        resp = self._http.get(self._world_url(), headers=self._headers())
        return _parse(resp, WorldStatus)

    def orders(self) -> list[OpenOrder]:
        """GET …/orders. ``qty`` / ``remaining`` are shares."""
        data = _payload(self._http.get(self._world_url("/orders"), headers=self._headers()))
        items = data.get("items") if isinstance(data, dict) else None
        return [OpenOrder.model_validate(row) for row in items or ()]

    def subscribe(self, *, since: int = 0, token: str | None = None) -> StreamSubscription:
        """Open ``…/stream?since=``. ``since`` is the last received seq (unitless)."""
        wid = self._require_world()
        headers = self._headers(token=token)
        connect = getattr(self._http, "websocket_connect", None)
        if connect is None:
            raise StateError("HTTP client does not support websocket_connect")

        def _open(seq: int) -> AbstractContextManager[Any]:
            return connect(f"/v1/worlds/{wid}/stream?since={seq}", headers=headers)

        return StreamSubscription(_open, since=since)


class AsyncHttpClient(_BoundClient):
    """Async v1 client. Wraps ``httpx.AsyncClient`` (ASGI transport is a live server)."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        *,
        token: str | None = None,
        world_id: str | None = None,
        client: httpx.AsyncClient | None = None,
        app: Any | None = None,
    ) -> None:
        super().__init__(token=token, world_id=world_id)
        if client is not None:
            self._http = client
            self._owns = False
        elif app is not None:
            self._http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)
            self._owns = True
        else:
            self._http = httpx.AsyncClient(base_url=base_url)
            self._owns = True

    async def aclose(self) -> None:
        if self._owns:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncHttpClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def create_world(self, spec: WorldSpec) -> WorldStatus:
        """POST /v1/worlds. Binds ``world_id`` and the bootstrap admin bearer."""
        resp = await self._http.post("/v1/worlds", json=_dump(spec))
        return self._adopt_created(_parse(resp, WorldStatus))

    async def register_agent(self, registration: AgentRegistration) -> AgentAccount:
        """POST …/agents. ``starting_capital`` is cr; token is a SHA-256 hex digest."""
        resp = await self._http.post(self._world_url("/agents"), json=_dump(registration), headers=self._headers())
        return _parse(resp, AgentAccount)

    async def observe(self, agent_id: str | None = None, *, since: int | None = None) -> Observation:
        """GET …/observe. Agent is the bearer; ``since`` is a tick (days)."""
        del agent_id
        params: dict[str, int] = {}
        if since is not None:
            params["since"] = since
        resp = await self._http.get(self._world_url("/observe"), params=params or None, headers=self._headers())
        return _parse(resp, Observation)

    async def submit(self, agent_id: str, actions: Any) -> None:
        """POST orders and/or firm decisions for ``agent_id`` (token-bound)."""
        decisions, orders = _flatten_actions(actions)
        if decisions:
            await self.submit_decisions(agent_id, decisions)
        if orders:
            await self.submit_orders(agent_id, orders)

    async def submit_orders(self, agent_id: str, orders: Sequence[OrderRequest] | OrderRequest) -> list[OrderAck]:
        """POST …/orders. ``qty`` is shares; prices are cr / share."""
        del agent_id
        acks: list[OrderAck] = []
        for order in _order_batch(orders):
            extra = {"Idempotency-Key": order.idempotency_key} if order.idempotency_key else None
            resp = await self._http.post(
                self._world_url("/orders"),
                json=_dump(order),
                headers=self._headers(extra),
            )
            acks.append(_parse(resp, OrderAck))
        return acks

    async def submit_decisions(
        self,
        agent_id: str,
        decisions: Sequence[ApiFirmDecision | EngineFirmDecision] | ApiFirmDecision | EngineFirmDecision,
    ) -> list[ApiFirmDecision]:
        """POST …/firms/{fid}/decisions. Lever units match ``FirmDecision`` field docs."""
        del agent_id
        echoed: list[ApiFirmDecision] = []
        for decision in _decision_batch(decisions):
            resp = await self._http.post(
                self._world_url(f"/firms/{decision.firm_id}/decisions"),
                json=_dump(decision),
                headers=self._headers(),
            )
            echoed.append(_parse(resp, ApiFirmDecision))
        return echoed

    async def step(self, n: int = 1) -> WorldStatus:
        """POST …/step. Advance ``n`` ticks (days)."""
        resp = await self._http.post(self._world_url("/step"), json={"n": n}, headers=self._headers())
        return _parse(resp, WorldStatus)

    async def reset(self) -> WorldStatus:
        """POST …/reset. Rewind to tick 0 (days) with the same seed."""
        resp = await self._http.post(self._world_url("/reset"), headers=self._headers())
        return _parse(resp, WorldStatus)

    async def state_hash(self) -> str:
        """Hex digest from GET … status (unitless)."""
        return (await self.status()).state_hash

    async def status(self) -> WorldStatus:
        """GET /v1/worlds/{wid}. ``tick`` is simulated days."""
        resp = await self._http.get(self._world_url(), headers=self._headers())
        return _parse(resp, WorldStatus)

    async def orders(self) -> list[OpenOrder]:
        """GET …/orders. ``qty`` / ``remaining`` are shares."""
        data = _payload(await self._http.get(self._world_url("/orders"), headers=self._headers()))
        items = data.get("items") if isinstance(data, dict) else None
        return [OpenOrder.model_validate(row) for row in items or ()]
