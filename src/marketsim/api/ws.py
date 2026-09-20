"""WebSocket streams: ticks, fills, news, reports, releases (T7.05 / §7.2).

``GET /v1/worlds/{wid}/stream?since=`` (WebSocket). Bearer token. Event ``seq`` is
a per-world unitless counter starting at 1. ``since`` is the last received ``seq``;
replay sends ``seq > since`` with no gaps or duplicates. Fills and addressed news
are filtered to the authenticated agent. Live delivery coalesces ticks (keep the
latest; never drop fills). No wall-clock: sequences follow publish order.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from marketsim.api.sessions import AuthError, WorldManager
from marketsim.core.errors import ConfigError
from marketsim.firms.accounts import AGENT_PREFIX

EVENT_KINDS: frozenset[str] = frozenset({"tick", "fill", "news", "report", "release"})

# 1-based event seq (unitless). 0 = ``subscribed`` control frame.
FIRST_EVENT_SEQ = 1
SUBSCRIBED_SEQ = 0

# Close codes: HTTP status + 4000 (no yaml key; T7.05 file list).
WS_CLOSE_BAD_REQUEST = 4400
WS_CLOSE_UNAUTHORIZED = 4401
WS_CLOSE_NOT_FOUND = 4404


@dataclass(frozen=True)
class StreamEvent:
    """One log record. ``tick`` is simulated days; ``seq`` is unitless."""

    seq: int
    kind: str
    tick: int
    payload: dict[str, Any]
    agent_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """JSON envelope. ``tick`` is days; ``seq`` is unitless."""
        return {"seq": int(self.seq), "type": self.kind, "tick": int(self.tick), "payload": dict(self.payload)}

    def to_state(self) -> dict[str, Any]:
        """Serialise one event (days / unitless seq) including optional ``agent_id``."""
        return {**self.to_wire(), "kind": self.kind, "agent_id": self.agent_id}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> StreamEvent:
        raw = state.get("agent_id")
        return cls(
            seq=int(state["seq"]),
            kind=str(state.get("kind") or state.get("type")),
            tick=int(state.get("tick", 0)),
            payload=dict(state.get("payload") or {}),
            agent_id=None if raw is None else str(raw),
        )


def _aliases(raw: str) -> set[str]:
    text = str(raw)
    return {text, text[len(AGENT_PREFIX) :] if text.startswith(AGENT_PREFIX) else f"{AGENT_PREFIX}{text}"}


def event_visible(event: StreamEvent, agent_id: str) -> bool:
    """Whether ``agent_id`` (unitless) may see this event. Fills/news are filtered."""
    if event.kind == "fill":
        parties: set[str] = set()
        for key in ("maker", "taker"):
            value = event.payload.get(key)
            if value:
                parties.update(_aliases(str(value)))
        if event.agent_id is not None:
            parties.update(_aliases(event.agent_id))
        return bool(parties.intersection(_aliases(agent_id)))
    audience = event.agent_id
    if event.kind == "news" and audience is None:
        extra = event.payload.get("agent_id")
        if extra:
            audience = str(extra)
    return audience is None or agent_id in _aliases(audience)


def coalesce_ticks(events: Sequence[StreamEvent]) -> list[StreamEvent]:
    """Keep the latest tick; never drop fills or other kinds. ``seq`` unchanged."""
    last_tick_i = None
    for i, event in enumerate(events):
        if event.kind == "tick":
            last_tick_i = i
    return [event for i, event in enumerate(events) if event.kind != "tick" or i == last_tick_i]


class _Mailbox:
    """Per-connection live buffer. Unsent ticks coalesce until the consumer drains."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[StreamEvent] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._gate: asyncio.Event | None = None

    def bind(self, loop: asyncio.AbstractEventLoop, gate: asyncio.Event) -> None:
        with self._lock:
            self._loop = loop
            self._gate = gate
            pending = bool(self._items)
        if pending:
            loop.call_soon_threadsafe(gate.set)

    def push_many(self, events: Sequence[StreamEvent]) -> None:
        if not events:
            return
        with self._lock:
            self._items = coalesce_ticks([*self._items, *events])
            loop, gate = self._loop, self._gate
        if loop is not None and gate is not None:
            loop.call_soon_threadsafe(gate.set)

    def take(self) -> list[StreamEvent]:
        with self._lock:
            batch, self._items = self._items, []
            if self._gate is not None:
                self._gate.clear()
            return batch


class StreamHub:
    """Per-world event log plus live subscribers. Sequences are deterministic."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._logs: dict[str, list[StreamEvent]] = {}
        self._subs: dict[str, dict[str, list[_Mailbox]]] = {}

    def publish(
        self,
        world_id: str,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        tick: int = 0,
        agent_id: str | None = None,
    ) -> StreamEvent:
        """Append one event. ``tick`` is simulated days; returned ``seq`` is unitless."""
        return self.publish_many(
            world_id,
            ({"kind": kind, "payload": dict(payload or {}), "tick": tick, "agent_id": agent_id},),
        )[0]

    def publish_many(self, world_id: str, items: Sequence[Mapping[str, Any]]) -> list[StreamEvent]:
        """Append a batch. Each item: ``kind``, ``payload``, optional ``tick`` (days)."""
        recorded: list[StreamEvent] = []
        with self._lock:
            log = self._logs.setdefault(world_id, [])
            seq = log[-1].seq + 1 if log else FIRST_EVENT_SEQ
            for item in items:
                kind = str(item["kind"])
                if kind not in EVENT_KINDS:
                    raise ValueError(f"unknown stream kind {kind!r}")
                raw_agent = item.get("agent_id")
                event = StreamEvent(
                    seq=seq,
                    kind=kind,
                    tick=int(item.get("tick", 0)),
                    payload=dict(item.get("payload") or {}),
                    agent_id=None if raw_agent is None else str(raw_agent),
                )
                log.append(event)
                recorded.append(event)
                seq += 1
            sub_map = {aid: list(boxes) for aid, boxes in self._subs.get(world_id, {}).items()}
        for agent_id in sorted(sub_map):
            visible = [event for event in recorded if event_visible(event, agent_id)]
            if visible:
                for box in sub_map[agent_id]:
                    box.push_many(visible)
        return recorded

    def replay(self, world_id: str, agent_id: str, since: int) -> list[StreamEvent]:
        """Events with ``seq > since`` visible to ``agent_id``. Exact log; no coalesce."""
        with self._lock:
            log = list(self._logs.get(world_id, ()))
        return [event for event in log if event.seq > since and event_visible(event, agent_id)]

    def attach(self, world_id: str, agent_id: str, since: int) -> tuple[_Mailbox, list[StreamEvent]]:
        """Register a live mailbox and snapshot the resume window under one lock."""
        box = _Mailbox()
        with self._lock:
            self._subs.setdefault(world_id, {}).setdefault(agent_id, []).append(box)
            log = list(self._logs.get(world_id, ()))
        return box, [event for event in log if event.seq > since and event_visible(event, agent_id)]

    def detach(self, world_id: str, agent_id: str, box: _Mailbox) -> None:
        """Drop ``box`` from the subscriber table (unitless ids)."""
        with self._lock:
            boxes = self._subs.get(world_id, {}).get(agent_id)
            if not boxes:
                return
            remain = [item for item in boxes if item is not box]
            if remain:
                self._subs[world_id][agent_id] = remain
            else:
                del self._subs[world_id][agent_id]
                if not self._subs[world_id]:
                    del self._subs[world_id]

    def to_state(self) -> dict[str, Any]:
        """Serialise world logs. Live sockets are not part of state."""
        with self._lock:
            return {"logs": {wid: [e.to_state() for e in ev] for wid, ev in sorted(self._logs.items())}}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> StreamHub:
        hub = cls()
        raw = state.get("logs") or {}
        hub._logs = {str(wid): [StreamEvent.from_state(row) for row in raw[wid]] for wid in sorted(raw)}
        return hub


def _bearer(websocket: WebSocket) -> str | None:
    header = websocket.headers.get("authorization")
    if not header:
        return None
    scheme, _, token = str(header).partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _parse_since(websocket: WebSocket) -> int | None:
    try:
        since = int(websocket.query_params.get("since", "0"))
    except (TypeError, ValueError):
        return None
    return since if since >= 0 else None


def stream_router() -> APIRouter:
    """WebSocket router. Path is ``/v1/worlds/{wid}/stream``."""
    router = APIRouter(tags=["stream"])

    @router.websocket("/v1/worlds/{wid}/stream")
    async def stream(websocket: WebSocket, wid: str) -> None:
        await stream_world(websocket, wid)

    return router


async def stream_world(websocket: WebSocket, wid: str) -> None:
    """Serve one agent stream. ``wid`` is a world id (unitless)."""
    manager: WorldManager = websocket.app.state.ws_manager
    hub: StreamHub = websocket.app.state.ws_hub
    since = _parse_since(websocket)
    if since is None:
        await websocket.close(code=WS_CLOSE_BAD_REQUEST, reason="since must be an integer seq >= 0")
        return
    token = _bearer(websocket)
    if token is None:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED, reason="missing bearer token")
        return
    if wid not in manager.world_ids():
        await websocket.close(code=WS_CLOSE_NOT_FOUND, reason=f"unknown world {wid!r}")
        return
    try:
        session = manager.authenticate(wid, token)
    except AuthError:
        await websocket.close(code=WS_CLOSE_UNAUTHORIZED, reason="invalid token")
        return

    box, replayed = hub.attach(wid, session.agent_id, since)
    gate = asyncio.Event()
    box.bind(asyncio.get_running_loop(), gate)
    await websocket.accept()
    try:
        try:
            world_tick = int(manager.world(wid).clock.tick)
        except ConfigError:
            world_tick = 0
        await websocket.send_json(
            {
                "seq": SUBSCRIBED_SEQ,
                "type": "subscribed",
                "tick": world_tick,
                "payload": {"since": since, "agent_id": session.agent_id},
            }
        )
        last_seq = since
        for event in replayed:
            await websocket.send_json(event.to_wire())
            last_seq = event.seq
        while True:
            batch = box.take()
            if not batch:
                await gate.wait()
                continue
            fresh = [event for event in batch if event.seq > last_seq]
            if not fresh:
                continue
            last_seq = max(event.seq for event in fresh)
            for event in coalesce_ticks(fresh):
                await websocket.send_json(event.to_wire())
    except WebSocketDisconnect:
        return
    finally:
        hub.detach(wid, session.agent_id, box)


def include_ws(
    app: FastAPI,
    *,
    manager: WorldManager | None = None,
    hub: StreamHub | None = None,
) -> FastAPI:
    """Attach ``/v1/worlds/{wid}/stream``. T7.04 may call this on ``create_app``."""
    resolved = manager or getattr(app.state, "ws_manager", None)
    if resolved is None:
        api = getattr(app.state, "api", None)
        resolved = getattr(api, "manager", None) if api is not None else None
    app.state.ws_manager = resolved or WorldManager()
    app.state.ws_hub = hub or getattr(app.state, "ws_hub", None) or StreamHub()
    if not getattr(app.state, "ws_routes_included", False):
        app.include_router(stream_router())
        app.state.ws_routes_included = True
    return app
