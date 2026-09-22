"""In-process servicer: ``LOCAL_METHODS`` on ``WorldManager`` / ``LocalClient`` (T9.06).

Auth is the REST bearer token in metadata key ``authorization``. Methods are
safe to call directly (no socket). Hidden observe keys never enter
``Observation`` (``extra='forbid'``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marketsim.api.schemas import FirmDecision as ApiFirmDecision
from marketsim.api.schemas import Observation, OrderRequest, SubmitPayload, WorldStatus
from marketsim.api.sessions import AuthError, Session, WorldManager
from marketsim.firms.levers import FirmDecision as EngineFirmDecision
from marketsim.sdk.local import LOCAL_METHODS, LocalClient

# Proto service name (unitless). Matches ``marketsim.proto``.
SERVICE_NAME = "marketsim.api.v1.MarketSim"
# Same header as REST ``Authorization`` (gRPC metadata is lowercase).
AUTHORIZATION_KEY = "authorization"


def authorization_metadata(token: str) -> tuple[tuple[str, str], ...]:
    """REST-identical bearer pair. ``token`` is a SHA-256 hex digest."""
    return ((AUTHORIZATION_KEY, f"Bearer {token}"),)


def invocation_metadata(context: Any) -> tuple[tuple[str, str], ...]:
    """Normalise a gRPC context, mapping, or sequence of pairs to lowercase keys."""
    if context is None:
        return ()
    raw: Any
    getter = getattr(context, "invocation_metadata", None)
    if callable(getter):
        raw = getter() or ()
    elif isinstance(context, Mapping):
        raw = context.items()
    elif isinstance(context, (list, tuple)):
        raw = context
    else:
        return ()
    out: list[tuple[str, str]] = []
    for key, value in raw:
        text = value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        out.append((str(key).lower(), text))
    return tuple(out)


def bearer_token(context: Any) -> str:
    """Extract the bearer token. Raises ``AuthError`` if missing or malformed."""
    for key, value in invocation_metadata(context):
        if key != AUTHORIZATION_KEY:
            continue
        scheme, _, token = value.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("invalid authorization header")
        return token.strip()
    raise AuthError("missing bearer token")


def _req_get(request: Any, key: str, default: Any = None) -> Any:
    if isinstance(request, Mapping):
        return request.get(key, default)
    return getattr(request, key, default)


def _hydrate(value: Any) -> Any:
    if isinstance(value, (SubmitPayload, OrderRequest, ApiFirmDecision, EngineFirmDecision)):
        return value
    if isinstance(value, Mapping):
        if "side" in value and "qty" in value:
            return OrderRequest.model_validate(value)
        if "firm_id" in value:
            return ApiFirmDecision.model_validate(value)
        if "decisions" in value or "orders" in value:
            return SubmitPayload.model_validate(value)
        return value
    if isinstance(value, (list, tuple)):
        return [_hydrate(item) for item in value]
    return value


def _to_observation(raw: Mapping[str, Any], *, tick: int, agent_id: str) -> Observation:
    payload = dict(raw)
    payload.setdefault("tick", tick)
    payload.setdefault("agent_id", agent_id)
    return Observation.model_validate(payload)


class MarketSimServicer:
    """Unary methods named exactly as ``LOCAL_METHODS``. Tick units are days."""

    methods: frozenset[str] = LOCAL_METHODS

    def __init__(self, manager: WorldManager) -> None:
        self._manager = manager

    def _bind(self, request: Any, context: Any) -> tuple[Session, LocalClient]:
        world_id = str(_req_get(request, "world_id") or "")
        if not world_id.strip():
            raise AuthError("world_id is required")
        session = self._manager.authenticate(world_id, bearer_token(context))
        client = LocalClient.from_manager(self._manager, session.world_id, session.token)
        return session, client

    def observe(self, request: Any, context: Any = None) -> Observation:
        """Published observation (T6.22 whitelist). ``tick`` is simulated days."""
        session, client = self._bind(request, context)
        agent_id = _req_get(request, "agent_id")
        raw = client.observe(None if agent_id in (None, "") else str(agent_id))
        return _to_observation(raw, tick=int(raw.get("tick", 0)), agent_id=session.agent_id)

    def submit(self, request: Any, context: Any = None) -> None:
        """Queue ``actions`` for the token-bound agent. Applied on the next step (days)."""
        session, client = self._bind(request, context)
        actions = _req_get(request, "actions")
        if actions is None:
            return
        client.submit(session.agent_id, _hydrate(actions))

    def submit_orders(self, request: Any, context: Any = None) -> None:
        """Submit orders. ``qty`` is shares; prices are cr / share."""
        session, client = self._bind(request, context)
        orders = _req_get(request, "orders")
        if orders is None:
            return
        client.submit_orders(session.agent_id, _hydrate(orders))

    def submit_decisions(self, request: Any, context: Any = None) -> None:
        """Submit firm decisions. Lever units match ``FirmDecision`` field docs."""
        session, client = self._bind(request, context)
        decisions = _req_get(request, "decisions")
        if decisions is None:
            return
        client.submit_decisions(session.agent_id, _hydrate(decisions))

    def step(self, request: Any, context: Any = None) -> WorldStatus:
        """Advance ``n`` ticks (days). ``n`` is a count; default 1."""
        session, client = self._bind(request, context)
        n = int(_req_get(request, "n", 1))
        if n < 1:
            raise ValueError("n must be >= 1 (ticks / days)")
        client.step(n)
        return self._manager.status(session.world_id)

    def reset(self, request: Any, context: Any = None) -> WorldStatus:
        """Rewind to tick 0 (days) with the same seed."""
        session, client = self._bind(request, context)
        client.reset()
        return self._manager.status(session.world_id)

    def state_hash(self, request: Any, context: Any = None) -> str:
        """Deterministic hex digest of world state (unitless)."""
        _, client = self._bind(request, context)
        return client.state_hash()

    def status(self, request: Any, context: Any = None) -> WorldStatus:
        """Live ``WorldStatus``. ``tick`` is simulated days."""
        session, client = self._bind(request, context)
        got = client.status()
        return got if got is not None else self._manager.status(session.world_id)
