"""In-process client. Same snake_case surface T7.04 / T7.07 must match (T7.03 / §7.2).

No wall-clock, network, or global RNG — every call delegates to ``World`` or
``WorldManager``. ``LOCAL_METHODS`` is the contract the REST SDK (T7.07) exposes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from marketsim.api.schemas import FirmDecision as ApiFirmDecision
from marketsim.api.schemas import OrderRequest, SubmitPayload, WorldStatus
from marketsim.api.sessions import AuthError, Session, WorldManager
from marketsim.core.errors import StateError
from marketsim.firms.levers import FirmDecision as EngineFirmDecision
from marketsim.scenarios.randomise import observe as public_observe
from marketsim.world import World

# REST SDK (T7.07) must expose this exact set of public method names.
LOCAL_METHODS: frozenset[str] = frozenset(
    {
        "observe",
        "submit",
        "submit_orders",
        "submit_decisions",
        "step",
        "reset",
        "state_hash",
        "status",
    }
)


class LocalClient:
    """In-process handle. Tick units are simulated days; cash fields are cr."""

    def __init__(self, world: World) -> None:
        self._world = world
        self._manager: WorldManager | None = None
        self._session: Session | None = None

    @classmethod
    def from_manager(cls, manager: WorldManager, world_id: str, token: str) -> LocalClient:
        """Bind to one session. ``world_id`` is unitless; ``token`` is a SHA-256 hex digest."""
        session = manager.authenticate(world_id, token)
        client = cls(manager.world(world_id))
        client._manager = manager
        client._session = session
        return client

    def observe(self, agent_id: str | None = None) -> dict[str, Any]:
        """Published observation (T6.22 whitelist). ``tick`` is simulated days."""
        return public_observe(self._world, self._agent(agent_id))

    def submit(self, agent_id: str, actions: Any) -> None:
        """Queue actions for ``agent_id``. Applied on the next ``step`` (days)."""
        aid = self._agent(agent_id)
        if isinstance(actions, SubmitPayload):
            self.submit_decisions(aid, actions.decisions)
            self.submit_orders(aid, actions.orders)
            return
        if isinstance(actions, (list, tuple)):
            for item in actions:
                self.submit(aid, item)
            return
        if isinstance(actions, OrderRequest):
            self.submit_orders(aid, actions)
            return
        if isinstance(actions, (ApiFirmDecision, EngineFirmDecision)):
            self.submit_decisions(aid, actions)
            return
        self._enqueue(aid, actions)

    def submit_orders(self, agent_id: str, orders: Sequence[OrderRequest] | OrderRequest) -> None:
        """Submit orders. ``qty`` is shares; prices are cr / share."""
        aid = self._agent(agent_id)
        batch: Sequence[OrderRequest] = (orders,) if isinstance(orders, OrderRequest) else orders
        for order in batch:
            self._enqueue(aid, order)

    def submit_decisions(
        self,
        agent_id: str,
        decisions: Sequence[ApiFirmDecision | EngineFirmDecision] | ApiFirmDecision | EngineFirmDecision,
    ) -> None:
        """Submit firm decisions. Lever units match ``FirmDecision`` field docs."""
        aid = self._agent(agent_id)
        if isinstance(decisions, (ApiFirmDecision, EngineFirmDecision)):
            batch: Sequence[ApiFirmDecision | EngineFirmDecision] = (decisions,)
        else:
            batch = decisions
        for decision in batch:
            self._enqueue(aid, decision)

    def step(self, n: int = 1) -> None:
        """Advance ``n`` ticks (days)."""
        if self._manager is not None and self._session is not None:
            self._manager.step(self._session.world_id, n)
            return
        self._world.step(n)

    def reset(self) -> None:
        """Rewind to tick 0 (days) with the same seed. Inbox and observations clear."""
        self._world.reset()

    def state_hash(self) -> str:
        """Deterministic hex digest of world state (unitless)."""
        return self._world.state_hash()

    def status(self) -> WorldStatus | None:
        """Live ``WorldStatus`` when bound to a manager; otherwise ``None``."""
        if self._manager is None or self._session is None:
            return None
        return self._manager.status(self._session.world_id)

    def _agent(self, agent_id: str | None) -> str:
        if self._session is not None:
            if agent_id is None or agent_id == self._session.agent_id:
                return self._session.agent_id
            raise AuthError(f"token is bound to {self._session.agent_id!r}")
        if agent_id is None or agent_id == "":
            raise StateError("agent_id is required")
        return agent_id

    def _enqueue(self, agent_id: str, action: Any) -> None:
        if self._manager is not None and self._session is not None:
            if isinstance(action, OrderRequest):
                self._manager.submit_order(self._session.world_id, self._session.token, action)
                return
            if isinstance(action, ApiFirmDecision):
                self._manager.record_decision(self._session.world_id, self._session.token, action)
                return
            if isinstance(action, EngineFirmDecision):
                self._manager.record_decision(
                    self._session.world_id,
                    self._session.token,
                    ApiFirmDecision.from_engine(action),
                )
                return
        self._world.submit(agent_id, action)
