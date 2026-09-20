"""Multi-world sessions, agent tokens, and per-agent limits (T7.02 / §7.2)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marketsim.api.schemas import (
    AgentRegistration,
    AgentRole,
    AuthorityName,
    FirmDecision,
    OrderRequest,
    WorldRunStatus,
    WorldSpec,
    WorldStatus,
)
from marketsim.core.errors import ConfigError, MarketsimError, StateError
from marketsim.firms.accounts import agent_entity
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent
from marketsim.world import World

# Per-agent caps (§7.2). Literals stay in this module — T7.02 cannot add yaml / config.py keys.
_ORDERS_PER_TICK = 32
_DECISIONS_PER_TICK = 8
_MAX_OPEN_ORDERS = 64

BANK_ENTITY = "BANKSYS"
CAPITAL_TAG = "capital_transfer"


class AuthError(MarketsimError):
    """Unknown world, invalid token, or a per-agent limit was exceeded."""


@dataclass(frozen=True)
class AgentLimits:
    """Per-agent API caps (§7.2). Units: counts (per tick or resting orders)."""

    orders_per_tick: int = _ORDERS_PER_TICK  # §7.2
    decisions_per_tick: int = _DECISIONS_PER_TICK  # §7.2
    max_open_orders: int = _MAX_OPEN_ORDERS  # §7.2


DEFAULT_LIMITS = AgentLimits()


@dataclass(frozen=True)
class Session:
    """Authenticated handle. ``starting_capital`` is posted in cr on the world ledger."""

    world_id: str
    agent_id: str
    role: AgentRole
    entity: str
    token: str
    limits: AgentLimits = DEFAULT_LIMITS
    authority: AuthorityName | None = None


@dataclass
class _AgentRecord:
    registration: AgentRegistration
    token: str
    entity: str
    orders_this_tick: int = 0
    decisions_this_tick: int = 0
    open_orders: int = 0

    @property
    def authority(self) -> AuthorityName | None:
        return self.registration.authority if self.registration.role == "policymaker" else None


@dataclass
class _ManagedWorld:
    world_id: str
    spec: WorldSpec
    world: World
    ledger: Ledger
    config_dir: str
    agents: dict[str, _AgentRecord] = field(default_factory=dict)
    status: WorldRunStatus = "created"


def issue_token(world_id: str, agent_id: str, seed: int, role: str) -> str:
    """Deterministic session token. Unitless hex digest; no OS entropy."""
    payload = f"{world_id}:{agent_id}:{seed}:{role}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _limits_payload() -> AgentLimits:
    return DEFAULT_LIMITS


class WorldManager:
    """Many isolated worlds, each with its own ``World``, ledger, and agent table."""

    def __init__(self) -> None:
        self._seq = 0
        self._worlds: dict[str, _ManagedWorld] = {}
        self._config_dir: str | None = None

    def create_world(self, spec: WorldSpec, config_dir: str | Path) -> WorldStatus:
        """Create one world. ``spec.seed`` is the RNG root; tick starts at 0 (days)."""
        root = Path(config_dir).resolve()
        self._config_dir = str(root)
        self._seq += 1
        world_id = f"w{self._seq:04d}"
        world = World.create(
            root,
            seed=spec.seed,
            overrides=dict(spec.overrides) if spec.overrides else None,
            scenario=spec.scenario,
        )
        handle = _ManagedWorld(
            world_id=world_id,
            spec=spec,
            world=world,
            ledger=Ledger.empty(),
            config_dir=str(root),
        )
        self._worlds[world_id] = handle
        return self.status(world_id)

    def register_agent(self, world_id: str, registration: AgentRegistration) -> dict[str, Any]:
        """Register ``AGENT:<id>``. ``starting_capital`` is cr; token is a hex digest."""
        handle = self._handle(world_id)
        if registration.authority is not None and registration.role != "policymaker":
            raise ConfigError("authority requires role=policymaker")
        if registration.agent_id in handle.agents:
            raise ConfigError(f"agent {registration.agent_id!r} already registered")
        entity = agent_entity(registration.agent_id)
        handle.ledger.register_entity(entity)
        capital = float(registration.starting_capital)
        if capital > 0:
            if BANK_ENTITY not in handle.ledger.entities:
                handle.ledger.register_entity(BANK_ENTITY)
            handle.ledger.post(
                Tx(
                    handle.world.clock.tick,
                    CAPITAL_TAG,
                    (
                        Entry(BANK_ENTITY, "DEP", -capital),
                        Entry(entity, "DEP", capital),
                    ),
                    memo=f"start {registration.agent_id}",
                )
            )
        assert_consistent(handle.ledger)
        token = issue_token(world_id, registration.agent_id, handle.world.seed, registration.role)
        handle.agents[registration.agent_id] = _AgentRecord(
            registration=registration,
            token=token,
            entity=entity,
        )
        return {"token": token, "entity": entity, "limits": _limits_payload()}

    def authenticate(self, world_id: str, token: str) -> Session:
        """Return the session for ``token`` on ``world_id``. Raises AuthError on mismatch."""
        if world_id not in self._worlds:
            raise AuthError(f"unknown world {world_id!r}")
        handle = self._worlds[world_id]
        for agent_id in sorted(handle.agents):
            rec = handle.agents[agent_id]
            if rec.token == token:
                return Session(
                    world_id=world_id,
                    agent_id=agent_id,
                    role=rec.registration.role,
                    entity=rec.entity,
                    token=rec.token,
                    limits=DEFAULT_LIMITS,
                    authority=rec.authority,
                )
        raise AuthError("invalid token")

    def submit_order(self, world_id: str, token: str, order: OrderRequest | None = None) -> None:
        """Count one order this tick (unitless). Rejects at ``AgentLimits.orders_per_tick``."""
        session = self.authenticate(world_id, token)
        handle = self._handle(world_id)
        rec = handle.agents[session.agent_id]
        if rec.orders_this_tick >= DEFAULT_LIMITS.orders_per_tick:
            raise AuthError(f"orders_per_tick exceeded ({DEFAULT_LIMITS.orders_per_tick})")
        if rec.open_orders >= DEFAULT_LIMITS.max_open_orders:
            raise AuthError(f"max_open_orders exceeded ({DEFAULT_LIMITS.max_open_orders})")
        rec.orders_this_tick += 1
        rec.open_orders += 1
        if order is not None:
            handle.world.submit(session.agent_id, order)

    def record_decision(self, world_id: str, token: str, decision: FirmDecision | None = None) -> None:
        """Count one firm decision this tick (unitless). Rejects at ``AgentLimits.decisions_per_tick``."""
        session = self.authenticate(world_id, token)
        handle = self._handle(world_id)
        rec = handle.agents[session.agent_id]
        if rec.decisions_this_tick >= DEFAULT_LIMITS.decisions_per_tick:
            raise AuthError(f"decisions_per_tick exceeded ({DEFAULT_LIMITS.decisions_per_tick})")
        rec.decisions_this_tick += 1
        if decision is not None:
            handle.world.submit(session.agent_id, decision)

    def step(self, world_id: str, n: int = 1) -> WorldStatus:
        """Advance ``n`` ticks (days) on one world; reset that world's per-tick counters."""
        handle = self._handle(world_id)
        handle.world.step(n)
        handle.status = "running"
        for agent_id in sorted(handle.agents):
            rec = handle.agents[agent_id]
            rec.orders_this_tick = 0
            rec.decisions_this_tick = 0
        return self.status(world_id)

    def status(self, world_id: str) -> WorldStatus:
        """Live ``WorldStatus``. ``tick`` is simulated days; ``n_agents`` is a count."""
        handle = self._handle(world_id)
        return WorldStatus(
            world_id=handle.world_id,
            tick=handle.world.clock.tick,
            seed=handle.world.seed,
            mode=handle.spec.mode,
            run_mode=handle.spec.run_mode,
            state_hash=handle.world.state_hash(),
            n_agents=len(handle.agents),
            status=handle.status,
        )

    def world(self, world_id: str) -> World:
        """The engine ``World`` for ``world_id``."""
        return self._handle(world_id).world

    def ledger(self, world_id: str) -> Ledger:
        """That world's agent ledger (cr positions)."""
        return self._handle(world_id).ledger

    def world_ids(self) -> tuple[str, ...]:
        """Sorted world ids (unitless)."""
        return tuple(sorted(self._worlds))

    def to_state(self) -> dict[str, Any]:
        """Serialise every world, ledger, and session. Iteration is sorted by id."""
        worlds: dict[str, Any] = {}
        for wid in sorted(self._worlds):
            handle = self._worlds[wid]
            agents = {
                aid: {
                    "registration": rec.registration.model_dump(mode="python"),
                    "token": rec.token,
                    "entity": rec.entity,
                    "orders_this_tick": rec.orders_this_tick,
                    "decisions_this_tick": rec.decisions_this_tick,
                    "open_orders": rec.open_orders,
                }
                for aid, rec in sorted(handle.agents.items())
            }
            worlds[wid] = {
                "world_id": handle.world_id,
                "spec": handle.spec.model_dump(mode="python"),
                "status": handle.status,
                "config_dir": handle.config_dir,
                "world": handle.world.to_state(),
                "ledger": handle.ledger.to_state(),
                "agents": agents,
            }
        return {"seq": self._seq, "config_dir": self._config_dir, "worlds": worlds}

    @classmethod
    def from_state(cls, state: dict[str, Any], *, config_dir: str | Path | None = None) -> WorldManager:
        """Restore a manager. ``config_dir`` is a filesystem path used to rebuild each ``World``."""
        mgr = cls()
        mgr._seq = int(state.get("seq", 0))
        raw_worlds = state.get("worlds") or {}
        root: Path | None
        if config_dir is not None:
            root = Path(config_dir)
        elif state.get("config_dir"):
            root = Path(str(state["config_dir"]))
        else:
            root = None
        mgr._config_dir = str(root) if root is not None else None
        if raw_worlds and root is None:
            raise StateError("WorldManager.from_state needs config_dir")
        for wid in sorted(raw_worlds):
            raw = raw_worlds[wid]
            spec = WorldSpec.model_validate(raw["spec"])
            cfg = Path(raw["config_dir"]) if raw.get("config_dir") else root
            if cfg is None:
                raise StateError(f"missing config_dir for world {wid!r}")
            world = World.create(
                cfg,
                seed=spec.seed,
                overrides=dict(spec.overrides) if spec.overrides else None,
                scenario=spec.scenario,
            )
            world.load_state(raw["world"])
            handle = _ManagedWorld(
                world_id=str(raw.get("world_id", wid)),
                spec=spec,
                world=world,
                ledger=Ledger.from_state(raw["ledger"]),
                config_dir=str(cfg),
                status=raw.get("status", "created"),
            )
            for aid in sorted(raw.get("agents") or {}):
                araw = raw["agents"][aid]
                handle.agents[str(aid)] = _AgentRecord(
                    registration=AgentRegistration.model_validate(araw["registration"]),
                    token=str(araw["token"]),
                    entity=str(araw["entity"]),
                    orders_this_tick=int(araw.get("orders_this_tick", 0)),
                    decisions_this_tick=int(araw.get("decisions_this_tick", 0)),
                    open_orders=int(araw.get("open_orders", 0)),
                )
            mgr._worlds[str(wid)] = handle
        return mgr

    def _handle(self, world_id: str) -> _ManagedWorld:
        try:
            return self._worlds[world_id]
        except KeyError as exc:
            raise ConfigError(f"unknown world {world_id!r}") from exc
