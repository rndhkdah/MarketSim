"""Event-sourced replay log (T7.10 / §7.2).

The log header is ``seed``, JSON-sorted ``overrides`` plus their SHA-256, and
domain-randomisation draws. The body is every agent input, scripted events, and
lightweight snapshots (tick + ``World.state_hash()``). ``replay`` applies inputs
in ``(agent_id, sequence)`` order — the same order as ``LockstepBarrier``.
No wall-clock; RNG is only whatever ``World`` already owns.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marketsim.scenarios.randomise import config_overrides
from marketsim.world import World

# Export series (T7.10). A key change is a breaking schema change.
# tick: days (clock.tick after each step)
# state_hash: hex SHA-256 of World.to_state() (64 ascii chars)
# seed: RNG root seed (unitless)
# config_hash: hex SHA-256 of json-sorted overrides (64 ascii chars)
EXPORT_KEYS: tuple[str, ...] = ("tick", "state_hash", "seed", "config_hash")
EXPORT_DTYPES: dict[str, str] = {
    "tick": "int64",
    "state_hash": "U64",
    "seed": "int64",
    "config_hash": "U64",
}
EXPORT_UNITS: dict[str, str] = {
    "tick": "days",
    "state_hash": "hex SHA-256",
    "seed": "unitless",
    "config_hash": "hex SHA-256",
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(value[k]) for k in value}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _json_copy(value: Any) -> Any:
    return _jsonable(value)


def hash_overrides(overrides: Mapping[str, Any] | None) -> str:
    """SHA-256 hex (unitless) of ``overrides`` encoded as JSON with sorted keys.

    ``overrides`` are dotted ``load_config`` paths. Encoding is UTF-8; no wall-clock.
    """
    payload = json.dumps(
        _jsonable({} if overrides is None else dict(overrides)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def world_overrides(log: ReplayLog) -> dict[str, Any]:
    """Dotted paths for ``World.create``: header overrides plus randomisation draws."""
    merged = dict(log.overrides)
    draws = {str(k): v for k, v in log.randomisation_draws.items()}
    merged.update(config_overrides(draws) if draws else {})
    return merged


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """One agent input. ``tick`` is dest tick (days); ``sequence`` is client-local."""

    tick: int
    agent_id: str
    sequence: int
    payload: Any


@dataclass(frozen=True, slots=True)
class ScriptedEvent:
    """One scripted fire. ``tick`` is days; ``priority`` is the queue key (unitless)."""

    tick: int
    event_id: str
    payload: dict[str, Any]
    priority: int = 0


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    """Lightweight snapshot. ``tick`` is ``clock.tick`` (days); hash is hex SHA-256."""

    tick: int
    state_hash: str


@dataclass
class ReplayLog:
    """Event-sourced run: header + agent inputs + scripted events + snapshots."""

    seed: int
    overrides: dict[str, Any] = field(default_factory=dict)
    randomisation_draws: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    records: list[ReplayRecord] = field(default_factory=list)
    scripted_events: list[ScriptedEvent] = field(default_factory=list)
    snapshots: list[ReplaySnapshot] = field(default_factory=list)
    n_ticks: int = 0

    def __post_init__(self) -> None:
        self.overrides = {str(k): _json_copy(v) for k, v in self.overrides.items()}
        self.randomisation_draws = {str(k): _json_copy(v) for k, v in self.randomisation_draws.items()}
        if not self.config_hash:
            self.config_hash = hash_overrides(self.overrides)

    def append(
        self,
        tick: int,
        agent_id: str,
        payload: Any,
        *,
        sequence: int | None = None,
    ) -> None:
        """Record one agent input. ``tick`` is dest tick (days); ``sequence`` is client-local."""
        seq = int(sequence) if sequence is not None else self._next_sequence(int(tick), str(agent_id))
        self.records.append(ReplayRecord(int(tick), str(agent_id), seq, _json_copy(payload)))

    def append_scripted(
        self,
        tick: int,
        event_id: str,
        payload: Mapping[str, Any] | None = None,
        *,
        priority: int = 0,
    ) -> None:
        """Record one scripted event. ``tick`` is days; ``priority`` matches ``EventQueue``."""
        body = (
            dict(_json_copy(payload))
            if payload is not None
            else {"kind": "scripted", "event_id": str(event_id)}
        )
        self.scripted_events.append(ScriptedEvent(int(tick), str(event_id), body, int(priority)))

    def record_snapshot(self, world: World) -> ReplaySnapshot:
        """Append tick + ``state_hash``. ``world`` is the live engine object."""
        snap = ReplaySnapshot(tick=int(world.clock.tick), state_hash=world.state_hash())
        self.snapshots.append(snap)
        if snap.tick > self.n_ticks:
            self.n_ticks = snap.tick
        return snap

    def record_applied(self, world: World, applied: list[tuple[str, int, Any]]) -> None:
        """Append a just-stepped inbox. ``applied`` is ``(agent_id, sequence, payload)``.

        Call after ``world.step()`` / ``LockstepBarrier.maybe_advance()``. The dest
        tick is ``clock.tick - 1`` (days).
        """
        dest = int(world.clock.tick) - 1
        if dest < 0:
            dest = 0
        for agent_id, sequence, payload in applied:
            self.append(dest, agent_id, payload, sequence=int(sequence))
        self.record_snapshot(world)

    def horizon(self) -> int:
        """First unreplayed ``clock.tick`` (days)."""
        candidates = [0]
        if self.n_ticks > 0:
            candidates.append(int(self.n_ticks))
        if self.snapshots:
            candidates.append(max(s.tick for s in self.snapshots))
        if self.records:
            candidates.append(max(r.tick for r in self.records) + 1)
        return max(candidates)

    def to_state(self) -> dict[str, Any]:
        """JSON-safe log. Keys are sorted at write time; records stay in append order."""
        self.config_hash = hash_overrides(self.overrides)
        return {
            "seed": int(self.seed),
            "config_hash": self.config_hash,
            "overrides": {k: self.overrides[k] for k in sorted(self.overrides, key=str)},
            "randomisation_draws": {
                k: self.randomisation_draws[k] for k in sorted(self.randomisation_draws, key=str)
            },
            "n_ticks": int(self.n_ticks),
            "records": [
                {
                    "tick": r.tick,
                    "agent_id": r.agent_id,
                    "sequence": r.sequence,
                    "payload": r.payload,
                }
                for r in self.records
            ],
            "scripted_events": [
                {
                    "tick": e.tick,
                    "event_id": e.event_id,
                    "payload": e.payload,
                    "priority": e.priority,
                }
                for e in self.scripted_events
            ],
            "snapshots": [{"tick": s.tick, "state_hash": s.state_hash} for s in self.snapshots],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> ReplayLog:
        """Restore a log. ``state`` is a ``to_state()`` mapping."""
        log = cls(
            seed=int(state["seed"]),
            overrides=dict(state.get("overrides") or {}),
            randomisation_draws=dict(state.get("randomisation_draws") or {}),
            config_hash=str(state.get("config_hash") or ""),
            n_ticks=int(state.get("n_ticks") or 0),
        )
        for raw in state.get("records") or ():
            log.records.append(
                ReplayRecord(
                    tick=int(raw["tick"]),
                    agent_id=str(raw["agent_id"]),
                    sequence=int(raw["sequence"]),
                    payload=_json_copy(raw.get("payload")),
                )
            )
        for raw in state.get("scripted_events") or ():
            log.scripted_events.append(
                ScriptedEvent(
                    tick=int(raw["tick"]),
                    event_id=str(raw["event_id"]),
                    payload=dict(_json_copy(raw.get("payload") or {})),
                    priority=int(raw.get("priority") or 0),
                )
            )
        for raw in state.get("snapshots") or ():
            log.snapshots.append(ReplaySnapshot(tick=int(raw["tick"]), state_hash=str(raw["state_hash"])))
        return log

    def _next_sequence(self, tick: int, agent_id: str) -> int:
        seqs = [r.sequence for r in self.records if r.tick == tick and r.agent_id == agent_id]
        return (max(seqs) + 1) if seqs else 0


def _records_for_tick(log: ReplayLog, tick: int) -> list[ReplayRecord]:
    due = [r for r in log.records if r.tick == tick]
    due.sort(key=lambda r: (r.agent_id, r.sequence))
    return due


def schedule_scripted(world: World, log: ReplayLog) -> None:
    """Queue logged scripted events. ``world`` is the live engine object."""
    for ev in log.scripted_events:
        world.clock.queue.schedule(ev.tick, _json_copy(ev.payload), priority=ev.priority)


def replay(world: World, log: ReplayLog) -> list[str]:
    """Apply logged inputs in ``(agent_id, sequence)`` order; return per-tick hashes.

    Scripted events are scheduled on ``world.clock.queue`` in log order before
    stepping. ``world`` is a fresh (or hash-compatible) engine object. Each
    returned string is ``World.state_hash()`` after that step (hex SHA-256).
    """
    schedule_scripted(world, log)
    hashes: list[str] = []
    end = log.horizon()
    while world.clock.tick < end:
        for rec in _records_for_tick(log, world.clock.tick):
            world.submit(rec.agent_id, rec.payload)
        world.step()
        hashes.append(world.state_hash())
    return hashes


def series_from_log(log: ReplayLog) -> dict[str, Any]:
    """Build the documented ``EXPORT_KEYS`` arrays from snapshots.

    Returns numpy arrays: ``tick`` (days, int64), ``state_hash`` (U64 hex),
    ``seed`` (int64), ``config_hash`` (U64 hex).
    """
    ticks = np.array([s.tick for s in log.snapshots], dtype=np.int64)
    hashes = np.array([s.state_hash for s in log.snapshots], dtype="U64")
    return {
        "tick": ticks,
        "state_hash": hashes,
        "seed": np.int64(log.seed),
        "config_hash": np.array(log.config_hash, dtype="U64"),
    }
