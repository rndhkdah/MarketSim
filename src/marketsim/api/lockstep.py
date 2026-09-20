"""Lockstep barrier and real-time pacing (T7.06 / §7.2).

The core never samples wall-clock: lateness is an explicit ``arrive_tick`` / ``late``
argument. ``ticks_per_second`` only configures a game-loop helper that tests may mock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marketsim.core.errors import ConfigError
from marketsim.world import World

# ticks/s — §7.2 real-time pacing. Constructor override; not a yaml key (T7.06 file list).
DEFAULT_TICKS_PER_SECOND = 1.0

# Lockstep no-op. Matches T7.01 ``SubmitPayload`` empty lists (unitless).
EMPTY_ACTION: dict[str, Any] = {"decisions": [], "orders": []}


def empty_action() -> dict[str, Any]:
    """A fresh empty action (no-op). ``decisions`` / ``orders`` are unitless lists."""
    return {"decisions": [], "orders": []}


@dataclass(slots=True)
class _Pending:
    """One buffered input. ``dest_tick`` is simulated days; ``sequence`` is client-local."""

    agent_id: str
    sequence: int
    payload: Any
    dest_tick: int


class LockstepBarrier:
    """Advance the world when every registered agent has acted or timed out.

    Actions are applied in ``(agent_id, sequence)`` order, then ``world.step()``.
    Timeouts are injected by ``timeout_missing()`` — tests call it; no wall-clock.
    """

    def __init__(self, world: World) -> None:
        self.world = world
        self._agents: list[str] = []
        self._known: set[str] = set()
        self._pending: list[_Pending] = []
        self._timed_out: set[str] = set()
        self._last_applied: list[tuple[str, int, Any]] = []

    @property
    def agents(self) -> tuple[str, ...]:
        """Registered agent ids, sorted (unitless)."""
        return tuple(self._agents)

    @property
    def current_tick(self) -> int:
        """Tick the barrier is collecting for (simulated days)."""
        return self.world.clock.tick

    @property
    def last_applied(self) -> tuple[tuple[str, int, Any], ...]:
        """Last flushed inbox as ``(agent_id, sequence, payload)`` in apply order."""
        return tuple(self._last_applied)

    def register(self, agent_id: str) -> None:
        """Register ``agent_id`` (unitless). Idempotent; iteration stays sorted."""
        if agent_id in self._known:
            return
        self._known.add(agent_id)
        self._agents.append(agent_id)
        self._agents.sort()

    def act(
        self,
        agent_id: str,
        payload: Any,
        *,
        sequence: int,
        target_tick: int | None = None,
        arrive_tick: int | None = None,
        late: bool = False,
    ) -> None:
        """Buffer one action. ``sequence`` is the client sequence (unitless).

        ``target_tick`` / ``arrive_tick`` are simulated days. ``late=True`` or
        ``arrive_tick > target_tick`` sends the input to the next tick (§7.2).
        If the barrier has already advanced past ``target_tick``, the input lands
        on the current (next available) tick.
        """
        self._require(agent_id)
        dest = self._destination_tick(target_tick, arrive_tick, late)
        self._pending.append(_Pending(agent_id, int(sequence), payload, dest))

    def timeout_missing(self) -> None:
        """Fill an empty action for every agent with no input on this tick (days)."""
        current = self.current_tick
        present = {p.agent_id for p in self._pending if p.dest_tick == current}
        for agent_id in self._agents:
            if agent_id in present:
                continue
            self._pending.append(_Pending(agent_id, 0, empty_action(), current))
            self._timed_out.add(agent_id)

    def maybe_advance(self) -> bool:
        """Step when every registered agent has acted or timed out.

        Returns True after applying the sorted inbox and ``world.step()``.
        Missing agents receive an empty action only after ``timeout_missing()``.
        """
        if not self._all_present():
            return False
        current = self.current_tick
        due = [p for p in self._pending if p.dest_tick == current]
        due.sort(key=lambda p: (p.agent_id, p.sequence))
        self._last_applied = [(p.agent_id, p.sequence, p.payload) for p in due]
        for agent_id, _sequence, payload in self._last_applied:
            self.world.submit(agent_id, payload)
        self.world.step()
        self._pending = [p for p in self._pending if p.dest_tick != current]
        self._timed_out.clear()
        return True

    def already_advanced(self, target_tick: int) -> bool:
        """True when the barrier has left ``target_tick`` (simulated days)."""
        return self.current_tick > int(target_tick)

    def to_state(self) -> dict[str, Any]:
        """Serialise registered agents and buffered inputs. Keys are sorted."""
        pending = sorted(self._pending, key=lambda p: (p.dest_tick, p.agent_id, p.sequence))
        return {
            "agents": list(self._agents),
            "pending": [
                {
                    "agent_id": p.agent_id,
                    "sequence": p.sequence,
                    "payload": p.payload,
                    "dest_tick": p.dest_tick,
                }
                for p in pending
            ],
            "timed_out": sorted(self._timed_out),
            "last_applied": [
                {"agent_id": a, "sequence": s, "payload": p} for a, s, p in self._last_applied
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], world: World) -> LockstepBarrier:
        """Restore a barrier bound to ``world``. ``world`` is the live engine object."""
        barrier = cls(world)
        for agent_id in state.get("agents") or ():
            barrier.register(str(agent_id))
        for raw in state.get("pending") or ():
            barrier._pending.append(
                _Pending(
                    agent_id=str(raw["agent_id"]),
                    sequence=int(raw["sequence"]),
                    payload=raw["payload"],
                    dest_tick=int(raw["dest_tick"]),
                )
            )
        barrier._timed_out = {str(a) for a in state.get("timed_out") or ()}
        barrier._last_applied = [
            (str(row["agent_id"]), int(row["sequence"]), row["payload"])
            for row in state.get("last_applied") or ()
        ]
        return barrier

    def _require(self, agent_id: str) -> None:
        if agent_id not in self._known:
            raise ConfigError(f"unknown agent {agent_id!r}")

    def _all_present(self) -> bool:
        current = self.current_tick
        present = {p.agent_id for p in self._pending if p.dest_tick == current}
        return all(agent_id in present for agent_id in self._agents)

    def _destination_tick(
        self,
        target_tick: int | None,
        arrive_tick: int | None,
        late: bool,
    ) -> int:
        """Resolve the landing tick (days). Late inputs go to the next tick (§7.2)."""
        current = self.current_tick
        dest = current if target_tick is None else int(target_tick)
        if dest < current:
            return current
        if late or (arrive_tick is not None and arrive_tick > dest):
            dest = dest + 1  # next tick (§7.2)
            return current if dest < current else dest
        return dest


class RealtimePacer:
    """Tag inputs with a target tick and optional lateness; pace a game loop.

    ``ticks_per_second`` is ticks/s for ``interval_s()``. Tests inject lateness via
    ``arrive_tick`` / ``late`` — this class does not call ``time.time()``.
    """

    def __init__(
        self,
        barrier: LockstepBarrier,
        *,
        ticks_per_second: float = DEFAULT_TICKS_PER_SECOND,
    ) -> None:
        if ticks_per_second <= 0:
            raise ConfigError("ticks_per_second must be > 0 (ticks/s)")
        self.barrier = barrier
        self.ticks_per_second = float(ticks_per_second)

    def act(
        self,
        agent_id: str,
        payload: Any,
        *,
        sequence: int,
        target_tick: int | None = None,
        arrive_tick: int | None = None,
        late: bool = False,
    ) -> None:
        """Forward one input. ``target_tick`` / ``arrive_tick`` are simulated days."""
        self.barrier.act(
            agent_id,
            payload,
            sequence=sequence,
            target_tick=target_tick,
            arrive_tick=arrive_tick,
            late=late,
        )

    def timeout_missing(self) -> None:
        """Delegate: fill empty actions for agents silent on this tick (days)."""
        self.barrier.timeout_missing()

    def maybe_advance(self) -> bool:
        """Delegate: True when the barrier applied the inbox and stepped."""
        return self.barrier.maybe_advance()

    def interval_s(self) -> float:
        """Wall-clock period (s) at ``ticks_per_second`` (ticks/s). Not used by the core."""
        return 1.0 / self.ticks_per_second

    def due_ticks(self, elapsed_s: float) -> int:
        """Ticks due after ``elapsed_s`` seconds (s). Caller supplies the clock."""
        return int(elapsed_s * self.ticks_per_second)

    def pace(self, sleep: Any | None = None) -> float:
        """Optional game-loop pause. ``sleep`` is ``(s) -> None``; default is a no-op.

        Returns the interval (s). Simulation state is unchanged.
        """
        interval = self.interval_s()
        if sleep is not None:
            sleep(interval)
        return interval

    def to_state(self) -> dict[str, Any]:
        """Serialise pacing config and the wrapped barrier."""
        return {
            "ticks_per_second": self.ticks_per_second,
            "barrier": self.barrier.to_state(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], world: World) -> RealtimePacer:
        """Restore a pacer + barrier bound to ``world``."""
        barrier = LockstepBarrier.from_state(state.get("barrier") or {}, world)
        rate = float(state.get("ticks_per_second", DEFAULT_TICKS_PER_SECOND))
        return cls(barrier, ticks_per_second=rate)
