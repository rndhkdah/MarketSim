"""T7.06 — lockstep barrier, gate 2, timeout empty action, late real-time input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from marketsim.api.lockstep import EMPTY_ACTION, LockstepBarrier, RealtimePacer, empty_action
from marketsim.core.module import Phase
from marketsim.world import TickContext, World


class _Record:
    """Persist ingested actions so ``World.state_hash()`` sees apply order."""

    name = "t706_record"

    def __init__(self) -> None:
        self.log: list[tuple[int, str, Any]] = []

    def reset(self, ctx: TickContext) -> None:
        del ctx
        self.log = []

    def on_phase(self, ctx: TickContext, phase: Phase) -> None:
        if phase is Phase.INGEST:
            tick = ctx.tick
            for agent_id, acts in sorted(ctx.world._inbox.items()):
                for act in acts:
                    self.log.append((tick, agent_id, act))

    def to_state(self) -> dict[str, Any]:
        return {"log": [list(row) for row in self.log]}

    def from_state(self, state: dict[str, Any]) -> None:
        self.log = [(int(t), str(a), p) for t, a, p in state.get("log", ())]

    def on_tick(self, tick: int) -> list[tuple[str, Any]]:
        return [(a, p) for t, a, p in self.log if t == tick]


def _harness(config_dir: Path, seed: int, agent_ids: tuple[str, ...]) -> tuple[World, _Record, LockstepBarrier]:
    rec = _Record()
    world = World.create(config_dir, seed=seed, modules=[rec])
    barrier = LockstepBarrier(world)
    for agent_id in agent_ids:
        barrier.register(agent_id)
    return world, rec, barrier


def test_gate2_permuted_submission_same_hash(config_dir: Path) -> None:
    """Gate 2: same seed + same actions → same hash regardless of submit order."""
    actions: list[tuple[str, int, dict[str, int]]] = [
        ("carol", 1, {"n": 1}),
        ("alice", 2, {"n": 2}),
        ("alice", 1, {"n": 3}),
        ("bob", 0, {"n": 4}),
    ]
    orders = (
        actions,
        list(reversed(actions)),
        [actions[i] for i in (2, 0, 3, 1)],
    )
    hashes: list[str] = []
    logs: list[list[tuple[int, str, Any]]] = []
    for order in orders:
        world, rec, barrier = _harness(config_dir, 11, ("alice", "bob", "carol"))
        for agent_id, sequence, payload in order:
            barrier.act(agent_id, payload, sequence=sequence)
        assert barrier.maybe_advance()
        hashes.append(world.state_hash())
        logs.append(list(rec.log))
    assert hashes[0] == hashes[1] == hashes[2]
    expected = [
        (0, "alice", {"n": 3}),
        (0, "alice", {"n": 2}),
        (0, "bob", {"n": 4}),
        (0, "carol", {"n": 1}),
    ]
    for log in logs:
        assert log == expected


def test_timeout_empty_action(config_dir: Path) -> None:
    world, rec, barrier = _harness(config_dir, 2, ("alice", "bob"))
    barrier.act("alice", {"id": "present"}, sequence=1)
    assert barrier.maybe_advance() is False
    barrier = LockstepBarrier.from_state(barrier.to_state(), world)
    barrier.timeout_missing()
    assert barrier.maybe_advance() is True
    assert rec.on_tick(0) == [
        ("alice", {"id": "present"}),
        ("bob", empty_action()),
    ]
    assert barrier.last_applied == (
        ("alice", 1, {"id": "present"}),
        ("bob", 0, EMPTY_ACTION),
    )


def test_late_realtime_input_lands_next_tick(config_dir: Path) -> None:
    world, rec, barrier = _harness(config_dir, 5, ("alice", "bob"))
    pacer = RealtimePacer(barrier, ticks_per_second=2.0)
    assert pacer.interval_s() == 0.5

    pacer.act("alice", {"id": "a0"}, sequence=1, target_tick=0)
    pacer.act("bob", {"id": "b0"}, sequence=1, target_tick=0)
    assert pacer.maybe_advance()
    assert world.clock.tick == 1
    assert rec.on_tick(0) == [("alice", {"id": "a0"}), ("bob", {"id": "b0"})]

    # Intended for tick 0; arrive_tick / late mark it after the barrier advanced.
    pacer.act(
        "bob",
        {"id": "b-late"},
        sequence=2,
        target_tick=0,
        arrive_tick=1,
        late=True,
    )
    pacer.act("alice", {"id": "a1"}, sequence=1, target_tick=1)
    assert pacer.maybe_advance()
    assert ("bob", {"id": "b-late"}) not in rec.on_tick(0)
    assert rec.on_tick(1) == [("alice", {"id": "a1"}), ("bob", {"id": "b-late"})]
