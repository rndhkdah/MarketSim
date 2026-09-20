"""T7.10 — event-sourced replay, save/load continue, export schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from marketsim.api.lockstep import LockstepBarrier
from marketsim.api.replay import (
    EXPORT_DTYPES,
    EXPORT_KEYS,
    EXPORT_UNITS,
    create_world,
    load_snapshot,
    read_replay_log,
    write_export_npz,
    write_replay_log,
    write_snapshot,
)
from marketsim.core.module import Phase
from marketsim.scenarios.replay import ReplayLog, hash_overrides, replay
from marketsim.world import RandomWalkModule, TickContext, World


class _InboxMod:
    """Persist ingested actions so ``World.state_hash()`` sees apply order."""

    name = "t710_inbox"

    def __init__(self) -> None:
        self.applied: list[tuple[int, str, Any]] = []

    def reset(self, ctx: TickContext) -> None:
        del ctx
        self.applied = []

    def on_phase(self, ctx: TickContext, phase: Phase) -> None:
        if phase is Phase.INGEST:
            tick = ctx.tick
            for agent_id, acts in sorted(ctx.world._inbox.items()):
                for act in acts:
                    self.applied.append((tick, agent_id, act))

    def to_state(self) -> dict[str, Any]:
        return {"applied": [list(row) for row in self.applied]}

    def from_state(self, state: dict[str, Any]) -> None:
        self.applied = [(int(t), str(a), p) for t, a, p in state.get("applied", ())]


def _modules() -> list[Any]:
    return [_InboxMod(), RandomWalkModule()]


def _harness(
    config_dir: Path, seed: int, agent_ids: tuple[str, ...]
) -> tuple[World, LockstepBarrier, ReplayLog]:
    world = World.create(config_dir, seed=seed, modules=_modules())
    barrier = LockstepBarrier(world)
    for agent_id in agent_ids:
        barrier.register(agent_id)
    return world, barrier, ReplayLog(seed=seed)


def test_replay_reproduces_per_tick_hashes(config_dir: Path, tmp_path: Path) -> None:
    """Gate 2 (replay half): the log reproduces every per-tick hash."""
    seed = 11
    world, barrier, log = _harness(config_dir, seed, ("alice", "bob"))
    event = {"kind": "scripted", "event_id": "chip_shortage_2020", "magnitude": 0.5}
    log.append_scripted(2, "chip_shortage_2020", event)
    world.clock.queue.schedule(2, event, priority=0)

    actions: list[list[tuple[str, int, dict[str, int]]]] = [
        [("bob", 1, {"n": 2}), ("alice", 2, {"n": 1}), ("alice", 1, {"n": 3})],
        [("alice", 1, {"n": 4}), ("bob", 0, {"n": 5})],
        [("bob", 2, {"n": 7}), ("alice", 1, {"n": 6}), ("bob", 1, {"n": 8})],
        [("alice", 1, {"n": 9}), ("bob", 1, {"n": 10})],
    ]
    live_hashes: list[str] = []
    for batch in actions:
        for agent_id, sequence, payload in batch:
            barrier.act(agent_id, payload, sequence=sequence)
        assert barrier.maybe_advance()
        log.record_applied(world, list(barrier.last_applied))
        live_hashes.append(world.state_hash())

    assert [s.state_hash for s in log.snapshots] == live_hashes
    clone = create_world(log, config_dir, modules=_modules())
    replayed = replay(clone, log)
    assert replayed == live_hashes
    assert clone.state_hash() == world.state_hash()
    assert clone.clock.tick == world.clock.tick

    shuffled = ReplayLog(seed=seed)
    shuffled.append_scripted(2, "chip_shortage_2020", event)
    for rec in reversed(log.records):
        shuffled.append(rec.tick, rec.agent_id, rec.payload, sequence=rec.sequence)
    for snap in log.snapshots:
        shuffled.snapshots.append(snap)
    shuffled.n_ticks = log.n_ticks
    again = create_world(shuffled, config_dir, modules=_modules())
    assert replay(again, shuffled) == live_hashes

    path = tmp_path / "run.json"
    write_replay_log(path, log)
    loaded = read_replay_log(path)
    assert loaded.config_hash == log.config_hash
    from_disk = create_world(loaded, config_dir, modules=_modules())
    assert replay(from_disk, loaded) == live_hashes


def test_save_load_continue_equals_uninterrupted(config_dir: Path, tmp_path: Path) -> None:
    """Save → load → continue matches an uninterrupted run."""
    seed = 9
    live = World.create(config_dir, seed=seed, modules=_modules())
    for step in range(8):
        live.submit("alice", {"t": step, "k": 1})
        live.submit("bob", {"t": step, "k": 2})
        live.step()
    mid_hash = live.state_hash()
    snap = tmp_path / "world.json"
    write_snapshot(snap, live)

    for step in range(8, 16):
        live.submit("alice", {"t": step, "k": 1})
        live.submit("bob", {"t": step, "k": 2})
        live.step()
    live_hash = live.state_hash()

    restored = load_snapshot(snap, config_dir, modules=_modules())
    assert restored.state_hash() == mid_hash
    for step in range(8, 16):
        restored.submit("alice", {"t": step, "k": 1})
        restored.submit("bob", {"t": step, "k": 2})
        restored.step()
    assert restored.state_hash() == live_hash
    assert restored.clock.tick == live.clock.tick


def test_export_schema_stable(config_dir: Path, tmp_path: Path) -> None:
    """``.npz`` keys and dtypes are the documented export schema."""
    assert EXPORT_KEYS == ("tick", "state_hash", "seed", "config_hash")
    assert set(EXPORT_KEYS) == set(EXPORT_DTYPES) == set(EXPORT_UNITS)
    assert hash_overrides({}) == hash_overrides({})
    assert hash_overrides({"b": 1, "a": 2}) == hash_overrides({"a": 2, "b": 1})
    assert hash_overrides({}) == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    assert hash_overrides({"markets.turnover": 0.004}) != hash_overrides({})

    world, barrier, log = _harness(config_dir, 3, ("alice", "bob"))
    log.overrides = {"markets.turnover": 0.004}
    log.config_hash = hash_overrides(log.overrides)
    log.randomisation_draws = {"markets.turnover": 0.004, "mispricing.phi_s": 0.97}
    for i in range(5):
        barrier.act("alice", {"n": i}, sequence=1)
        barrier.act("bob", {"n": -i}, sequence=1)
        assert barrier.maybe_advance()
        log.record_applied(world, list(barrier.last_applied))

    path = tmp_path / "series.npz"
    write_export_npz(path, log)
    loaded = np.load(path, allow_pickle=False)
    assert tuple(sorted(loaded.files)) == tuple(sorted(EXPORT_KEYS))
    assert set(loaded.files) == set(EXPORT_KEYS)
    assert loaded["tick"].dtype == np.dtype(np.int64)
    assert loaded["seed"].dtype == np.dtype(np.int64)
    assert loaded["state_hash"].dtype == np.dtype("U64")
    assert loaded["config_hash"].dtype == np.dtype("U64")
    assert loaded["tick"].tolist() == [s.tick for s in log.snapshots]
    assert loaded["state_hash"].tolist() == [s.state_hash for s in log.snapshots]
    assert int(loaded["seed"]) == log.seed
    assert loaded["config_hash"].item() == log.config_hash
    for key in EXPORT_KEYS:
        assert key in EXPORT_UNITS
        assert EXPORT_DTYPES[key]
    loaded.close()
