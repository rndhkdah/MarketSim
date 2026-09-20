"""T9.02 — filesystem world registry, snapshots, and columnar export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marketsim.scenarios.replay import EXPORT_KEYS, ReplayLog
from marketsim.storage import FileStore, WorldRegistry
from marketsim.world import World


def test_registry_roundtrip_and_sorted_list(config_dir: Path, tmp_path: Path) -> None:
    reg = WorldRegistry(tmp_path)
    a = World.create(config_dir, seed=3)
    b = World.create(config_dir, seed=5)
    a.step(2)
    b.step(1)
    log = ReplayLog(seed=3)
    log.record_snapshot(a)
    rec_b = reg.register("w-b", b)
    rec_a = reg.register("w-a", a, log=log)
    assert rec_a.tick == 2
    assert rec_a.seed == 3
    assert rec_a.state_hash == a.state_hash()
    ids = [row.world_id for row in reg.list_worlds()]
    assert ids == ["w-a", "w-b"]
    loaded = reg.load_world("w-a", config_dir)
    assert loaded.state_hash() == a.state_hash()
    assert loaded.clock.tick == 2
    assert rec_b.tick == 1


def test_store_export_npz_keys(config_dir: Path, tmp_path: Path) -> None:
    world = World.create(config_dir, seed=1)
    world.step(3)
    log = ReplayLog(seed=1)
    log.record_snapshot(world)
    store = FileStore(tmp_path, "w1")
    store.write_log(log)
    dest = store.write_export(log)
    data = np.load(dest)
    assert tuple(data.files) == EXPORT_KEYS or set(data.files) == set(EXPORT_KEYS)
    assert int(data["tick"][-1]) == 3
    assert str(data["state_hash"][-1]) == world.state_hash()
    assert store.list_snapshot_ticks() == ()
    store.write_world(world)
    assert store.list_snapshot_ticks() == (3,)
