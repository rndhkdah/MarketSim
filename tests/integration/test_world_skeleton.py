from __future__ import annotations

from marketsim.world import RandomWalkModule, World


def test_empty_world_1000_ticks(config_dir) -> None:
    w = World.create(config_dir, seed=1)
    w.step(1000)
    assert w.clock.tick == 1000
    assert w.state_hash()


def test_equal_seeds_equal_hash(config_dir) -> None:
    hashes = []
    for _ in range(2):
        w = World.create(config_dir, seed=3, modules=[RandomWalkModule()])
        tick_hashes = []
        for _t in range(50):
            w.step()
            tick_hashes.append(w.state_hash())
        hashes.append(tick_hashes)
    assert hashes[0] == hashes[1]


def test_different_seeds_diverge(config_dir) -> None:
    a = World.create(config_dir, seed=1, modules=[RandomWalkModule()])
    b = World.create(config_dir, seed=2, modules=[RandomWalkModule()])
    a.step(40)
    b.step(40)
    assert a.state_hash() != b.state_hash()


def test_save_load_mid_run(config_dir, tmp_path) -> None:
    live = World.create(config_dir, seed=9, modules=[RandomWalkModule()])
    live.step(25)
    path = tmp_path / "world.json"
    live.save(path)
    live.step(25)
    restored = World.load(path, config_dir, modules=[RandomWalkModule()])
    restored.step(25)
    assert restored.state_hash() == live.state_hash()
