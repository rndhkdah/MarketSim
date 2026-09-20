"""T7.12 — spawn VectorEnv: per-worker seeds, 0.75 scaling, close joins."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from marketsim.sdk.gym_env import MarketSimEnv  # noqa: E402
from marketsim.sdk.vector_env import VectorEnv  # noqa: E402

_ROOT_SEED = 21  # unitless; matches SeedSequence.spawn in the card
_N = 2
_HORIZON = 16  # days; warmup + timed idle steps stay unterminated
_MIN_RATIO = 0.75  # N * t_single / t_vec; do not loosen


def _worker_seeds(root_seed: int, n: int) -> tuple[int, ...]:
    """Card formula: SeedSequence(root).spawn(n) → generate_state(1)[0] (unitless)."""
    spawned = np.random.SeedSequence(int(root_seed)).spawn(int(n))
    return tuple(int(ss.generate_state(1)[0]) for ss in spawned)


def _obs_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def _standalone_first_step(config_dir: Path, seed: int) -> dict[str, np.ndarray]:
    env = MarketSimEnv(config_dir, seed=seed, horizon=_HORIZON)
    env.reset()
    obs, _, _, _, _ = env.step(env.idle_action())
    env.close()
    return obs


def test_per_worker_determinism(config_dir: Path) -> None:
    seeds = _worker_seeds(_ROOT_SEED, _N)
    vecs: list[VectorEnv] = []
    stepped: list[list[dict[str, np.ndarray]]] = []
    try:
        for _ in range(2):
            vec = VectorEnv(config_dir, n=_N, seed=_ROOT_SEED, horizon=_HORIZON)
            vecs.append(vec)
            vec.reset()
            obs, _, _, _, _ = vec.step(vec.idle_action())
            stepped.append(obs)
        for i in range(_N):
            _obs_equal(stepped[0][i], stepped[1][i])
            _obs_equal(stepped[0][i], _standalone_first_step(config_dir, seeds[i]))
    finally:
        for vec in vecs:
            vec.close()


def _time_single(config_dir: Path, n_steps: int) -> float:
    env = MarketSimEnv(config_dir, seed=_worker_seeds(_ROOT_SEED, 1)[0], horizon=_HORIZON)
    env.reset()
    idle = env.idle_action()
    env.step(idle)  # warmup: exclude first-step setup
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step(idle)
    elapsed = time.perf_counter() - t0
    env.close()
    return elapsed


def _time_vec(config_dir: Path, n: int, n_steps: int) -> float:
    vec = VectorEnv(config_dir, n=n, seed=_ROOT_SEED, horizon=_HORIZON)
    vec.reset()
    idle = vec.idle_action()
    vec.step(idle)  # warmup: exclude spawn / import
    t0 = time.perf_counter()
    for _ in range(n_steps):
        vec.step(idle)
    elapsed = time.perf_counter() - t0
    vec.close()
    return elapsed


def test_throughput_scales(config_dir: Path) -> None:
    n = _N
    inner = 8  # ticks in the timed region
    ratio = 0.0
    t_single = t_vec = 0.0
    for inner in (8, 32, 128):
        t_single = _time_single(config_dir, inner)
        t_vec = _time_vec(config_dir, n, inner)
        ratio = n * t_single / t_vec
        if ratio >= _MIN_RATIO:
            break
    print(f"T7.12 throughput n={n} inner={inner} t_single={t_single:.6f}s t_vec={t_vec:.6f}s ratio={ratio:.4f}")
    assert ratio >= _MIN_RATIO, (
        f"N*t_single/t_vec={ratio:.4f} < {_MIN_RATIO} (n={n} inner={inner} "
        f"t_single={t_single:.6f}s t_vec={t_vec:.6f}s)"
    )


def test_close_joins_workers(config_dir: Path) -> None:
    vec = VectorEnv(config_dir, n=_N, seed=_ROOT_SEED, horizon=_HORIZON)
    procs = list(vec.processes)
    assert procs and all(p.is_alive() for p in procs)
    vec.reset()
    vec.close()
    assert all(not p.is_alive() for p in procs)
    assert vec.closed
