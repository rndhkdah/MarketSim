"""Spawn process-pool batch evaluation (T9.03 / §8.4).

Each job is one ``MarketSimEnv`` (idle policy) in a child started with the
``spawn`` context — ``World`` is not fork-safe. Seeds come from
``np.random.SeedSequence(root).spawn(n)`` then ``int(ss.generate_state(1)[0])``
when the caller passes a root seed. No global RNG; no wall-clock in results.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marketsim.sdk.gym_env import MarketSimEnv

_SPAWN = "spawn"


@dataclass(frozen=True)
class EvalJob:
    """One evaluation. ``ticks`` / ``horizon`` are days; ``seed`` is unitless."""

    job_id: str
    seed: int
    ticks: int
    horizon: int | None = None


@dataclass(frozen=True)
class EvalResult:
    """Job outcome. ``tick`` is days; ``state_hash`` is hex SHA-256."""

    job_id: str
    seed: int
    ticks: int
    tick: int
    state_hash: str


def job_seeds(root_seed: int, n: int) -> tuple[int, ...]:
    """Per-job ``World`` seeds (unitless). Same formula as T7.12 ``worker_seeds``."""
    spawned = np.random.SeedSequence(int(root_seed)).spawn(int(n))
    return tuple(int(ss.generate_state(1)[0]) for ss in spawned)


def run_eval_job(config_dir: str | Path, job: EvalJob) -> EvalResult:
    """Idle ``MarketSimEnv`` for ``job.ticks`` days. ``config_dir`` is a filesystem path."""
    horizon = int(job.horizon) if job.horizon is not None else int(job.ticks) + 2
    env = MarketSimEnv(config_dir, seed=int(job.seed), horizon=horizon)
    try:
        env.reset(seed=int(job.seed))
        idle = env.idle_action()
        for _ in range(int(job.ticks)):
            env.step(idle)
        world = env._world
        assert world is not None
        return EvalResult(
            job_id=str(job.job_id),
            seed=int(job.seed),
            ticks=int(job.ticks),
            tick=int(world.clock.tick),
            state_hash=world.state_hash(),
        )
    finally:
        env.close()


def _worker(payload: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    config_dir, raw = payload
    job = EvalJob(
        job_id=str(raw["job_id"]),
        seed=int(raw["seed"]),
        ticks=int(raw["ticks"]),
        horizon=None if raw.get("horizon") is None else int(raw["horizon"]),
    )
    result = run_eval_job(config_dir, job)
    return {
        "job_id": result.job_id,
        "seed": result.seed,
        "ticks": result.ticks,
        "tick": result.tick,
        "state_hash": result.state_hash,
    }


def batch_eval(
    config_dir: str | Path,
    jobs: Sequence[EvalJob],
    *,
    n_workers: int = 2,
) -> tuple[EvalResult, ...]:
    """Run ``jobs`` on a spawn pool. Results keep input order. ``n_workers`` is a count."""
    if int(n_workers) < 1:
        raise ValueError("n_workers must be >= 1")
    root = str(Path(config_dir))
    payloads = [
        (
            root,
            {
                "job_id": job.job_id,
                "seed": int(job.seed),
                "ticks": int(job.ticks),
                "horizon": job.horizon,
            },
        )
        for job in jobs
    ]
    if not payloads:
        return ()
    ctx = mp.get_context(_SPAWN)
    with ctx.Pool(processes=min(int(n_workers), len(payloads))) as pool:
        rows = pool.map(_worker, payloads)
    return tuple(
        EvalResult(
            job_id=str(row["job_id"]),
            seed=int(row["seed"]),
            ticks=int(row["ticks"]),
            tick=int(row["tick"]),
            state_hash=str(row["state_hash"]),
        )
        for row in rows
    )


def make_jobs(root_seed: int, n: int, ticks: int) -> tuple[EvalJob, ...]:
    """``n`` jobs with T7.12 seeds. ``ticks`` is days; ``root_seed`` is unitless."""
    seeds = job_seeds(root_seed, n)
    return tuple(
        EvalJob(job_id=f"j{i:02d}", seed=seed, ticks=int(ticks))
        for i, seed in enumerate(seeds)
    )
