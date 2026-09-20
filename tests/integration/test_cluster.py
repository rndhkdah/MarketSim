"""T9.03 — spawn batch eval: per-job determinism vs standalone."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gymnasium")

from marketsim.sdk.cluster import EvalJob, batch_eval, job_seeds, run_eval_job  # noqa: E402

_ROOT = 21
_TICKS = 4  # days


def test_job_seeds_match_seedsequence() -> None:
    spawned = __import__("numpy").random.SeedSequence(_ROOT).spawn(3)
    assert job_seeds(_ROOT, 3) == tuple(int(ss.generate_state(1)[0]) for ss in spawned)


def test_batch_matches_standalone(config_dir: Path) -> None:
    seeds = job_seeds(_ROOT, 2)
    jobs = tuple(EvalJob(job_id=f"j{i}", seed=s, ticks=_TICKS) for i, s in enumerate(seeds))
    batch = batch_eval(config_dir, jobs, n_workers=2)
    assert len(batch) == 2
    for job, row in zip(jobs, batch, strict=True):
        solo = run_eval_job(config_dir, job)
        assert row.state_hash == solo.state_hash
        assert row.tick == _TICKS
        assert len(row.state_hash) == 64
