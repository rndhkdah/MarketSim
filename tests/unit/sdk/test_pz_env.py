"""T7.09 — PettingZoo parallel env: API test, mid-episode bankruptcy, seeding."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from pettingzoo.test import parallel_api_test  # noqa: E402

from marketsim.sdk.gym_env import net_worth_cr  # noqa: E402
from marketsim.sdk.pz_env import MarketSimParallelEnv  # noqa: E402


def _env(
    config_dir: Path,
    *,
    horizon: int = 6,
    seed: int = 0,
    agent_ids: tuple[str, ...] = ("alice", "bob"),
) -> MarketSimParallelEnv:
    return MarketSimParallelEnv(
        config_dir,
        horizon=horizon,
        n_order_slots=2,
        seed=seed,
        agent_ids=agent_ids,
    )


def test_parallel_api(config_dir: Path) -> None:
    parallel_api_test(_env(config_dir), num_cycles=10)


def test_mid_episode_bankruptcy_removes_agent(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _env(config_dir, horizon=8)
    env.reset(seed=3)
    idle = {agent: env.idle_action(agent) for agent in env.agents}
    env.step(idle)
    assert env.agents == ["alice", "bob"]

    def negative_nw_for_alice(raw: Mapping[str, Any]) -> float:
        if raw.get("agent_id") == "alice":
            return -1.0  # cr; strictly negative ⇒ bankrupt
        return net_worth_cr(raw)

    monkeypatch.setattr("marketsim.sdk.pz_env.net_worth_cr", negative_nw_for_alice)

    obs, _rew, terminated, _trunc, infos = env.step(idle)
    assert terminated["alice"] is True
    assert terminated["bob"] is False
    assert "alice" in obs and "alice" in infos
    assert "alice" not in env.agents
    assert env.agents == ["bob"]

    follow = {agent: env.idle_action(agent) for agent in env.agents}
    obs2, rew2, term2, trunc2, infos2 = env.step(follow)
    assert "alice" not in env.agents
    assert "alice" not in obs2
    assert "alice" not in rew2
    assert "alice" not in term2
    assert "alice" not in trunc2
    assert "alice" not in infos2
    assert "bob" in obs2 and "bob" in infos2
    assert env.agents == ["bob"]


def test_seeding_deterministic(config_dir: Path) -> None:
    env_a = _env(config_dir)
    env_b = _env(config_dir)
    obs_a, _ = env_a.reset(seed=21)
    obs_b, _ = env_b.reset(seed=21)
    assert sorted(obs_a) == sorted(obs_b)
    for agent in sorted(obs_a):
        assert obs_a[agent].keys() == obs_b[agent].keys()
        for key in obs_a[agent]:
            np.testing.assert_array_equal(obs_a[agent][key], obs_b[agent][key])
