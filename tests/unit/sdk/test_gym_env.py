"""T7.08 — Gymnasium single-agent env: checker, seeding, autopilot mask."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from marketsim.firms.firm import LEVERS  # noqa: E402
from marketsim.sdk.gym_env import MASK_AGENT, MarketSimEnv  # noqa: E402


def _env(config_dir: Path) -> MarketSimEnv:
    return MarketSimEnv(config_dir, horizon=4, n_order_slots=2, seed=0)


def test_check_env(config_dir: Path) -> None:
    check_env(_env(config_dir), skip_render_check=True)


def test_seeding_deterministic(config_dir: Path) -> None:
    env_a = _env(config_dir)
    env_b = _env(config_dir)
    obs_a, _ = env_a.reset(seed=21)
    obs_b, _ = env_b.reset(seed=21)
    assert obs_a.keys() == obs_b.keys()
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key])


def test_masked_levers_ignored(config_dir: Path) -> None:
    env_a = _env(config_dir)
    env_b = _env(config_dir)
    env_a.reset(seed=5)
    env_b.reset(seed=5)
    act_default = env_a.idle_action()
    act_changed = deepcopy(act_default)
    # Autopilot mask: pricing stays 0. Changing the Box must not reach World.
    act_changed["firm"]["pricing"] = np.array([1.0], dtype=np.float32)
    assert int(act_changed["firm"]["mask"][LEVERS.index("pricing")]) == 0
    env_a.step(act_default)
    env_b.step(act_changed)
    assert env_a.client.state_hash() == env_b.client.state_hash()
    assert env_a.last_decision is not None and env_b.last_decision is not None
    assert env_a.last_decision == env_b.last_decision
    assert env_a.last_decision.posted_price is None
    applied = [m for m in env_a.world.modules if getattr(m, "name", None) == "firm_agent"]
    if applied:
        other = [m for m in env_b.world.modules if getattr(m, "name", None) == "firm_agent"]
        assert applied[0].applied == other[0].applied
        assert applied[0].applied[0][1].posted_price is None

    env_c = _env(config_dir)
    env_c.reset(seed=5)
    act_on = deepcopy(act_default)
    act_on["firm"]["mask"][LEVERS.index("pricing")] = MASK_AGENT
    act_on["firm"]["pricing"] = np.array([1.0], dtype=np.float32)
    env_c.step(act_on)
    assert env_c.last_decision is not None
    assert env_c.last_decision.posted_price is not None
