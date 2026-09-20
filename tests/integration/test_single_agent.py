"""T5.18 — in-process FirmDecision loop; determinism with an agent."""

from __future__ import annotations

import pytest

from marketsim.core.config import default_config_dir
from marketsim.firms.levers import FirmDecision
from marketsim.world import FirmAgentModule, World


def _dec(t: int) -> FirmDecision:
    return FirmDecision(
        firm_id="acme",
        operator="alice",
        posted_price={("INDUSTRIAL", "AUTOS"): 1.0},
        target_output={("INDUSTRIAL", "AUTOS"): 4.0 + t % 2},
        expand_capacity={("INDUSTRIAL", "AUTOS"): 0.0},
    )


def test_scripted_agent_deterministic() -> None:
    def play(seed: int) -> str:
        w = World.create(default_config_dir(), seed=seed, modules=[FirmAgentModule()])
        for t in range(24):
            w.submit("alice", _dec(t))
            w.step(1)
        return w.state_hash()

    assert play(3) == play(3)


def test_observe_sees_submit() -> None:
    w = World.create(default_config_dir(), seed=1, modules=[FirmAgentModule()])
    w.submit("alice", _dec(0))
    w.step(1)
    obs = w.observe("alice")
    assert obs["last_decision"] == "acme"


@pytest.mark.slow
def test_random_policy_smoke() -> None:
    w = World.create(default_config_dir(), seed=2, modules=[FirmAgentModule()])
    rng = w.rng.stream("agent.random")
    for _ in range(200):
        w.submit(
            "alice",
            FirmDecision(
                firm_id="acme",
                operator="alice",
                posted_price={("INDUSTRIAL", "AUTOS"): 1.0 + 0.05 * float(rng.normal())},
            ),
        )
        w.step(1)
    assert w.clock.tick == 200
