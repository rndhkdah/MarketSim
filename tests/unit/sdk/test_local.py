"""T7.03 — in-process client method surface and World-call identity."""

from __future__ import annotations

from pathlib import Path

from marketsim.firms.levers import FirmDecision
from marketsim.sdk.local import LOCAL_METHODS, LocalClient
from marketsim.world import FirmAgentModule, World

# REST SDK (T7.07) must expose the same public names as LOCAL_METHODS.


def test_method_parity_introspection() -> None:
    """Every LOCAL_METHODS name exists on LocalClient (T7.07 must match this set)."""
    assert isinstance(LOCAL_METHODS, frozenset)
    missing = sorted(name for name in LOCAL_METHODS if not hasattr(LocalClient, name))
    assert missing == []
    for name in LOCAL_METHODS:
        assert callable(getattr(LocalClient, name))


def test_results_identical_to_direct_world_calls(config_dir: Path) -> None:
    decision = FirmDecision(
        firm_id="acme",
        operator="alice",
        posted_price={("INDUSTRIAL", "AUTOS"): 1.0},
        target_output={("INDUSTRIAL", "AUTOS"): 4.0},
    )
    world = World.create(config_dir, seed=7, modules=[FirmAgentModule()])
    via = World.create(config_dir, seed=7, modules=[FirmAgentModule()])
    client = LocalClient(via)
    world.submit("alice", decision)
    client.submit("alice", decision)
    world.step()
    client.step()
    assert client.observe("alice")["tick"] == world.observe("alice")["tick"]
    assert client.state_hash() == world.state_hash()
