"""T7.02 — WorldManager isolation, auth, per-agent limits, starting-capital SFC."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from marketsim.api.schemas import AgentRegistration, WorldSpec
from marketsim.api.sessions import AgentLimits, AuthError, WorldManager
from marketsim.firms.accounts import agent_entity
from marketsim.ledger.sfc import assert_consistent


def test_world_isolation(config_dir: Path) -> None:
    mgr = WorldManager()
    a = mgr.create_world(WorldSpec(seed=11), config_dir)
    b = mgr.create_world(WorldSpec(seed=22), config_dir)
    assert a.world_id != b.world_id
    assert a.seed != b.seed
    hash_b = b.state_hash

    rec = mgr.register_agent(
        a.world_id,
        AgentRegistration(agent_id="alice", starting_capital=100.0),
    )
    entity = rec["entity"]
    assert entity == agent_entity("alice")
    assert mgr.ledger(a.world_id).position(entity, "DEP") == pytest.approx(100.0)

    mgr.step(a.world_id)
    b_after = mgr.status(b.world_id)
    assert b_after.state_hash == hash_b
    assert b_after.tick == b.tick
    assert mgr.world(a.world_id).clock.tick != mgr.world(b.world_id).clock.tick

    led_a = mgr.ledger(a.world_id)
    led_b = mgr.ledger(b.world_id)
    assert led_a.position(entity, "DEP") == pytest.approx(100.0)
    assert entity not in led_b.entities
    assert "BANKSYS" not in led_b.entities
    assert_consistent(led_a)
    assert_consistent(led_b)


def test_auth(config_dir: Path) -> None:
    mgr = WorldManager()
    world = mgr.create_world(WorldSpec(seed=7), config_dir)
    other = mgr.create_world(WorldSpec(seed=8), config_dir)
    rec = mgr.register_agent(world.world_id, AgentRegistration(agent_id="alice", role="agent"))
    token = rec["token"]
    expected = hashlib.sha256(f"{world.world_id}:alice:{world.seed}:agent".encode()).hexdigest()
    assert token == expected

    session = mgr.authenticate(world.world_id, token)
    assert session.agent_id == "alice"
    assert session.entity == agent_entity("alice")
    assert session.role == "agent"
    assert session.world_id == world.world_id

    obs = mgr.register_agent(world.world_id, AgentRegistration(agent_id="viewer", role="observer"))
    seen = mgr.authenticate(world.world_id, obs["token"])
    assert seen.role == "observer"
    assert seen.entity == agent_entity("viewer")

    with pytest.raises(AuthError):
        mgr.authenticate(world.world_id, "0" * 64)
    with pytest.raises(AuthError):
        mgr.authenticate(other.world_id, token)


def test_limits(config_dir: Path) -> None:
    mgr = WorldManager()
    world = mgr.create_world(WorldSpec(seed=3), config_dir)
    rec = mgr.register_agent(world.world_id, AgentRegistration(agent_id="trader"))
    token = rec["token"]
    for _ in range(AgentLimits.orders_per_tick):
        mgr.submit_order(world.world_id, token)
    with pytest.raises(AuthError):
        mgr.submit_order(world.world_id, token)
    mgr.step(world.world_id)
    mgr.submit_order(world.world_id, token)


def test_starting_capital_sfc(config_dir: Path) -> None:
    mgr = WorldManager()
    world = mgr.create_world(WorldSpec(seed=5), config_dir)
    rec = mgr.register_agent(
        world.world_id,
        AgentRegistration(agent_id="carol", starting_capital=40.0),
    )
    assert rec["entity"] == agent_entity("carol")
    led = mgr.ledger(world.world_id)
    assert_consistent(led)
    assert led.position(rec["entity"], "DEP") == pytest.approx(40.0)
    assert led.position("BANKSYS", "DEP") == pytest.approx(-40.0)
    assert led.period_flows.get(("BANKSYS", rec["entity"], "capital_transfer")) == pytest.approx(40.0)
