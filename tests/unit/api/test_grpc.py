"""T9.06 — optional gRPC transport: methods, closed Observation, in-process identity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from marketsim.api.grpc.codec import decode_model, decode_payload, encode_model, encode_payload
from marketsim.api.grpc.servicer import MarketSimServicer, authorization_metadata
from marketsim.api.schemas import AgentRegistration, Observation, WorldSpec
from marketsim.api.sessions import AuthError, WorldManager
from marketsim.ledger.sfc import assert_consistent
from marketsim.scenarios.randomise import HIDDEN_KEYS
from marketsim.sdk.local import LOCAL_METHODS, LocalClient
from marketsim.world import World


def test_importing_package_does_not_import_grpc() -> None:
    before = {name for name in sys.modules if name == "grpc" or name.startswith("grpc.")}
    import marketsim.api.grpc as pkg

    assert isinstance(pkg.GRPC_AVAILABLE, bool)
    encode_payload({"tick": 0})
    after = {name for name in sys.modules if name == "grpc" or name.startswith("grpc.")}
    assert after == before


def test_every_local_method_is_on_the_servicer() -> None:
    missing = sorted(name for name in LOCAL_METHODS if not hasattr(MarketSimServicer, name))
    assert missing == []
    for name in LOCAL_METHODS:
        assert callable(getattr(MarketSimServicer, name))
    assert MarketSimServicer.methods == LOCAL_METHODS


def test_observation_round_trip_rejects_hidden_xi() -> None:
    raw = {"tick": 0, "agent_id": "alice", "xi": 0.2}
    with pytest.raises(ValidationError):
        Observation.model_validate(raw)
    with pytest.raises(ValidationError):
        decode_model(encode_payload(raw), Observation)
    clean = Observation(tick=0, agent_id="alice")
    assert decode_model(encode_model(clean), Observation) == clean
    dumped = decode_payload(encode_model(clean))
    assert "xi" not in dumped
    assert set(dumped).isdisjoint(HIDDEN_KEYS)


def test_step_observe_hash_identical_to_local_client(config_dir: Path) -> None:
    mgr = WorldManager()
    status = mgr.create_world(WorldSpec(seed=7), config_dir)
    rec = mgr.register_agent(status.world_id, AgentRegistration(agent_id="alice"))
    ctx = authorization_metadata(str(rec["token"]))
    svc = MarketSimServicer(mgr)
    req = {"world_id": status.world_id}

    client = LocalClient(World.create(config_dir, seed=7))
    stepped = svc.step({**req, "n": 1}, ctx)
    client.step(1)
    assert_consistent(mgr.ledger(status.world_id))
    assert svc.state_hash(req, ctx) == client.state_hash()
    assert stepped.state_hash == client.state_hash()

    obs = svc.observe(req, ctx)
    via_codec = decode_model(encode_model(obs), Observation)
    assert via_codec == obs
    assert obs == Observation.model_validate(client.observe("alice"))
    assert set(obs.model_dump()).isdisjoint(HIDDEN_KEYS)


def test_missing_token_is_auth_error(config_dir: Path) -> None:
    mgr = WorldManager()
    status = mgr.create_world(WorldSpec(seed=3), config_dir)
    svc = MarketSimServicer(mgr)
    with pytest.raises(AuthError):
        svc.status({"world_id": status.world_id}, None)


def test_serve_requires_grpcio_and_does_not_bind() -> None:
    from marketsim.api.grpc.server import GRPC_AVAILABLE, serve

    if GRPC_AVAILABLE:
        pytest.skip("live gRPC server is not started in CI (no TCP bind)")
    with pytest.raises(ImportError, match=r"marketsim\[grpc\]"):
        serve(WorldManager())
