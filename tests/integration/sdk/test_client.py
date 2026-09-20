"""T7.07 — HTTP / WS SDK against a live ``create_app`` TestClient."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from marketsim.api.schemas import AgentRegistration, Observation, WorldSpec
from marketsim.api.sessions import WorldManager
from marketsim.ledger.sfc import assert_consistent
from marketsim.sdk.local import LOCAL_METHODS

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from marketsim.api.rest import create_app  # noqa: E402
from marketsim.api.ws import StreamHub, include_ws  # noqa: E402
from marketsim.sdk.client import AsyncHttpClient, HttpClient  # noqa: E402

# Starlette TestClient default (unitless URL).
_TEST_BASE_URL = "http://testserver"


def _fill(*, maker: str, taker: str, tick: int, qty: int = 1) -> dict:
    """Minimal fill payload. ``price`` is cr / share; ``qty`` is shares; ``tick`` is days."""
    return {
        "symbol": "EQ:FIRM:acme",
        "price": 1.0,
        "qty": qty,
        "maker": maker,
        "taker": taker,
        "maker_order_id": 1,
        "taker_order_id": 2,
        "notional": float(qty),
        "tick": tick,
    }


def _read_events(stream: object, n: int) -> list[dict]:
    """Skip ``subscribed`` control frames. ``n`` is an event count."""
    out: list[dict] = []
    while len(out) < n:
        msg = stream.recv()  # type: ignore[attr-defined]
        if msg["type"] != "subscribed":
            out.append(msg)
    return out


@pytest.fixture
def api(config_dir: Path):
    mgr = WorldManager()
    app = create_app(manager=mgr, config_dir=config_dir)
    include_ws(app, manager=mgr)
    with TestClient(app) as http:
        yield http, mgr, app


def test_local_methods_subset_of_http_client() -> None:
    """LOCAL_METHODS ⊆ HttpClient public methods (T7.03 / T7.07 contract)."""
    assert isinstance(LOCAL_METHODS, frozenset)
    missing = sorted(name for name in LOCAL_METHODS if not hasattr(HttpClient, name))
    assert missing == []
    for name in LOCAL_METHODS:
        assert callable(getattr(HttpClient, name))


def test_create_register_observe_step_hash_matches_manager(api) -> None:
    http, mgr, _app = api
    client = HttpClient(client=http)
    status = client.create_world(WorldSpec(seed=7, mode="professional", run_mode="lockstep"))
    acct = client.register_agent(AgentRegistration(agent_id="alice"))
    assert acct.account == "AGENT:alice"
    client.bind(token=acct.token)
    obs = client.observe()
    assert isinstance(obs, Observation)
    assert obs.agent_id == "alice"
    assert obs.tick == 0
    stepped = client.step(1)
    assert stepped.tick == 1
    assert_consistent(mgr.ledger(status.world_id))
    assert client.state_hash() == mgr.status(status.world_id).state_hash
    assert client.state_hash() == mgr.world(status.world_id).state_hash()


def test_ws_reconnect_with_since_no_gaps_or_duplicates(api) -> None:
    http, _mgr, app = api
    hub: StreamHub = app.state.ws_hub
    client = HttpClient(client=http)
    status = client.create_world(WorldSpec(seed=7))
    acct = client.register_agent(AgentRegistration(agent_id="alice"))
    client.bind(token=acct.token)
    wid = status.world_id
    for day in (1, 2, 3):
        hub.publish(wid, "tick", {"tick": day}, tick=day)
        hub.publish(wid, "fill", _fill(maker="mm", taker="alice", tick=day, qty=day), tick=day)

    stream = client.subscribe()
    first = _read_events(stream, 6)
    seqs = [msg["seq"] for msg in first]
    assert seqs == [1, 2, 3, 4, 5, 6]
    assert stream.last_seq == 6
    stream.close()

    hub.publish(wid, "tick", {"tick": 4}, tick=4)
    hub.publish(wid, "fill", _fill(maker="mm", taker="alice", tick=4, qty=4), tick=4)

    stream.reconnect()
    second = _read_events(stream, 2)
    stream.close()
    second_seqs = [msg["seq"] for msg in second]
    assert second_seqs == [7, 8]
    assert seqs + second_seqs == list(range(1, 9))
    assert set(seqs).isdisjoint(set(second_seqs))


def test_async_observe_step_happy_path(config_dir: Path) -> None:
    mgr = WorldManager()
    app = create_app(manager=mgr, config_dir=config_dir)

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=_TEST_BASE_URL) as http:
            client = AsyncHttpClient(client=http)
            status = await client.create_world(WorldSpec(seed=11, mode="professional", run_mode="lockstep"))
            acct = await client.register_agent(AgentRegistration(agent_id="bob"))
            client.bind(token=acct.token)
            obs = await client.observe()
            assert isinstance(obs, Observation)
            assert obs.agent_id == "bob"
            stepped = await client.step(1)
            assert stepped.tick == 1
            assert_consistent(mgr.ledger(status.world_id))
            assert await client.state_hash() == mgr.status(status.world_id).state_hash

    asyncio.run(_run())
