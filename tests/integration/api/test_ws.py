"""T7.05 — WebSocket streams: resume, own fills, slow-consumer tick coalesce."""

from __future__ import annotations

from pathlib import Path

import pytest

from marketsim.api.schemas import AgentRegistration, WorldSpec
from marketsim.api.sessions import WorldManager

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from marketsim.api.ws import StreamHub, include_ws  # noqa: E402


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _fill(*, maker: str, taker: str, tick: int, qty: int = 1) -> dict:
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


def _read_data(ws, n: int) -> list[dict]:
    out: list[dict] = []
    while len(out) < n:
        msg = ws.receive_json()
        if msg["type"] != "subscribed":
            out.append(msg)
    return out


@pytest.fixture
def ws_api(config_dir: Path):
    mgr = WorldManager()
    status = mgr.create_world(WorldSpec(seed=7), config_dir)
    alice = mgr.register_agent(status.world_id, AgentRegistration(agent_id="alice"))
    bob = mgr.register_agent(status.world_id, AgentRegistration(agent_id="bob"))
    app = FastAPI()
    include_ws(app, manager=mgr)
    with TestClient(app) as client:
        yield {
            "client": client,
            "hub": app.state.ws_hub,
            "wid": status.world_id,
            "alice": alice["token"],
            "bob": bob["token"],
        }


def test_resume_no_gaps_or_duplicates(ws_api) -> None:
    hub: StreamHub = ws_api["hub"]
    client: TestClient = ws_api["client"]
    wid = ws_api["wid"]
    url = f"/v1/worlds/{wid}/stream"
    headers = _auth(ws_api["alice"])

    for day in (1, 2, 3):
        hub.publish(wid, "tick", {"tick": day}, tick=day)
        hub.publish(wid, "fill", _fill(maker="mm", taker="alice", tick=day, qty=day), tick=day)

    with client.websocket_connect(url, headers=headers) as ws:
        assert ws.receive_json()["type"] == "subscribed"
        first = _read_data(ws, 6)
    seqs = [msg["seq"] for msg in first]
    assert seqs == [1, 2, 3, 4, 5, 6]
    last = seqs[-1]

    hub.publish(wid, "tick", {"tick": 4}, tick=4)
    hub.publish(wid, "fill", _fill(maker="mm", taker="alice", tick=4, qty=4), tick=4)

    with client.websocket_connect(f"{url}?since={last}", headers=headers) as ws:
        assert ws.receive_json()["type"] == "subscribed"
        second = _read_data(ws, 2)
    second_seqs = [msg["seq"] for msg in second]
    assert second_seqs == [7, 8]
    assert seqs + second_seqs == list(range(1, 9))
    assert set(seqs).isdisjoint(set(second_seqs))


def test_agent_sees_only_own_fills(ws_api) -> None:
    hub: StreamHub = ws_api["hub"]
    client: TestClient = ws_api["client"]
    wid, url = ws_api["wid"], f"/v1/worlds/{ws_api['wid']}/stream"
    hub.publish(wid, "tick", {"tick": 1}, tick=1)
    hub.publish(wid, "fill", _fill(maker="mm", taker="alice", tick=1, qty=3), tick=1)
    hub.publish(wid, "fill", _fill(maker="bob", taker="mm", tick=1, qty=9), tick=1)
    hub.publish(wid, "news", {"id": "n-alice", "headline": "alice only"}, tick=1, agent_id="alice")
    hub.publish(wid, "news", {"id": "n-pub", "headline": "public"}, tick=1)

    with client.websocket_connect(url, headers=_auth(ws_api["alice"])) as ws:
        alice_msgs = _read_data(ws, 4)
    with client.websocket_connect(url, headers=_auth(ws_api["bob"])) as ws:
        bob_msgs = _read_data(ws, 3)

    alice_fills = [msg for msg in alice_msgs if msg["type"] == "fill"]
    bob_fills = [msg for msg in bob_msgs if msg["type"] == "fill"]
    assert len(alice_fills) == 1 and alice_fills[0]["payload"]["taker"] == "alice"
    assert len(bob_fills) == 1 and bob_fills[0]["payload"]["maker"] == "bob"
    alice_news = {msg["payload"]["id"] for msg in alice_msgs if msg["type"] == "news"}
    bob_news = {msg["payload"]["id"] for msg in bob_msgs if msg["type"] == "news"}
    assert alice_news == {"n-alice", "n-pub"}
    assert bob_news == {"n-pub"}


def test_slow_consumer_coalesces_ticks(ws_api) -> None:
    hub: StreamHub = ws_api["hub"]
    client: TestClient = ws_api["client"]
    wid = ws_api["wid"]
    burst = [
        {"kind": "tick", "tick": 1, "payload": {"tick": 1}},
        {"kind": "fill", "tick": 1, "payload": _fill(maker="mm", taker="alice", tick=1, qty=5)},
        {"kind": "tick", "tick": 2, "payload": {"tick": 2}},
        {"kind": "tick", "tick": 3, "payload": {"tick": 3}},
        {"kind": "news", "tick": 3, "payload": {"id": "n1", "headline": "kept"}},
    ]
    with client.websocket_connect(f"/v1/worlds/{wid}/stream", headers=_auth(ws_api["alice"])) as ws:
        assert ws.receive_json()["type"] == "subscribed"
        hub.publish_many(wid, burst)
        msgs = _read_data(ws, 3)
    types = [msg["type"] for msg in msgs]
    assert types.count("tick") == 1
    assert "fill" in types and "news" in types
    assert next(msg for msg in msgs if msg["type"] == "tick")["tick"] == 3
    assert next(msg for msg in msgs if msg["type"] == "fill")["payload"]["qty"] == 5
    assert [event.kind for event in hub.replay(wid, "alice", 0)] == ["tick", "fill", "tick", "tick", "news"]
