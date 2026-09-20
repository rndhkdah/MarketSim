"""T7.04 — REST endpoints: happy path, 401, 422, operator/policy 403, hidden observe keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from marketsim.api.schemas import Observation
from marketsim.api.sessions import WorldManager
from marketsim.ledger.sfc import assert_consistent
from marketsim.scenarios.randomise import HIDDEN_KEYS

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from marketsim.api.rest import create_app, world_admin_token  # noqa: E402

ORDER_BODY = {
    "agent_id": "alice",
    "side": "buy",
    "qty": 10,
    "symbol": "EQ:FIRM:acme",
    "price": 1.25,
}

PROTECTED: tuple[tuple[str, str, dict | None], ...] = (
    ("GET", "/v1/worlds/{wid}", None),
    ("POST", "/v1/worlds/{wid}/reset", None),
    ("POST", "/v1/worlds/{wid}/step", {"n": 1}),
    ("DELETE", "/v1/worlds/{wid}", None),
    ("POST", "/v1/worlds/{wid}/agents", {"agent_id": "alice"}),
    ("GET", "/v1/worlds/{wid}/observe", None),
    ("POST", "/v1/worlds/{wid}/orders", ORDER_BODY),
    ("GET", "/v1/worlds/{wid}/orders", None),
    ("GET", "/v1/worlds/{wid}/fills", None),
    ("DELETE", "/v1/worlds/{wid}/orders/1", None),
    ("POST", "/v1/worlds/{wid}/firms", {"firm_id": "acme", "operator": "alice", "capital": 10.0}),
    ("GET", "/v1/worlds/{wid}/firms/acme/state", None),
    ("GET", "/v1/worlds/{wid}/firms/acme/financials", None),
    ("GET", "/v1/worlds/{wid}/firms/acme/reports", None),
    ("POST", "/v1/worlds/{wid}/firms/acme/decisions", {"firm_id": "acme", "operator": "alice"}),
    ("GET", "/v1/worlds/{wid}/goods", None),
    ("GET", "/v1/worlds/{wid}/labour/CAPITAL", None),
    ("GET", "/v1/worlds/{wid}/regions", None),
    ("GET", "/v1/worlds/{wid}/bonds", None),
    ("GET", "/v1/worlds/{wid}/bonds/auctions", None),
    ("GET", "/v1/worlds/{wid}/government/state", None),
    ("POST", "/v1/worlds/{wid}/government/decisions", {}),
    ("GET", "/v1/worlds/{wid}/cenbank/state", None),
    ("POST", "/v1/worlds/{wid}/cenbank/decisions", {}),
)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _error(resp) -> dict:
    body = resp.json()
    assert "code" in body and "message" in body and "details" in body
    return body


@pytest.fixture
def api(config_dir: Path):
    mgr = WorldManager()
    app = create_app(manager=mgr, config_dir=config_dir)
    with TestClient(app) as client:
        yield client, mgr


def _create(client: TestClient, *, seed: int = 7) -> tuple[dict, str]:
    resp = client.post("/v1/worlds", json={"seed": seed, "mode": "professional", "run_mode": "lockstep"})
    assert resp.status_code == 200, resp.text
    status = resp.json()
    token = world_admin_token(status["world_id"], status["seed"])
    return status, token


def _register(
    client: TestClient,
    wid: str,
    admin: str,
    agent_id: str,
    *,
    role: str = "agent",
    **extra: object,
) -> dict:
    body = {"agent_id": agent_id, "role": role, **extra}
    resp = client.post(f"/v1/worlds/{wid}/agents", json=body, headers=_auth(admin))
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_openapi_published(api) -> None:
    client, _mgr = api
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/v1/worlds" in paths
    assert "/v1/worlds/{wid}/observe" in paths
    assert "/v1/worlds/{wid}/orders" in paths
    schemas = resp.json()["components"]["schemas"]
    assert "ErrorModel" in schemas


def test_worlds_happy_reset_step_delete(api) -> None:
    client, mgr = api
    status, admin = _create(client)
    wid = status["world_id"]
    assert status["tick"] == 0
    assert status["mode"] == "professional"
    assert status["n_agents"] == 0

    got = client.get(f"/v1/worlds/{wid}", headers=_auth(admin))
    assert got.status_code == 200
    assert got.json()["state_hash"] == status["state_hash"]

    stepped = client.post(f"/v1/worlds/{wid}/step", json={"n": 2}, headers=_auth(admin))
    assert stepped.status_code == 200
    assert stepped.json()["tick"] == 2
    assert_consistent(mgr.ledger(wid))

    reset = client.post(f"/v1/worlds/{wid}/reset", headers=_auth(admin))
    assert reset.status_code == 200
    assert reset.json()["tick"] == 0

    deleted = client.delete(f"/v1/worlds/{wid}", headers=_auth(admin))
    assert deleted.status_code == 200
    missing = client.get(f"/v1/worlds/{wid}", headers=_auth(admin))
    assert missing.status_code == 404
    assert _error(missing)["code"] == "not_found"


def test_worlds_auth_and_validation(api) -> None:
    client, _mgr = api
    status, _admin = _create(client)
    wid = status["world_id"]

    unauth = client.get(f"/v1/worlds/{wid}")
    assert unauth.status_code == 401
    assert _error(unauth)["code"] == "unauthorized"

    bad = client.get(f"/v1/worlds/{wid}", headers=_auth("0" * 64))
    assert bad.status_code == 401

    extra = client.post("/v1/worlds", json={"seed": 1, "nope": True})
    assert extra.status_code == 422
    assert _error(extra)["code"] == "validation_error"

    neg = client.post("/v1/worlds", json={"seed": -1})
    assert neg.status_code == 422


def test_agents_happy_auth_validation(api) -> None:
    client, _mgr = api
    status, admin = _create(client)
    wid = status["world_id"]

    rec = _register(client, wid, admin, "alice", starting_capital=25.0)
    assert rec["account"] == "AGENT:alice"
    assert rec["token"]
    assert rec["limits"]["orders_per_tick"] >= 1

    unauth = client.post(f"/v1/worlds/{wid}/agents", json={"agent_id": "bob"})
    assert unauth.status_code == 401
    assert _error(unauth)["code"] == "unauthorized"

    extra = client.post(
        f"/v1/worlds/{wid}/agents",
        json={"agent_id": "bob", "nope": 1},
        headers=_auth(admin),
    )
    assert extra.status_code == 422
    assert _error(extra)["code"] == "validation_error"


def test_observe_happy_auth_validation_and_hidden_keys(api) -> None:
    client, mgr = api
    status, admin = _create(client)
    wid = status["world_id"]
    alice = _register(client, wid, admin, "alice")

    hidden = {key: 1.0 for key in ("xi", "arb_capital")}
    mgr.world(wid)._observations["*"] = {**hidden, "quotes": {"EQ:NPC:AUTOS": 1.0}}

    resp = client.get(f"/v1/worlds/{wid}/observe", headers=_auth(alice["token"]))
    assert resp.status_code == 200
    body = resp.json()
    obs = Observation.model_validate(body)
    assert obs.agent_id == "alice"
    assert obs.reports == []
    assert "xi" not in body
    assert "arb_capital" not in body
    for key in HIDDEN_KEYS:
        assert key not in body
    assert body["quotes"]["EQ:NPC:AUTOS"] == 1.0

    with pytest.raises(ValidationError):
        Observation.model_validate({**body, "xi": 0.2})
    with pytest.raises(ValidationError):
        Observation.model_validate({**body, "arb_capital": 1.0})

    unauth = client.get(f"/v1/worlds/{wid}/observe")
    assert unauth.status_code == 401

    bad_since = client.get(f"/v1/worlds/{wid}/observe?since=not-a-tick", headers=_auth(alice["token"]))
    assert bad_since.status_code == 422
    assert _error(bad_since)["code"] == "validation_error"


def test_orders_happy_idempotency_auth_validation(api) -> None:
    client, _mgr = api
    status, admin = _create(client)
    wid = status["world_id"]
    alice = _register(client, wid, admin, "alice")
    headers = {**_auth(alice["token"]), "Idempotency-Key": "k1"}

    first = client.post(f"/v1/worlds/{wid}/orders", json=ORDER_BODY, headers=headers)
    assert first.status_code == 200, first.text
    ack = first.json()
    assert ack["status"] == "resting"
    assert ack["order_id"] == 1
    assert ack["idempotency_key"] == "k1"

    again = client.post(f"/v1/worlds/{wid}/orders", json=ORDER_BODY, headers=headers)
    assert again.status_code == 200
    assert again.json() == ack

    listed = client.get(f"/v1/worlds/{wid}/orders", headers=_auth(alice["token"]))
    assert listed.status_code == 200
    assert listed.json()["items"][0]["order_id"] == 1

    fills = client.get(f"/v1/worlds/{wid}/fills", headers=_auth(alice["token"]))
    assert fills.status_code == 200
    assert fills.json()["items"] == []

    cancelled = client.delete(f"/v1/worlds/{wid}/orders/1", headers=_auth(alice["token"]))
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    unauth = client.post(f"/v1/worlds/{wid}/orders", json=ORDER_BODY)
    assert unauth.status_code == 401

    extra = client.post(
        f"/v1/worlds/{wid}/orders",
        json={**ORDER_BODY, "nope": True},
        headers=_auth(alice["token"]),
    )
    assert extra.status_code == 422
    assert _error(extra)["code"] == "validation_error"

    missing_px = client.post(
        f"/v1/worlds/{wid}/orders",
        json={"agent_id": "alice", "side": "buy", "qty": 10, "symbol": "EQ:FIRM:acme"},
        headers=_auth(alice["token"]),
    )
    assert missing_px.status_code == 422


def test_step_happy_auth_validation(api) -> None:
    client, mgr = api
    status, admin = _create(client)
    wid = status["world_id"]

    ok = client.post(f"/v1/worlds/{wid}/step", json={"n": 1}, headers=_auth(admin))
    assert ok.status_code == 200
    assert ok.json()["tick"] == 1
    assert_consistent(mgr.ledger(wid))

    unauth = client.post(f"/v1/worlds/{wid}/step", json={"n": 1})
    assert unauth.status_code == 401

    extra = client.post(f"/v1/worlds/{wid}/step", json={"n": 1, "nope": 1}, headers=_auth(admin))
    assert extra.status_code == 422
    assert _error(extra)["code"] == "validation_error"

    zero = client.post(f"/v1/worlds/{wid}/step", json={"n": 0}, headers=_auth(admin))
    assert zero.status_code == 422


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_every_world_endpoint_requires_bearer(api, method: str, path: str, body: dict | None) -> None:
    client, _mgr = api
    status, _admin = _create(client)
    url = path.format(wid=status["world_id"])
    resp = client.request(method, url, json=body)
    assert resp.status_code == 401
    assert _error(resp)["code"] == "unauthorized"


def test_operator_only_firm_state_and_happy_firm_reads(api) -> None:
    client, _mgr = api
    status, admin = _create(client)
    wid = status["world_id"]
    alice = _register(client, wid, admin, "alice")
    bob = _register(client, wid, admin, "bob")

    found = client.post(
        f"/v1/worlds/{wid}/firms",
        json={"firm_id": "acme", "operator": "alice", "capital": 50.0},
        headers=_auth(alice["token"]),
    )
    assert found.status_code == 200, found.text

    live = client.get(f"/v1/worlds/{wid}/firms/acme/state", headers=_auth(alice["token"]))
    assert live.status_code == 200
    assert live.json()["operator"] == "alice"
    assert live.json()["live"] is True

    forbidden = client.get(f"/v1/worlds/{wid}/firms/acme/state", headers=_auth(bob["token"]))
    assert forbidden.status_code == 403
    assert _error(forbidden)["code"] == "forbidden"

    fin = client.get(f"/v1/worlds/{wid}/firms/acme/financials", headers=_auth(bob["token"]))
    assert fin.status_code == 200
    reports = client.get(f"/v1/worlds/{wid}/firms/acme/reports", headers=_auth(bob["token"]))
    assert reports.status_code == 200
    assert reports.json()["items"] == []

    decision = client.post(
        f"/v1/worlds/{wid}/firms/acme/decisions",
        json={"firm_id": "acme", "operator": "alice", "vacancies": 2.0},
        headers=_auth(alice["token"]),
    )
    assert decision.status_code == 200

    other = client.post(
        f"/v1/worlds/{wid}/firms/acme/decisions",
        json={"firm_id": "acme", "operator": "alice"},
        headers=_auth(bob["token"]),
    )
    assert other.status_code == 403

    extra = client.post(
        f"/v1/worlds/{wid}/firms",
        json={"firm_id": "beta", "operator": "alice", "capital": 10.0, "nope": 1},
        headers=_auth(alice["token"]),
    )
    assert extra.status_code == 422


def test_policymaker_only_government_decisions(api) -> None:
    client, _mgr = api
    status, admin = _create(client)
    wid = status["world_id"]
    alice = _register(client, wid, admin, "alice")
    pm = _register(client, wid, admin, "pm", role="policymaker", authority="GOVT")

    forbidden = client.post(
        f"/v1/worlds/{wid}/government/decisions",
        json={"tau_y": 0.2},
        headers=_auth(alice["token"]),
    )
    assert forbidden.status_code == 403
    assert _error(forbidden)["code"] == "forbidden"

    ok = client.post(
        f"/v1/worlds/{wid}/government/decisions",
        json={"tau_y": 0.2},
        headers=_auth(pm["token"]),
    )
    assert ok.status_code == 200
    assert ok.json()["authority"] == "GOVT"
    assert ok.json()["levers"]["tau_y"] == 0.2

    extra = client.post(
        f"/v1/worlds/{wid}/government/decisions",
        json={"tau_y": 0.2, "nope": True},
        headers=_auth(pm["token"]),
    )
    assert extra.status_code == 422

    gov = client.get(f"/v1/worlds/{wid}/government/state", headers=_auth(alice["token"]))
    assert gov.status_code == 200
    cb = client.get(f"/v1/worlds/{wid}/cenbank/state", headers=_auth(alice["token"]))
    assert cb.status_code == 200


def test_goods_labour_regions_bonds_happy(api) -> None:
    client, _mgr = api
    status, admin = _create(client)
    wid = status["world_id"]
    alice = _register(client, wid, admin, "alice")
    headers = _auth(alice["token"])

    goods = client.get(f"/v1/worlds/{wid}/goods", headers=headers)
    assert goods.status_code == 200
    labour = client.get(f"/v1/worlds/{wid}/labour/CAPITAL", headers=headers)
    assert labour.status_code == 200
    assert labour.json()["region"] == "CAPITAL"
    regions = client.get(f"/v1/worlds/{wid}/regions", headers=headers)
    assert regions.status_code == 200
    codes = {row["code"] for row in regions.json()["regions"]}
    assert "CAPITAL" in codes
    bonds = client.get(f"/v1/worlds/{wid}/bonds", headers=headers)
    assert bonds.status_code == 200
    auctions = client.get(f"/v1/worlds/{wid}/bonds/auctions", headers=headers)
    assert auctions.status_code == 200
    unknown = client.get(f"/v1/worlds/{wid}/labour/NOWHERE", headers=headers)
    assert unknown.status_code == 404
