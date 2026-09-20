"""T8.06 — explainability REST identities (1e-9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from marketsim.api.sessions import WorldManager
from marketsim.explain.traces import TRACE_ATOL

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from marketsim.api.rest import create_app, world_admin_token  # noqa: E402


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api(config_dir: Path):
    mgr = WorldManager()
    app = create_app(manager=mgr, config_dir=config_dir)
    with TestClient(app) as client:
        yield client


def _world(client: TestClient) -> tuple[str, str]:
    resp = client.post("/v1/worlds", json={"seed": 7, "mode": "professional", "run_mode": "lockstep"})
    assert resp.status_code == 200, resp.text
    status = resp.json()
    return status["world_id"], world_admin_token(status["world_id"], status["seed"])


def test_explain_requires_auth(api: TestClient) -> None:
    wid, _admin = _world(api)
    assert api.get(f"/v1/worlds/{wid}/explain/demand").status_code == 401


def test_demand_identity_via_post(api: TestClient) -> None:
    wid, admin = _world(api)
    body = {
        "baseline": 100.0,
        "income": 1.1,
        "relative_price": 0.95,
        "rate": 0.98,
        "typed_edge": 1.02,
        "want_shifter": 0.9,
        "events": 1.2,
        "own_price": 0.85,
        "availability": 0.7,
        "stickiness": 1.05,
        "rationing": 0.8,
    }
    resp = api.post(f"/v1/worlds/{wid}/explain/demand", json=body, headers=_auth(admin))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["identity_ok"] is True
    assert abs(data["residual"]) <= TRACE_ATOL
    cell = (
        data["baseline"]
        * data["income"]
        * data["relative_price"]
        * data["rate"]
        * data["typed_edge"]
        * data["want_shifter"]
        * data["events"]
    )
    share = data["own_price"] * data["availability"] * data["stickiness"]
    assert data["cell_demand"] == pytest.approx(cell, abs=TRACE_ATOL)
    assert data["share"] == pytest.approx(share, abs=TRACE_ATOL)
    assert data["qty"] == pytest.approx(cell * share * data["rationing"], abs=TRACE_ATOL)


def test_return_identity_via_post(api: TestClient) -> None:
    wid, admin = _world(api)
    body = {
        "ee0": 10.0,
        "ee1": 11.0,
        "pe0_0": 12.0,
        "pe0_1": 12.0,
        "duration": 7.0,
        "rho0": 0.05,
        "rho1": 0.06,
        "g0": 0.02,
        "g1": 0.025,
        "impact0": 0.0,
        "impact1": 0.01,
        "sentiment0": 0.02,
        "sentiment1": 0.01,
        "noise0": 0.0,
        "noise1": 0.001,
    }
    resp = api.post(f"/v1/worlds/{wid}/explain/return", json=body, headers=_auth(admin))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["identity_ok"] is True
    assert abs(data["residual"]) <= TRACE_ATOL
    parts = data["earnings"] + data["pe0"] + data["discount"] + data["growth"]
    xi = data["impact"] + data["sentiment"] + data["noise"]
    assert data["dln_v"] == pytest.approx(parts, abs=TRACE_ATOL)
    assert data["dxi"] == pytest.approx(xi, abs=TRACE_ATOL)
    assert data["dln_p"] == pytest.approx(parts + xi, abs=TRACE_ATOL)


def test_demand_change_and_get_bundle(api: TestClient) -> None:
    wid, admin = _world(api)
    resp = api.post(
        f"/v1/worlds/{wid}/explain/demand/change",
        json={
            "before": {"baseline": 100.0},
            "after": {"baseline": 100.0, "income": 0.8, "rationing": 0.5},
        },
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["identity_ok"] is True
    assert abs(data["residual"]) <= TRACE_ATOL
    assert data["dlog_qty"] < 0.0
    bundle = api.get(f"/v1/worlds/{wid}/explain", headers=_auth(admin))
    assert bundle.status_code == 200, bundle.text
    payload = bundle.json()
    assert payload["demand"]["identity_ok"] is True
    assert payload["asset_return"]["identity_ok"] is True
    assert abs(payload["demand"]["residual"]) <= TRACE_ATOL
    assert abs(payload["asset_return"]["residual"]) <= TRACE_ATOL
