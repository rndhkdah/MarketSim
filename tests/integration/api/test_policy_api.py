"""T7.16 — policymaker role, clipped levers, PolicyEnv, gate 6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marketsim.api.schemas import AgentRegistration, WorldSpec
from marketsim.api.sessions import WorldManager
from marketsim.ledger.sfc import assert_consistent
from marketsim.sdk.policy_env import LAMBDA_B, LAMBDA_Y, MASK_AGENT, PolicyEnv, policy_loss

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
gymnasium = pytest.importorskip("gymnasium")

from fastapi.testclient import TestClient  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

from marketsim.api.rest import create_app, world_admin_token  # noqa: E402

ORDER = {
    "agent_id": "pm",
    "side": "buy",
    "qty": 1,
    "symbol": "EQ:FIRM:acme",
    "price": 1.0,
}


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api(config_dir: Path):
    mgr = WorldManager()
    app = create_app(manager=mgr, config_dir=config_dir)
    with TestClient(app) as client:
        yield client, mgr


def _world(client: TestClient, *, mode: str = "professional") -> tuple[str, str]:
    resp = client.post("/v1/worlds", json={"seed": 7, "mode": mode, "run_mode": "lockstep"})
    assert resp.status_code == 200, resp.text
    status = resp.json()
    return status["world_id"], world_admin_token(status["world_id"], status["seed"])


def _register(client: TestClient, wid: str, admin: str, **body: object) -> dict:
    resp = client.post(f"/v1/worlds/{wid}/agents", json=body, headers=_auth(admin))
    return {"status": resp.status_code, "json": resp.json(), "text": resp.text}


def test_gate6_authorities_and_policy_through_api(api) -> None:
    """Gate 6: no policymaker → autopilot; a bound client runs fiscal, monetary, DMO."""
    client, mgr = api
    wid, admin = _world(client)
    desk = mgr.policy_desk(wid)
    assert desk.govt.control == "autopilot"
    assert desk.cenbank.control == "autopilot"

    alice = _register(client, wid, admin, agent_id="alice")
    assert alice["status"] == 200
    forbidden = client.post(
        f"/v1/worlds/{wid}/government/decisions",
        json={"tau_y": 0.2},
        headers=_auth(alice["json"]["token"]),
    )
    assert forbidden.status_code == 403

    gov = _register(client, wid, admin, agent_id="treasurer", role="policymaker", authority="GOVT")
    assert gov["status"] == 200
    assert mgr.policy_desk(wid).govt.control == "agent"
    assert mgr.policy_desk(wid).cenbank.control == "autopilot"

    clipped = client.post(
        f"/v1/worlds/{wid}/government/decisions",
        json={"tau_y": 0.9, "vat": 0.1},
        headers=_auth(gov["json"]["token"]),
    )
    assert clipped.status_code == 200, clipped.text
    body = clipped.json()
    assert body["authority"] == "GOVT"
    assert body["levers"]["tau_y"] == pytest.approx(0.6)
    assert body["levers"]["vat"] == pytest.approx(0.1)
    assert any(row["lever"] == "tau_y" for row in body["clips"])

    state = client.get(f"/v1/worlds/{wid}/government/state", headers=_auth(gov["json"]["token"]))
    assert state.status_code == 200
    payload = state.json()
    assert payload["control"] == "agent"
    assert payload["effective"]["tau_y"] == pytest.approx(0.6)
    assert "sizes" in payload["issuance_plan"]
    assert set(payload["issuance_plan"]["sizes"]) >= {"GB_BILL", "GB_NOTE", "GB_BOND"}

    cb = _register(client, wid, admin, agent_id="governor", role="policymaker", authority="CENBANK")
    assert cb["status"] == 200
    assert mgr.policy_desk(wid).cenbank.control == "agent"
    rate = client.post(
        f"/v1/worlds/{wid}/cenbank/decisions",
        json={"rate": 0.50, "pi_star": 0.02},
        headers=_auth(cb["json"]["token"]),
    )
    assert rate.status_code == 200, rate.text
    assert rate.json()["levers"]["rate"] == pytest.approx(0.20)
    assert rate.json()["levers"]["pi_star"] == pytest.approx(0.02)
    cb_state = client.get(f"/v1/worlds/{wid}/cenbank/state", headers=_auth(cb["json"]["token"]))
    assert cb_state.status_code == 200
    assert cb_state.json()["control"] == "agent"
    assert cb_state.json()["effective"]["rate"] == pytest.approx(0.20)

    gone = client.delete(f"/v1/worlds/{wid}/agents/treasurer", headers=_auth(admin))
    assert gone.status_code == 200
    assert mgr.policy_desk(wid).govt.control == "autopilot"
    assert mgr.policy_desk(wid).cenbank.control == "agent"
    client.delete(f"/v1/worlds/{wid}/agents/governor", headers=_auth(admin))
    assert mgr.policy_desk(wid).cenbank.control == "autopilot"
    assert_consistent(mgr.ledger(wid))


def test_policymaker_cannot_open_trading_account_professional(api) -> None:
    client, mgr = api
    wid, admin = _world(client, mode="professional")
    denied = _register(
        client,
        wid,
        admin,
        agent_id="pm",
        role="policymaker",
        authority="GOVT",
        starting_capital=50.0,
    )
    assert denied["status"] == 403
    assert denied["json"]["code"] == "forbidden"

    ok = _register(client, wid, admin, agent_id="pm", role="policymaker", authority="GOVT")
    assert ok["status"] == 200
    trade = client.post(f"/v1/worlds/{wid}/orders", json=ORDER, headers=_auth(ok["json"]["token"]))
    assert trade.status_code == 403

    # Direct manager path (same rule).
    other = mgr.create_world(WorldSpec(seed=3, mode="professional"), Path(mgr.world(wid).cfg.config_dir))
    with pytest.raises(Exception, match="trading account"):
        mgr.register_agent(
            other.world_id,
            AgentRegistration(
                agent_id="x",
                role="policymaker",
                authority="CENBANK",
                starting_capital=1.0,
            ),
        )


def test_policy_env_checker_reward_and_mask(config_dir: Path) -> None:
    env = PolicyEnv(config_dir, horizon=4, seed=0, authority="GOVT")
    check_env(env, skip_render_check=True)

    a, _ = PolicyEnv(config_dir, horizon=4, seed=21).reset(seed=21)
    b, _ = PolicyEnv(config_dir, horizon=4, seed=21).reset(seed=21)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])

    idle = env.idle_action()
    env.reset(seed=5)
    env.step(idle)
    assert env.last_decision is not None
    assert env.last_decision.tau_y is None

    on = env.idle_action()
    on["mask"][list(env.levers).index("tau_y")] = MASK_AGENT
    on["tau_y"] = np.array([1.0], dtype=np.float32)
    env.reset(seed=5)
    env.step(on)
    assert env.last_decision is not None
    assert env.last_decision.tau_y is not None

    loss = policy_loss(0.04, 0.02, 0.01, 0.6, 0.6)
    assert loss == pytest.approx(-((0.02) ** 2 + LAMBDA_Y * 0.01**2 + LAMBDA_B * 0.0))
