"""T7.17 — bond REST: calendar, uniform-price bids, coupons and redemptions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marketsim.api.schemas import (
    API_SCHEMA_VERSION,
    BondAuctionsView,
    BondBidRequest,
    BondsView,
    dump_api_schema,
)
from marketsim.api.sessions import WorldManager
from marketsim.firms.accounts import agent_entity
from marketsim.ledger.sfc import assert_consistent
from marketsim.market.auction import DAYS_PER_MONTH, auction_tick
from marketsim.pricing.bond_buckets import month_face_flows

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from marketsim.api.rest import create_app, world_admin_token  # noqa: E402

NOTE_AID = "0:GB_NOTE"
GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "data" / f"api_schema_{API_SCHEMA_VERSION}.json"
V1_GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "data" / "api_schema_v1.json"


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def api(config_dir: Path):
    mgr = WorldManager()
    app = create_app(manager=mgr, config_dir=config_dir)
    with TestClient(app) as client:
        yield client, mgr


def _world(client: TestClient) -> tuple[str, str]:
    resp = client.post("/v1/worlds", json={"seed": 7, "mode": "professional", "run_mode": "lockstep"})
    assert resp.status_code == 200, resp.text
    status = resp.json()
    wid = status["world_id"]
    return wid, world_admin_token(wid, status["seed"])


def _register(client: TestClient, wid: str, admin: str, agent_id: str, capital: float = 100.0) -> str:
    resp = client.post(
        f"/v1/worlds/{wid}/agents",
        headers=_auth(admin),
        json={"agent_id": agent_id, "starting_capital": capital},
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["token"])


def test_agent_sees_calendar(api) -> None:
    client, _mgr = api
    wid, admin = _world(client)
    token = _register(client, wid, admin, "alice")
    resp = client.get(f"/v1/worlds/{wid}/bonds/auctions", headers=_auth(token))
    assert resp.status_code == 200, resp.text
    view = BondAuctionsView.model_validate(resp.json())
    ids = [slot.auction_id for slot in view.calendar]
    assert NOTE_AID in ids
    note = next(slot for slot in view.calendar if slot.auction_id == NOTE_AID)
    assert note.instrument == "GB_NOTE"
    assert note.auction_tick == auction_tick(0)
    assert note.size == pytest.approx(10.0)  # 0.40 × 25 cr
    assert note.y_fair == pytest.approx(0.042)
    assert view.sizes["GB_NOTE"] == pytest.approx(10.0)
    assert view.results == []


def test_bid_wins_or_loses_at_stop_out(api) -> None:
    client, mgr = api
    wid, admin = _world(client)
    alice = _register(client, wid, admin, "alice")
    bob = _register(client, wid, admin, "bob")
    win = client.post(
        f"/v1/worlds/{wid}/bonds/auctions/{NOTE_AID}/bids",
        headers=_auth(alice),
        json={"yield_annual": 0.045, "qty": 8.0},
    )
    lose = client.post(
        f"/v1/worlds/{wid}/bonds/auctions/{NOTE_AID}/bids",
        headers=_auth(bob),
        json={"yield_annual": 0.080, "qty": 10.0},
    )
    assert win.status_code == 200, win.text
    assert lose.status_code == 200, lose.text
    BondBidRequest.model_validate({"yield_annual": 0.045, "qty": 8.0})

    stepped = client.post(f"/v1/worlds/{wid}/step", headers=_auth(admin), json={"n": auction_tick(0) + 1})
    assert stepped.status_code == 200, stepped.text

    auctions = client.get(f"/v1/worlds/{wid}/bonds/auctions", headers=_auth(alice))
    assert auctions.status_code == 200, auctions.text
    view = BondAuctionsView.model_validate(auctions.json())
    result = next(row for row in view.results if row.auction_id == NOTE_AID)
    assert result.agent_fill["alice"] == pytest.approx(8.0)
    assert "bob" not in result.agent_fill
    assert result.winners == ["alice"]
    assert result.stop_out == pytest.approx(0.045)
    assert result.size == pytest.approx(10.0)

    books = client.get(f"/v1/worlds/{wid}/bonds", headers=_auth(alice))
    assert books.status_code == 200, books.text
    bonds = BondsView.model_validate(books.json())
    assert bonds.holdings.own["GB_NOTE"] == pytest.approx(8.0)
    assert bonds.holdings.published["alice"]["GB_NOTE"] == pytest.approx(8.0)
    assert bonds.outstanding["GB_NOTE"] == pytest.approx(10.0)
    assert bonds.curve["GB_NOTE"] == pytest.approx(result.stop_out)

    led = mgr.ledger(wid)
    assert led.position(agent_entity("alice"), "GB_NOTE") == pytest.approx(8.0)
    assert led.position(agent_entity("bob"), "GB_NOTE") == pytest.approx(0.0)
    assert_consistent(led)

    late = client.post(
        f"/v1/worlds/{wid}/bonds/auctions/{NOTE_AID}/bids",
        headers=_auth(alice),
        json={"yield_annual": 0.041, "qty": 1.0},
    )
    assert late.status_code == 409
    missing = client.post(
        f"/v1/worlds/{wid}/bonds/auctions/9:GB_BILL/bids",
        headers=_auth(alice),
        json={"yield_annual": 0.04, "qty": 1.0},
    )
    assert missing.status_code == 404


def test_winner_receives_coupons_and_redemptions(api) -> None:
    client, mgr = api
    wid, admin = _world(client)
    alice = _register(client, wid, admin, "alice")
    client.post(
        f"/v1/worlds/{wid}/bonds/auctions/{NOTE_AID}/bids",
        headers=_auth(alice),
        json={"yield_annual": 0.045, "qty": 8.0},
    )
    client.post(f"/v1/worlds/{wid}/step", headers=_auth(admin), json={"n": DAYS_PER_MONTH})

    books = client.get(f"/v1/worlds/{wid}/bonds", headers=_auth(alice))
    assert books.status_code == 200, books.text
    bonds = BondsView.model_validate(books.json())
    flows = [row for row in bonds.cashflows if row.agent_id == "alice" and row.instrument == "GB_NOTE"]
    assert flows
    coupon, redeem, face_next = month_face_flows(8.0, 0.042, 1.0 / 3.0)
    assert flows[0].coupon == pytest.approx(coupon)
    assert flows[0].redemption == pytest.approx(redeem)
    assert flows[0].face_next == pytest.approx(face_next)
    assert bonds.holdings.own["GB_NOTE"] == pytest.approx(face_next)

    led = mgr.ledger(wid)
    assert led.position(agent_entity("alice"), "GB_NOTE") == pytest.approx(face_next)
    assert_consistent(led)


def test_schema_snapshot_includes_bond_objects() -> None:
    generated = json.loads(dump_api_schema())
    assert generated["version"] == "v1.1"
    for name in ("BondsView", "BondAuctionsView", "BondBidRequest", "BondBidAck"):
        assert name in generated["models"]
    assert GOLDEN.is_file(), f"missing golden {GOLDEN}"
    assert json.loads(GOLDEN.read_text()) == generated
    assert V1_GOLDEN.is_file()
    frozen_v1 = json.loads(V1_GOLDEN.read_text())
    assert frozen_v1["version"] == "v1"
    assert frozen_v1 != generated
