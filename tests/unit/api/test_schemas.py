"""T7.01 — versioned API schema round-trips and golden stability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from marketsim.api.schemas import (
    API_SCHEMA_VERSION,
    CARD_MODELS,
    OBSERVATION_FIELDS,
    AgentRegistration,
    CorporateAction,
    ErrorModel,
    Fill,
    FirmDecision,
    FoundFirmRequest,
    GoodsView,
    LabourView,
    ListingRequest,
    NewsItem,
    Observation,
    OrderAck,
    OrderRequest,
    Report,
    SubmitPayload,
    WorldSpec,
    WorldStatus,
    api_json_schema,
    dump_api_schema,
    observation_field_names,
)
from marketsim.events.news import NewsItem as EngineNewsItem
from marketsim.firms.levers import FirmDecision as EngineFirmDecision
from marketsim.market.clob import Fill as EngineFill
from marketsim.market.clob import Order as EngineOrder
from marketsim.market.clob import OrderStatus, OrderType, Side, SubmitResult, TimeInForce
from marketsim.scenarios.randomise import HIDDEN_KEYS, OBSERVE_PUBLIC_KEYS

GOLDEN = Path(__file__).resolve().parents[2] / "golden" / "data" / f"api_schema_{API_SCHEMA_VERSION}.json"

EXAMPLES: list[tuple[type, dict]] = [
    (WorldSpec, {"seed": 7, "scenario": "baseline", "overrides": {"world.scale": 1.0}}),
    (
        WorldStatus,
        {
            "world_id": "w1",
            "tick": 0,
            "seed": 7,
            "mode": "professional",
            "run_mode": "lockstep",
            "state_hash": "abc",
        },
    ),
    (AgentRegistration, {"agent_id": "alice", "starting_capital": 100.0}),
    (Observation, {"tick": 0, "agent_id": "alice"}),
    (
        OrderRequest,
        {"agent_id": "alice", "side": "buy", "qty": 10, "symbol": "EQ:FIRM:acme", "price": 1.25},
    ),
    (OrderAck, {"order_id": 1, "status": "resting", "remaining": 10}),
    (
        Fill,
        {
            "symbol": "EQ:FIRM:acme",
            "price": 1.25,
            "qty": 10,
            "maker": "MM",
            "taker": "alice",
            "maker_order_id": 1,
            "taker_order_id": 2,
            "notional": 12.5,
            "tick": 3,
        },
    ),
    (
        FirmDecision,
        {
            "firm_id": "acme",
            "operator": "alice",
            "posted_price": [{"region": "INDUSTRIAL", "sector": "AUTOS", "value": 1.02}],
            "vacancies": 2.0,
        },
    ),
    (FoundFirmRequest, {"firm_id": "acme", "operator": "alice", "capital": 50.0}),
    (ListingRequest, {"firm_id": "acme", "shares": 1_000_000, "reserve_price": 1.0}),
    (CorporateAction, {"firm_id": "acme", "kind": "dividend", "amount": 3.0}),
    (GoodsView, {"reference_prices": [{"region": "INDUSTRIAL", "sector": "AUTOS", "value": 1.0}]}),
    (LabourView, {"region": "INDUSTRIAL", "unemployed": 4.0, "wage_bar": 1.1}),
    (
        NewsItem,
        {
            "id": "chip_shortage",
            "tick": 4,
            "category": "supply_chain",
            "headline": "Chip plants halt",
            "severity_hint": 2,
        },
    ),
    (Report, {"firm_id": "acme", "live": False, "books": {"tick": 0}, "tick": 10}),
    (ErrorModel, {"code": "validation_error", "message": "bad field", "details": {"field": "qty"}}),
]


@pytest.mark.parametrize("cls,payload", EXAMPLES, ids=[c.__name__ for c, _ in EXAMPLES])
def test_json_round_trip(cls: type, payload: dict) -> None:
    obj = cls.model_validate(payload)
    again = cls.model_validate_json(obj.model_dump_json())
    assert again == obj
    assert json.loads(obj.model_dump_json()) == json.loads(again.model_dump_json())


def test_card_models_are_in_v1_document() -> None:
    doc = api_json_schema()
    assert doc["version"] == API_SCHEMA_VERSION
    assert set(CARD_MODELS)  # non-empty
    assert {cls.__name__ for cls in CARD_MODELS} <= set(doc["models"])


def test_firm_decision_engine_round_trip() -> None:
    raw = EngineFirmDecision(
        "acme",
        "alice",
        posted_price={("INDUSTRIAL", "AUTOS"): 1.02},
        target_output={("INDUSTRIAL", "AUTOS"): 8.0},
        new_plant=("COAST", "SEMIS", 2.5),
        borrow=10.0,
    )
    api = FirmDecision.from_engine(raw)
    back = api.to_engine()
    assert back.firm_id == raw.firm_id
    assert back.posted_price == raw.posted_price
    assert back.target_output == raw.target_output
    assert back.new_plant == raw.new_plant
    assert back.borrow == raw.borrow
    assert back.vacancies is None
    assert FirmDecision.from_engine(back) == api
    assert FirmDecision.model_validate_json(api.model_dump_json()) == api


def test_order_and_fill_engine_round_trip() -> None:
    order = EngineOrder(
        agent_id="alice",
        side=Side.BUY,
        qty=10,
        symbol="EQ:FIRM:acme",
        order_type=OrderType.LIMIT,
        tif=TimeInForce.GTC,
        price=1.25,
        sequence=3,
    )
    req = OrderRequest.from_engine(order, idempotency_key="k1")
    assert req.to_engine() == order
    assert req.idempotency_key == "k1"
    fill = EngineFill("EQ:FIRM:acme", 1.25, 10, "MM", "alice", 1, 2, 12.5, 3)
    wire = Fill.from_engine(fill)
    assert wire.to_engine() == fill
    ack = OrderAck.from_engine(
        SubmitResult(order_id=2, status=OrderStatus.PARTIAL, remaining=4, fills=(fill,), reason=None),
        idempotency_key="k1",
    )
    assert ack.status == "partial"
    assert ack.fills[0].to_engine() == fill
    assert OrderRequest.model_validate_json(req.model_dump_json()) == req


def test_news_item_engine_round_trip_and_no_magnitude() -> None:
    item = EngineNewsItem(
        id="oil_shock",
        tick=2,
        category="energy",
        headline="Oil spike",
        regions=("COAST",),
        sectors=("ENERGY",),
        severity_hint=3,
        is_rumour=False,
    )
    wire = NewsItem.from_engine(item)
    assert wire.to_engine() == item
    assert "magnitude" not in NewsItem.model_fields
    with pytest.raises(ValidationError):
        NewsItem.model_validate({**wire.model_dump(), "magnitude": 0.3})


def test_observation_whitelist_has_no_hidden_state() -> None:
    fields = observation_field_names()
    assert fields == OBSERVATION_FIELDS
    assert fields == OBSERVE_PUBLIC_KEYS
    assert fields.isdisjoint(HIDDEN_KEYS)
    for hidden in HIDDEN_KEYS:
        assert hidden not in Observation.model_fields
    with pytest.raises(ValidationError):
        Observation.model_validate({"tick": 0, "agent_id": "alice", "xi": 0.2})
    with pytest.raises(ValidationError):
        Observation.model_validate({"tick": 0, "agent_id": "alice", "arb_capital": 1.0})


def test_submit_payload_round_trip() -> None:
    body = SubmitPayload.model_validate(
        {
            "decisions": [{"firm_id": "acme", "operator": "alice"}],
            "orders": [
                {"agent_id": "alice", "side": "sell", "qty": 5, "symbol": "EQ:NPC:AUTOS", "order_type": "market"}
            ],
            "client_seq": 1,
        }
    )
    assert SubmitPayload.model_validate_json(body.model_dump_json()) == body
    assert body.orders[0].to_engine().order_type is OrderType.MARKET


def test_error_model_shape() -> None:
    err = ErrorModel(code="not_found", message="no such world")
    dumped = json.loads(err.model_dump_json())
    assert dumped == {"code": "not_found", "message": "no such world", "details": {}}


def test_golden_schema_stable() -> None:
    generated = json.loads(dump_api_schema())
    assert GOLDEN.is_file(), f"missing golden {GOLDEN}"
    frozen = json.loads(GOLDEN.read_text())
    assert generated == frozen, (
        "API JSON schema changed; bump API_SCHEMA_VERSION and replace "
        f"{GOLDEN.name} (do not edit the frozen v1 snapshot in place)"
    )
    assert frozen["version"] == API_SCHEMA_VERSION
