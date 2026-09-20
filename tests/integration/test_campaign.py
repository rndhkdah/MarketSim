"""T8.07 — scripted player finishes a campaign through the public API only.

Justified extra file: the card requires the test but lists only
``docs/game-contract.md`` and ``src/marketsim/scenarios/campaign.py``.
This module drives ``create_app`` + ``HttpClient`` / REST. It does not open a
``World`` or call engine hooks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marketsim.api.schemas import NewsItem as ApiNewsItem
from marketsim.events.news import NewsItem
from marketsim.scenarios.campaign import (
    DEMO_DURATION_TICKS,
    DEMO_EVENT_SEVERITY,
    DEMO_FOUNDING_CAPITAL_CR,
    DEMO_NPC_SCALE,
    STORM_EVENT_SEVERITY,
    STORM_FOUNDING_CAPITAL_CR,
    STORM_NPC_SCALE,
    CampaignSpec,
    PacingControls,
    campaign_scenarios,
    is_takeover_news,
    narrate,
    read_campaign_slot,
    run_scripted_player,
    sample_takeover_news,
    takeover_notification,
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def test_scripted_player_finishes_demo_campaign(config_dir: Path, tmp_path: Path) -> None:
    """A scripted player completes the short demo via create_app + HttpClient."""
    result = run_scripted_player(config_dir, slot_dir=tmp_path)
    assert result.finished is True
    assert result.campaign_id == "demo"
    assert result.agent_id == "player"
    assert result.firm_id == "campaign_co"
    assert result.tick == DEMO_DURATION_TICKS
    assert result.ticks_played == DEMO_DURATION_TICKS
    assert result.seed == 7
    assert result.world_id.startswith("w")
    assert len(result.state_hash) == 64
    expected = DEMO_FOUNDING_CAPITAL_CR * DEMO_NPC_SCALE * DEMO_EVENT_SEVERITY
    assert result.difficulty == expected
    assert result.score.net_worth == DEMO_FOUNDING_CAPITAL_CR
    assert result.score.valuation == 0.0
    assert result.score.market_share == 0.0
    assert result.slot_path is not None
    slot = read_campaign_slot(result.slot_path)
    assert slot.tick == result.tick
    assert slot.state_hash == result.state_hash
    assert slot.finished is True
    assert slot.spec.difficulty() == expected
    mid = tmp_path / "mid.campaign.json"
    assert mid.is_file()
    mid_slot = read_campaign_slot(mid)
    assert mid_slot.world_id == result.world_id
    assert mid_slot.tick == 0
    assert mid_slot.finished is False


def test_difficulty_is_founding_capital_times_npc_scale_times_severity() -> None:
    spec = CampaignSpec(
        id="probe",
        seed=0,
        founding_capital=STORM_FOUNDING_CAPITAL_CR,
        npc_scale=STORM_NPC_SCALE,
        event_severity=STORM_EVENT_SEVERITY,
    )
    assert spec.difficulty() == (
        STORM_FOUNDING_CAPITAL_CR * STORM_NPC_SCALE * STORM_EVENT_SEVERITY
    )


def test_narrative_templates_from_newsitem_fields() -> None:
    engine = NewsItem(
        id="chip_shortage_2020",
        tick=4,
        category="supply",
        headline="Chip plants report disruption",
        regions=("R1",),
        sectors=("SEMIS",),
        severity_hint=2,
        is_rumour=True,
    )
    text = narrate(engine)
    assert "Chip plants report disruption" in text
    assert "supply" in text
    assert "day 4" in text
    assert "hint 2" in text
    assert "R1" in text
    assert "SEMIS" in text
    assert text.startswith("Rumour")
    assert "0.4" not in text
    api = ApiNewsItem.from_engine(engine)
    assert narrate(api) == text
    assert is_takeover_news(engine) is False
    assert takeover_notification(engine) is None


def test_takeover_notification_matches_control_desk_news() -> None:
    item = sample_takeover_news(firm_id="acme", operator="bob", tick=20)
    assert item.category == "corporate"
    assert item.id == "control:acme:20"
    assert is_takeover_news(item) is True
    note = takeover_notification(item)
    assert note is not None
    assert note.startswith("Takeover:")
    assert "acme" in note
    assert "day 20" in note
    assert "%" not in note
    assert narrate(item).startswith("Operating control of acme")


def test_pacing_pause_and_speed_use_caller_clock() -> None:
    pacing = PacingControls(mode="realtime", ticks_per_second=1.0)
    pacing.pause()
    assert pacing.due_ticks(10.0) == 0
    pacing.resume()
    pacing.set_speed(2.0)
    assert pacing.interval_s() == 0.5
    assert pacing.due_ticks(2.0) == 4
    assert pacing.paused is False


def test_campaign_scenarios_catalog() -> None:
    pack = campaign_scenarios()
    assert set(pack) == {"demo", "operator", "storm_watch"}
    storm = pack["storm_watch"]
    assert storm.difficulty() == (
        STORM_FOUNDING_CAPITAL_CR * STORM_NPC_SCALE * STORM_EVENT_SEVERITY
    )
    assert pack["demo"].duration_ticks == DEMO_DURATION_TICKS
