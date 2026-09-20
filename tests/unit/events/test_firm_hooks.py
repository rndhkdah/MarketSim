"""T4.12 — firm hooks validate, schedule, and receive the payload."""

from __future__ import annotations

import numpy as np

from marketsim.events.firm_hooks import FIRM_EVENT_IDS, FirmHookBus, payload_from_event
from marketsim.events.scheduler import Events
from marketsim.events.schema import load_catalog


def test_firm_events_validate_and_hook_payload() -> None:
    cat = load_catalog("config/events")
    for eid in FIRM_EVENT_IDS:
        assert eid in cat
        assert cat[eid].category == "firm"
    bus = FirmHookBus()
    ev = Events(catalog=cat, max_concurrent=8)
    ev.firm_hooks = bus
    rng = np.random.default_rng(0)
    assert ev.try_fire("strike", rng, tick=5)
    assert bus.calls[0].event_id == "strike"
    assert bus.calls[0].tick == 5
    spec = cat["plant_accident"]
    p = payload_from_event(spec, 9)
    assert p.event_id == "plant_accident"
    bus.dispatch(p)
    assert bus.calls[-1].tick == 9
    raw = bus.to_state()
    bus2 = FirmHookBus()
    bus2.from_state(raw)
    assert bus2.calls[0].event_id == "strike"


def test_recall_and_bankruptcy_schedule() -> None:
    cat = load_catalog("config/events")
    ev = Events(catalog=cat, max_concurrent=8)
    rng = np.random.default_rng(1)
    assert ev.try_fire("recall", rng, tick=2)
    assert ev.try_fire("large_bankruptcy", rng, tick=3, ignore_cooldown=True)
    assert {c.event_id for c in ev.firm_hooks.calls} >= {"recall", "large_bankruptcy"}
