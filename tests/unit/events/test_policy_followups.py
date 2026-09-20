"""T4.14 — policy follow-ups respect authority control (D13 / gate 5)."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.events.chains import policy_followup_blocked
from marketsim.events.scheduler import Events
from marketsim.events.schema import load_catalog
from marketsim.real.economy import RealEconomy
from marketsim.world import RandomWalkModule, World


def test_policy_blocked_by_control_mode(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
    assert policy_followup_blocked("fiscal_stimulus", eco) is False
    eco.policy.govt.control = "scripted"
    assert policy_followup_blocked("fiscal_stimulus", eco) is True
    assert policy_followup_blocked("monetary_tightening", eco) is False
    eco.policy.cenbank.control = "agent"
    assert policy_followup_blocked("monetary_tightening", eco) is True
    eco.policy.govt.control = "agent"
    assert policy_followup_blocked("covid_fiscal_stimulus", eco) is True
    eco.policy.govt.control = "autopilot"
    eco.policy.cenbank.control = "autopilot"
    assert policy_followup_blocked("fiscal_stimulus", eco) is False
    assert policy_followup_blocked("monetary_tightening", eco) is False


def test_covid_chain_skips_stimulus_when_govt_is_agent(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
    eco.policy.govt.control = "agent"
    cat = load_catalog("config/events")
    ev = Events(catalog=cat, damping=1.0, max_depth=3, max_concurrent=8)
    rng = np.random.default_rng(0)
    class _Q:
        def __init__(self) -> None:
            self.items: list = []

        def schedule(self, timestamp: int, payload: object, priority: int = 0) -> None:
            self.items.append((timestamp, payload, priority))

    q = _Q()
    assert ev.try_fire("covid_2020", rng, tick=0, economy=eco)
    ev.schedule_followups(cat["covid_2020"], rng, 0, 0, q, economy=eco)
    ids = [p[1]["event_id"] for p in q.items]
    assert "covid_fiscal_stimulus" not in ids
    assert any(it.headline.startswith("policy pressure") for it in ev.news.published)


def test_autopilot_still_schedules_policy_followup(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
    cat = load_catalog("config/events")
    ev = Events(catalog=cat, damping=1.0, max_depth=3, max_concurrent=8)
    rng = np.random.default_rng(1)

    class _Q:
        def __init__(self) -> None:
            self.items: list = []

        def schedule(self, timestamp: int, payload: object, priority: int = 0) -> None:
            self.items.append(payload)

    q = _Q()
    ev.schedule_followups(cat["covid_2020"], rng, 0, 0, q, economy=eco)
    # p=0.85 with damping 1 → usually schedules; force by looping seeds if needed
    fired = any(p.get("event_id") == "covid_fiscal_stimulus" for p in q.items)
    if not fired:
        rng = np.random.default_rng(2)
        q.items.clear()
        ev.schedule_followups(cat["covid_2020"], rng, 0, 0, q, economy=eco)
        fired = any(p.get("event_id") == "covid_fiscal_stimulus" for p in q.items)
    assert fired


def test_event_rng_isolation_unchanged(config_dir) -> None:
    cat = load_catalog("config/events")
    # tiny catalog ping
    ping = cat["confidence_drop"]
    tiny = {"confidence_drop": ping}

    def hist(seed: int, extra=None):
        mods = [Events(catalog=tiny, max_concurrent=4)]
        if extra:
            mods.extend(extra)
        w = World.create(config_dir, seed=seed, modules=mods)
        w.step(120)
        ev = next(m for m in w.modules if m.name == "events")
        return [(n.tick, n.event_id) for n in ev.history]

    assert hist(3) == hist(3, extra=[RandomWalkModule()])


def test_fiscal_stimulus_sets_transfer_lever(cfg, io) -> None:
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
    cat = load_catalog("config/events")
    ev = Events(catalog=cat, max_concurrent=8)
    rng = np.random.default_rng(4)
    g0 = float(eco.real.flat(eco.real.G0).sum())
    assert ev.try_fire("fiscal_stimulus", rng, tick=0, economy=eco)
    assert eco.policy.govt.source_of.get("transfer_oneoff") == "event"
    assert float(eco.policy.govt.effective["transfer_oneoff"]) > 0.0
    assert float(eco.policy.govt.effective["transfer_oneoff"]) == pytest.approx(g0 * 0.035, rel=0.6)
