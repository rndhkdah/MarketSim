"""T4.05 — cascade graph checks, depth/concurrency caps, RNG isolation (gate 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from marketsim.core.errors import ConfigError
from marketsim.events.chains import check_subcritical, expected_cascade_size
from marketsim.events.scheduler import Events
from marketsim.events.schema import EventSpec
from marketsim.world import RandomWalkModule, World


def _ev(eid: str, followups: list[dict] | None = None, *, category: str = "disaster") -> EventSpec:
    return EventSpec.model_validate(
        {
            "id": eid,
            "category": category,
            "hazard": {"base_rate_per_year": 0.0},
            "cooldown_days": 0,
            "followups": followups or [],
        }
    )


def test_sum_p_above_cap_rejected() -> None:
    cat = {
        "root": _ev(
            "root",
            [
                {"event_id": "a", "probability": 0.5},
                {"event_id": "b", "probability": 0.5},
            ],
        ),
        "a": _ev("a"),
        "b": _ev("b"),
    }
    with pytest.raises(ConfigError, match="follow-up probabilities"):
        check_subcritical(cat)


def test_cycle_rejected() -> None:
    cat = {
        "a": _ev("a", [{"event_id": "b", "probability": 0.4}]),
        "b": _ev("b", [{"event_id": "a", "probability": 0.4}]),
    }
    with pytest.raises(ConfigError, match="cycle"):
        check_subcritical(cat)


def test_dag_ok_and_depth_cap() -> None:
    cat = {
        "a": _ev("a", [{"event_id": "b", "probability": 0.5}]),
        "b": _ev("b", [{"event_id": "c", "probability": 0.5}]),
        "c": _ev("c", [{"event_id": "d", "probability": 0.5}]),
        "d": _ev("d"),
    }
    check_subcritical(cat, max_depth=3)
    with pytest.raises(ConfigError, match="depth"):
        check_subcritical(cat, max_depth=2)
    sizes = expected_cascade_size(cat, damping=1.0)
    assert sizes["a"] == pytest.approx(1.0 + 0.5 * (1.0 + 0.5 * (1.0 + 0.5)))


def test_depth_never_exceeds_max() -> None:
    cat = {k: _ev(k) for k in ("r", "c1", "c2", "c3", "c4")}
    ev = Events(catalog=cat, max_depth=3, max_concurrent=8)
    rng = np.random.default_rng(0)
    assert ev.try_fire("r", rng, 0, depth=0)
    assert ev.try_fire("c1", rng, 1, depth=1, parent="r")
    assert ev.try_fire("c2", rng, 2, depth=2, parent="c1")
    assert ev.try_fire("c3", rng, 3, depth=3, parent="c2")
    assert ev.try_fire("c4", rng, 4, depth=4, parent="c3") is False
    assert max(n.depth for n in ev.history) <= 3


def test_concurrency_never_exceeds_cap() -> None:
    cat = {k: _ev(k, category="disaster" if k == "a" else "financial") for k in ("a", "b", "c")}
    ev = Events(catalog=cat, max_depth=3, max_concurrent=1)
    rng = np.random.default_rng(0)
    assert ev.try_fire("a", rng, 10)
    assert ev.try_fire("b", rng, 10) is False
    assert ev.concurrent(10) == 1
    assert ev.try_fire("c", rng, 11)


def test_gate4_event_rng_isolated(config_dir, tmp_path: Path) -> None:
    raw = {
        "id": "ping",
        "category": "disaster",
        "hazard": {"base_rate_per_year": 12.0},
        "cooldown_days": 5,
        "composition": [],
        "followups": [],
    }
    (tmp_path / "ping.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = EventSpec.model_validate(raw)
    cat = {"ping": spec}

    def history(seed: int, extra: list | None = None) -> list[tuple[int, str]]:
        mods = [Events(catalog=cat, damping=0.7, max_depth=3, max_concurrent=4)]
        if extra:
            mods.extend(extra)
        w = World.create(config_dir, seed=seed, modules=mods)
        w.step(252)
        ev = next(m for m in w.modules if m.name == "events")
        return [(n.tick, n.event_id) for n in ev.history]

    h1 = history(11)
    h2 = history(11)
    h3 = history(11, extra=[RandomWalkModule()])
    h4 = history(12)
    assert h1 == h2 == h3
    assert h1 != h4
    assert h1  # some fires at this rate
