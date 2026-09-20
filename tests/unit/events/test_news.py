"""T4.06 — news items hide magnitudes, honour lag, noisy severity, debug rumours."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.events.news import NewsFeed, NewsItem, item_from_event, misclassify, true_severity
from marketsim.events.schema import EventSpec, load_event

EXAMPLE = "config/events/_example.yaml"


def test_magnitudes_never_in_newsitem() -> None:
    spec = load_event(EXAMPLE)
    rng = np.random.default_rng(0)
    _, item = item_from_event(spec, 0, rng)
    blob = f"{item.headline} {item.id} {item.severity_hint} {item.category}"
    assert "0.08" not in blob
    assert "0.30" not in blob
    assert "median" not in blob
    assert not hasattr(item, "magnitude")
    assert "magnitude" not in item.__dataclass_fields__


def test_publication_lag_honoured() -> None:
    spec = EventSpec.model_validate(
        {
            "id": "lagged",
            "category": "disaster",
            "hazard": {"base_rate_per_year": 0.0},
            "news": {"headline": "Something happened", "publication_lag_days": 3, "noise": 0.0},
        }
    )
    feed = NewsFeed()
    rng = np.random.default_rng(1)
    feed.enqueue(spec, fire_tick=10, rng=rng)
    assert feed.publish_due(12, debug=False) == []
    due = feed.publish_due(13, debug=False)
    assert len(due) == 1
    assert due[0].tick == 13
    assert due[0].headline == "Something happened"


def test_severity_misclassification_rate_matches_noise() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    noise = 0.30
    n_wrong = sum(misclassify(1, noise, rng) != 1 for _ in range(n))
    assert n_wrong / n == pytest.approx(noise, rel=0.10)
    rng0 = np.random.default_rng(3)
    assert all(misclassify(2, 0.0, rng0) == 2 for _ in range(200))


def test_rumours_flagged_only_in_debug() -> None:
    spec = EventSpec.model_validate(
        {
            "id": "ghost",
            "category": "financial",
            "hazard": {"base_rate_per_year": 0.0},
            "news": {"headline": "Banks wobble", "publication_lag_days": 0, "noise": 0.0},
        }
    )
    feed = NewsFeed(rumour_rate=1.0)
    rng = np.random.default_rng(4)
    feed.enqueue(spec, 0, rng, is_rumour=True)
    pub_hidden = feed.publish_due(0, debug=False)
    assert pub_hidden[0].is_rumour is False
    feed2 = NewsFeed()
    feed2.enqueue(spec, 0, rng, is_rumour=True)
    pub_dbg = feed2.publish_due(0, debug=True)
    assert pub_dbg[0].is_rumour is True


def test_true_severity_not_exported() -> None:
    spec = load_event(EXAMPLE)
    sev = true_severity(spec)
    assert sev in (0, 1, 2, 3)
    item = NewsItem(
        id="x", tick=0, category="energy", headline="n", regions=(), sectors=(), severity_hint=sev
    )
    assert "0.08" not in str(item)
