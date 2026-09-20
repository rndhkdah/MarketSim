"""T6.22 — domain randomisation stays in-range; observe() whitelist hides state."""

from __future__ import annotations

import pytest

from marketsim.core.config import load_config
from marketsim.core.rng import RngHub
from marketsim.pricing.mispricing import CALM_INDEX_VOL_MAX, CALM_INDEX_VOL_MIN, STREAM_NAMES
from marketsim.scenarios.randomise import (
    DEFAULT_RANGES,
    HIDDEN_KEYS,
    OBSERVE_PUBLIC_KEYS,
    STREAM_NAME,
    DomainRandomiser,
    config_overrides,
    observe,
    replay_header,
    sample_overrides,
)
from marketsim.world import World


def test_sampled_overrides_within_ranges() -> None:
    draws = sample_overrides(RngHub(7))
    assert set(draws) == set(DEFAULT_RANGES)
    for key, (lo, hi) in DEFAULT_RANGES.items():
        assert lo <= draws[key] <= hi
    lo, hi = DEFAULT_RANGES["mispricing.target_ann_vol"]
    assert (lo, hi) == (CALM_INDEX_VOL_MIN, CALM_INDEX_VOL_MAX)


def test_sample_uses_named_stream_and_is_deterministic() -> None:
    assert STREAM_NAME == "randomise"
    a = sample_overrides(RngHub(7))
    b = sample_overrides(RngHub(7))
    assert a == b
    assert sample_overrides(RngHub(8)) != a
    hub = RngHub(11)
    before = {name: hub.stream(name).bit_generator.state for name in STREAM_NAMES}
    sample_overrides(hub)
    assert "randomise" in hub.to_state()["streams"]
    for name in STREAM_NAMES:
        assert hub.stream(name).bit_generator.state == before[name]


def test_replay_header_logs_draws() -> None:
    draws = sample_overrides(RngHub(3))
    header = replay_header(seed=3, overrides=draws, config_hash="abc")
    assert header["seed"] == 3
    assert header["stream"] == STREAM_NAME
    assert header["overrides"] == draws
    assert header["config_hash"] == "abc"
    assert list(header["overrides"]) == sorted(draws)


def test_config_overrides_apply(config_dir) -> None:
    draws = sample_overrides(RngHub(1))
    ov = config_overrides(draws)
    assert ov
    assert all(k.startswith("markets.") for k in ov)
    cfg = load_config(config_dir, overrides=ov)
    assert cfg.markets is not None
    assert cfg.markets.turnover == pytest.approx(draws["markets.turnover"])
    assert cfg.markets.impact.Y == pytest.approx(draws["markets.impact.Y"])
    assert cfg.markets.mm.s0 == pytest.approx(draws["markets.mm.s0"])


def test_episode_state_roundtrip() -> None:
    epi = DomainRandomiser()
    epi.draw(RngHub(2))
    clone = DomainRandomiser.from_state(epi.to_state())
    assert clone.overrides == epi.overrides
    assert clone.ranges == epi.ranges
    assert clone.replay_header(seed=2)["overrides"] == epi.overrides


def test_observe_leaks_no_hidden_field(config_dir) -> None:
    assert OBSERVE_PUBLIC_KEYS.isdisjoint(HIDDEN_KEYS)
    w = World.create(config_dir, seed=1)
    w._observations["*"] = {
        "releases": {"cpi": {"m:0": 1.01}, "_month_truth": {"cpi": 9.9}, "unpublished": 1.0},
        "quotes": {"EQ:NPC:AUTOS": 1.02},
        "xi": [0.25],
        "capital": 0.4,
        "arb_capital": 0.55,
        "last_impact": [0.1],
        "sentiment": [0.05],
        "noise": [0.02],
        "mispricing": {"n": [0.01]},
    }
    raw = w.observe("alice")
    assert raw["tick"] == 0
    assert raw["agent_id"] == "alice"
    obs = observe(w, "alice")
    assert obs["tick"] == 0
    assert obs["agent_id"] == "alice"
    assert obs["quotes"]["EQ:NPC:AUTOS"] == pytest.approx(1.02)
    assert obs["releases"]["cpi"]["m:0"] == pytest.approx(1.01)
    assert "_month_truth" not in obs["releases"]
    assert "unpublished" not in obs["releases"]
    for hidden in (
        "xi",
        "capital",
        "arb_capital",
        "last_impact",
        "sentiment",
        "noise",
        "mispricing",
        "_month_truth",
    ):
        assert hidden not in obs
    assert set(obs) <= OBSERVE_PUBLIC_KEYS
