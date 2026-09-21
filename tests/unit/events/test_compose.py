"""T4.02 — composition engine: masks, bounds, determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marketsim.events.compose import apply_composition, sample_dist
from marketsim.events.schema import DistSpec, EventSpec, load_event
from marketsim.real.shocks import ShockBus, mid_month_weight

_ROOT = Path(__file__).resolve().parents[3]

EXAMPLE = _ROOT / "config" / "events" / "_example.yaml"


def test_semis_supply_touches_no_other_cell(cfg) -> None:
    spec = load_event(EXAMPLE)
    bus = ShockBus.from_config(cfg)
    rng = np.random.default_rng(0)
    applied = apply_composition(spec, bus, rng, day=0)
    assert applied[0].kind == "supply"
    codes = list(cfg.codes)
    sup = np.asarray(bus.states["sup"], dtype=float)
    for i, code in enumerate(codes):
        if code != "SEMIS":
            assert sup[i] == 0.0
    assert sup[codes.index("SEMIS")] != 0.0
    assert np.all(np.asarray(bus.states["cost"], dtype=float) == 0.0)
    assert bus.states["dem"] == 0.0
    assert bus.states["mon"] == 0.0
    assert bus.states["fisc"] == 0.0
    assert bus.states["risk"] == 0.0


def test_sampled_magnitudes_respect_bounds() -> None:
    spec = load_event(EXAMPLE).composition[0].magnitude
    rng = np.random.default_rng(1)
    for _ in range(400):
        mag = sample_dist(spec, rng)
        assert spec.min <= abs(mag) <= spec.max
        assert mag * spec.sign >= 0.0 or mag == 0.0


def test_compose_deterministic(cfg) -> None:
    spec = load_event(EXAMPLE)
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    bus_a = ShockBus.from_config(cfg)
    bus_b = ShockBus.from_config(cfg)
    a = apply_composition(spec, bus_a, rng_a, day=7)
    b = apply_composition(spec, bus_b, rng_b, day=7)
    assert a[0].magnitude == b[0].magnitude
    np.testing.assert_array_equal(bus_a.states["sup"], bus_b.states["sup"])


def test_mid_month_weight_applied(cfg) -> None:
    event = EventSpec.model_validate(
        {
            "id": "fixed_demand",
            "category": "financial",
            "hazard": {"base_rate_per_year": 0.0},
            "composition": [
                {
                    "shock": "demand",
                    "magnitude": {"dist": "fixed", "value": 0.10},
                    "persistence_q": 1,
                }
            ],
        }
    )
    bus = ShockBus.from_config(cfg)
    rng = np.random.default_rng(0)
    apply_composition(event, bus, rng, day=7)
    assert bus.states["dem"] == pytest.approx(0.10 * mid_month_weight(7), abs=1e-12)


def test_normal_and_uniform_truncated() -> None:
    rng = np.random.default_rng(3)
    normal = DistSpec.model_validate(
        {"dist": "normal", "mean": 0.1, "sigma": 0.5, "min": 0.02, "max": 0.12}
    )
    uniform = DistSpec.model_validate({"dist": "uniform", "min": 0.2, "max": 0.4})
    for _ in range(200):
        n = sample_dist(normal, rng)
        u = sample_dist(uniform, rng)
        assert 0.02 <= n <= 0.12
        assert 0.2 <= u <= 0.4
