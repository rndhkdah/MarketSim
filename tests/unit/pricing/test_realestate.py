"""T8.09 — regional RE index; collateral matches the traded price."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from marketsim.pricing.realestate import (
    DEFAULT_REGIONS,
    DEFAULT_WEIGHTS,
    RealEstateSpec,
    collateral_from_price,
    load_realestate_spec,
    loc_weights,
    national_index,
    regional_prices,
    symbols,
    traded_national,
)
from marketsim.real.credit import collateral_index


def test_symbols_and_zero_xi_national_equals_v() -> None:
    assert symbols() == ("RE:CAPITAL", "RE:INDUSTRIAL", "RE:RESOURCE")
    v = 12.0
    prices = regional_prices(v)
    assert sum(DEFAULT_WEIGHTS) == pytest.approx(1.0)
    assert traded_national(v) == pytest.approx(v, abs=1e-12)
    assert prices["RE:CAPITAL"] == pytest.approx(v, abs=1e-12)
    assert prices["RE:RESOURCE"] == pytest.approx(v, abs=1e-12)


def test_collateral_channel_matches_traded_price() -> None:
    v = 10.0
    assert collateral_from_price(v, v) == pytest.approx(1.0)
    low = collateral_from_price(8.0, v)
    high = collateral_from_price(12.0, v)
    assert low < 1.0
    assert high > 1.0
    assert low == pytest.approx(collateral_index(math.log(8.0 / 10.0)))
    assert low < collateral_from_price(9.0, v)


def test_xi_moves_only_that_region() -> None:
    base = regional_prices(10.0)
    shocked = regional_prices(10.0, xi={"RE:CAPITAL": 0.1})
    assert shocked["RE:CAPITAL"] > base["RE:CAPITAL"]
    assert shocked["RE:INDUSTRIAL"] == pytest.approx(base["RE:INDUSTRIAL"])


def test_lower_traded_price_lowers_lambda_coll_monotonically() -> None:
    trend = 10.0
    path = [12.0, 10.0, 9.0, 8.0, 5.0]
    lams = [collateral_from_price(p, trend) for p in path]
    assert lams[1] == pytest.approx(1.0)
    assert all(lams[i] > lams[i + 1] for i in range(len(lams) - 1))


def test_explicit_weights_dict_national_identity() -> None:
    v = 10.0
    w = {"CAPITAL": 0.2, "INDUSTRIAL": 0.5, "RESOURCE": 0.3}
    assert loc_weights(w).sum() == pytest.approx(1.0)
    assert traded_national(v, weights=w) == pytest.approx(v, abs=1e-12)
    xi = {"RE:CAPITAL": 0.2, "RE:INDUSTRIAL": 0.0, "RE:RESOURCE": -0.1}
    prices = regional_prices(v, xi=xi)
    nat = traded_national(v, weights=w, xi=xi)
    wn = loc_weights(w)
    expected = sum(wn[i] * prices[f"RE:{r}"] for i, r in enumerate(DEFAULT_REGIONS))
    assert nat == pytest.approx(expected, abs=1e-12)
    assert nat == pytest.approx(national_index(prices, w), abs=1e-12)


def test_load_spec_from_yaml_and_explicit_weights(tmp_path) -> None:
    path = tmp_path / "markets.yaml"
    path.write_text(
        "\n".join(
            [
                "realestate:",
                '  prefix: "RE:"',
                "  venue: engine_mm",
                "  regions: [CAPITAL, INDUSTRIAL, RESOURCE]",
                "  weights: [0.45, 0.35, 0.20]",
                "",
            ]
        ), encoding="utf-8"
    )
    spec = load_realestate_spec(path)
    assert spec.prefix == "RE:"
    assert spec.venue == "engine_mm"
    assert spec.regions == DEFAULT_REGIONS
    assert spec.weights == pytest.approx(DEFAULT_WEIGHTS)
    assert sum(spec.weights) == pytest.approx(1.0)
    override = load_realestate_spec(path, weights={"CAPITAL": 1.0, "INDUSTRIAL": 0.0, "RESOURCE": 0.0})
    xi = {"RE:CAPITAL": 0.15}
    prices = regional_prices(4.0, xi=xi)
    assert traded_national(4.0, weights=override.weights, xi=xi) == pytest.approx(prices["RE:CAPITAL"])
    clone = RealEstateSpec.from_state(spec.to_state())
    assert clone == spec


def test_shipped_markets_yaml_loader_without_config_py() -> None:
    shipped = Path(__file__).resolve().parents[3] / "config" / "markets.yaml"
    spec = load_realestate_spec(shipped)
    assert spec.prefix == "RE:"
    assert spec.venue == "engine_mm"
    assert spec.regions == DEFAULT_REGIONS
    assert spec.weights == pytest.approx(DEFAULT_WEIGHTS)
    assert traded_national(7.5, weights=spec.weights) == pytest.approx(7.5)
