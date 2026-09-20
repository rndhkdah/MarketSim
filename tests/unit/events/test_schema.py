"""T4.01 — event schema validation (gate 1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marketsim.core.errors import ConfigError
from marketsim.events.schema import EventSpec, load_catalog, load_event, parse_extra_variable

EXAMPLE = Path("config/events/_example.yaml")


def test_example_loads() -> None:
    spec = load_event(EXAMPLE)
    assert spec.id == "chip_shortage"
    assert spec.composition[0].shock == "supply"
    assert spec.composition[0].targets == {"sectors": ["SEMIS"]}


def test_bad_primitive_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["id"] = "bad_prim"
    raw["composition"] = [
        {"shock": "weather", "magnitude": {"dist": "fixed", "value": 0.1, "min": 0.1, "max": 0.1}}
    ]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="seven primitives"):
        load_event(path)


def test_unknown_extra_variable_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["id"] = "bad_extra"
    raw["followups"] = []
    raw["effects_extra"] = [
        {
            "variable": "magic_wand",
            "shape": "step",
            "magnitude": {"dist": "fixed", "value": 0.1},
            "duration": 21,
        }
    ]
    path = tmp_path / "extra.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="whitelist"):
        load_event(path)


def test_missing_bounds_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["id"] = "no_bounds"
    raw["composition"][0]["magnitude"] = {"dist": "lognormal", "median": 0.08, "sigma": 0.4, "sign": -1}
    path = tmp_path / "bounds.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="truncation bounds"):
        load_event(path)


def test_dangling_followup_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["followups"] = [{"event_id": "does_not_exist", "probability": 0.4}]
    (tmp_path / "chip_shortage.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="dangling follow-up"):
        load_catalog(tmp_path)


def test_catalog_skips_underscore_example(tmp_path: Path) -> None:
    # underscore files are not catalog members (the shipped example stays a fixture)
    assert load_catalog(Path("config/events")) == {}
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["followups"] = []
    (tmp_path / "chip_shortage.yaml").write_text(yaml.safe_dump(raw))
    cat = load_catalog(tmp_path)
    assert "chip_shortage" in cat


def test_parse_extra_variable() -> None:
    assert parse_extra_variable("vat") == ("vat", None)
    assert parse_extra_variable("want_shift[EATING_OUT_LEISURE]")[0] == "want_shift"
    with pytest.raises(ValueError, match="whitelist"):
        parse_extra_variable("foo[bar]")


def test_unknown_sector_mask_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["id"] = "bad_mask"
    raw["composition"][0]["targets"] = {"sectors": ["NOTASECTOR"]}
    path = tmp_path / "mask.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="unknown sector"):
        load_event(path)


def test_eventspec_rejects_unknown_category() -> None:
    with pytest.raises(Exception, match="unknown category"):
        EventSpec.model_validate(
            {
                "id": "x",
                "category": "aliens",
                "hazard": {"base_rate_per_year": 0.0},
            }
        )
