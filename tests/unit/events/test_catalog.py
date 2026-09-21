"""T4.08 — shipped event catalog validates; chain graph is sub-critical."""

from __future__ import annotations

from pathlib import Path

from marketsim.events.chains import check_subcritical
from marketsim.events.schema import load_catalog

_ROOT = Path(__file__).resolve().parents[3]

CATALOG = _ROOT / "config" / "events"
TEMPLATES = {
    "oil_embargo_1973",
    "asian_crisis_1997",
    "dotcom_2000",
    "gfc_2008",
    "tohoku_2011",
    "thailand_floods_2011",
    "covid_2020",
    "chip_shortage_2020",
    "suez_2021",
    "energy_inflation_2022",
    "ai_boom_2023",
}


def test_catalog_loads_and_is_subcritical() -> None:
    cat = load_catalog(CATALOG)
    assert TEMPLATES <= set(cat)
    assert "fiscal_stimulus" in cat
    assert "monetary_tightening" in cat
    assert "auto_production_cuts" in cat
    check_subcritical(cat, max_depth=3, p_sum_cap=0.9)
    for spec in cat.values():
        for shock in spec.composition:
            mag = shock.magnitude
            if mag.dist != "fixed" and mag.low_days is None:
                assert mag.verify is False
        for extra in spec.effects_extra:
            mag = extra.magnitude
            if not isinstance(mag, float) and mag.dist != "fixed" and mag.low_days is None:
                assert mag.verify is False
        href = spec.historic_reference
        if href is not None:
            assert href.calibrated_params.get("verify") is not True
