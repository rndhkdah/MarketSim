"""T4.13 — no verify:true remains in the shipped catalog."""

from __future__ import annotations

from pathlib import Path

from marketsim.events.schema import DistSpec, load_catalog

CATALOG = Path("config/events")


def _verify_flags(spec) -> list[bool]:
    flags: list[bool] = []
    for shock in spec.composition:
        flags.append(bool(shock.magnitude.verify))
    for extra in spec.effects_extra:
        mag = extra.magnitude
        if isinstance(mag, DistSpec):
            flags.append(bool(mag.verify))
    href = spec.historic_reference
    if href is not None:
        flags.append(bool(href.calibrated_params.get("verify")))
    return flags


def test_catalog_has_no_verify_true() -> None:
    cat = load_catalog(CATALOG)
    for eid, spec in cat.items():
        assert not any(_verify_flags(spec)), eid
    for path in CATALOG.glob("*.yaml"):
        if path.name.startswith("_"):
            continue
        assert "verify: true" not in path.read_text()
