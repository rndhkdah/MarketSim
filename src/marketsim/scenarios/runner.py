"""T4.09 — run a scripted (and optionally random) event scenario."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from marketsim.core.config import load_config
from marketsim.events.scheduler import Events
from marketsim.events.schema import load_catalog
from marketsim.scenarios.loader import ScenarioSpec, load_scenario
from marketsim.world import World


def _zero_hazards(events: Events) -> None:
    """Scripted-only mode: keep follow-ups, disable random Poisson fires."""
    from marketsim.events.schema import EventSpec

    patched: dict[str, EventSpec] = {}
    for eid, spec in events.catalog.items():
        dumped = spec.model_dump()
        dumped["hazard"] = {"base_rate_per_year": 0.0, "multipliers": []}
        patched[eid] = EventSpec.model_validate(dumped)
    events.catalog = patched


def run_scenario(
    config_dir: str | Path,
    scenario: str | Path | ScenarioSpec,
    *,
    n_ticks: int,
    seed: int | None = None,
    extra_modules: list[Any] | None = None,
) -> World:
    """Build a World, schedule scripted fires, step ``n_ticks``. Hazards optional."""
    spec = scenario if isinstance(scenario, ScenarioSpec) else load_scenario(scenario)
    cfg = load_config(config_dir)
    catalog = load_catalog(Path(config_dir) / "events")
    ev = Events.from_config(cfg, catalog=catalog)
    if not spec.hazards:
        _zero_hazards(ev)
    mods: list[Any] = [ev, *(extra_modules or [])]
    world = World.create(config_dir, seed=seed if seed is not None else spec.seed, modules=mods)
    # recreate events on the live world (create() resets modules we passed)
    live = next(m for m in world.modules if getattr(m, "name", None) == "events")
    if not spec.hazards:
        _zero_hazards(live)
    for fire in spec.events:
        world.clock.queue.schedule(
            fire.tick,
            {
                "kind": "scripted",
                "event_id": fire.event_id,
                "magnitude": fire.magnitude if not hasattr(fire.magnitude, "dist") else fire.magnitude.model_dump(),
            },
            priority=0,
        )
    world.step(n_ticks)
    return world
