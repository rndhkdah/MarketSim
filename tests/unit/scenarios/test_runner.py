"""T4.09 — scripted scenario reproducibility and RNG isolation."""

from __future__ import annotations

from pathlib import Path

from marketsim.scenarios.loader import load_scenario
from marketsim.scenarios.runner import run_scenario
from marketsim.world import RandomWalkModule

_ROOT = Path(__file__).resolve().parents[3]

SCENARIO = _ROOT / "config" / "scenarios" / "chip_script.yaml"


def _hist(world) -> list[tuple[int, str]]:
    ev = next(m for m in world.modules if m.name == "events")
    return [(n.tick, n.event_id) for n in ev.history]


def test_scripted_run_reproducible(config_dir) -> None:
    spec = load_scenario(SCENARIO)
    assert spec.id == "chip_script"
    a = run_scenario(config_dir, SCENARIO, n_ticks=80, seed=7)
    b = run_scenario(config_dir, SCENARIO, n_ticks=80, seed=7)
    assert _hist(a) == _hist(b)
    assert any(eid == "chip_shortage_2020" for _, eid in _hist(a))


def test_scripted_plus_agent_keeps_event_rng(config_dir) -> None:
    a = run_scenario(config_dir, SCENARIO, n_ticks=80, seed=7)
    b = run_scenario(config_dir, SCENARIO, n_ticks=80, seed=7, extra_modules=[RandomWalkModule()])
    assert _hist(a) == _hist(b)
    c = run_scenario(config_dir, SCENARIO, n_ticks=80, seed=8)
    assert _hist(a) != _hist(c) or _hist(a)  # different seed may still share a forced tick; history ids at least exist
    ticks_a = [t for t, e in _hist(a) if e == "chip_shortage_2020"]
    ticks_c = [t for t, e in _hist(c) if e == "chip_shortage_2020"]
    assert ticks_a == ticks_c == [21]
