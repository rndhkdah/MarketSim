"""T7.14 — end-to-end multi-agent training smoke (§7.1 gate 1)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pettingzoo")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parents[2]


def _load():
    path = ROOT / "examples" / "train_smoke.py"
    spec = importlib.util.spec_from_file_location("examples.train_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_clean_summary(summary: dict, *, ticks: int, module) -> None:
    assert summary["ticks"] == ticks
    assert summary["requested_ticks"] == ticks
    assert list(summary["random_agents"]) == list(module.RANDOM_AGENT_IDS)
    assert list(summary["scripted_agents"]) == list(module.SCRIPTED_AGENT_IDS)
    assert len(summary["random_agents"]) == 4
    assert len(summary["scripted_agents"]) == 2
    assert summary["invariants"] == module.zero_invariants()
    assert all(summary["invariants"][key] == 0 for key in module.INVARIANT_KEYS)


def test_training_smoke_short(config_dir: Path) -> None:
    """CI smoke: same 4 random + 2 scripted ids as gate 1, a handful of days."""
    module = _load()
    ticks = module.SHORT_SMOKE_TICKS
    assert 8 <= ticks <= 20
    summary = module.run_training_smoke(ticks=ticks, seed=module.DEFAULT_SEED, config_dir=config_dir)
    _assert_clean_summary(summary, ticks=ticks, module=module)


@pytest.mark.slow
def test_training_smoke_gate1(config_dir: Path) -> None:
    """§7.1 gate 1: 50,000 ticks, invariants on, counters all zero."""
    module = _load()
    ticks = module.GATE1_TICKS
    assert ticks == 50_000
    summary = module.run_training_smoke(ticks=ticks, seed=module.DEFAULT_SEED, config_dir=config_dir)
    _assert_clean_summary(summary, ticks=ticks, module=module)
