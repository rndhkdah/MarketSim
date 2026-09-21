"""T7.11 — scenario packs, lever/severity curricula, evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pytest

from marketsim.scenarios.loader import ScenarioSpec, load_scenario
from marketsim.sdk.curriculum import (
    DAYS_PER_YEAR,
    LEVER_CURRICULUM,
    SEVERITY_TIERS,
    CurriculumSpec,
    bankruptcies,
    load_curriculum,
    load_evaluation_scenarios,
    load_pack_yaml,
    max_drawdown,
    net_worth,
    published_market_share,
    run_scenario_year,
    scenario_yaml_paths,
    sharpe,
)

_ROOT = Path(__file__).resolve().parents[3]

SCENARIOS = _ROOT / "config" / "scenarios"


def test_every_scenario_yaml_loads() -> None:
    paths = scenario_yaml_paths(SCENARIOS)
    assert paths, "expected shipped scenario YAML"
    for path in paths:
        loaded = load_pack_yaml(path)
        if isinstance(loaded, CurriculumSpec):
            assert loaded.lever_order == LEVER_CURRICULUM
            continue
        assert isinstance(loaded, ScenarioSpec)
        via_t409 = load_scenario(path)
        assert via_t409.id == loaded.id


def test_every_scenario_runs_one_year(config_dir: Path) -> None:
    specs = load_evaluation_scenarios(config_dir / "scenarios")
    assert specs
    runtimes: list[tuple[str, float]] = []
    for scenario_id in sorted(specs):
        t0 = perf_counter()
        metrics = run_scenario_year(config_dir, specs[scenario_id])
        runtimes.append((scenario_id, perf_counter() - t0))
        assert metrics.ticks == DAYS_PER_YEAR
        assert metrics.scenario_id == scenario_id
        for name in ("net_worth", "sharpe", "max_drawdown", "market_share", "bankruptcies"):
            assert hasattr(metrics, name)
    assert all(dt >= 0.0 for _, dt in runtimes)
    # Printed for the task report (year-run wall time is not used by the engine).
    print("year_runtimes_s", runtimes)


def test_curriculum_stages_expose_lever_order(config_dir: Path) -> None:
    pack = load_curriculum(config_dir / "scenarios")
    assert pack.lever_order == LEVER_CURRICULUM
    assert pack.lever_order == ("pricing", "production", "labour", "capex", "financing")
    unlocked: list[str] = []
    assert len(pack.stages) == len(LEVER_CURRICULUM)
    for stage, lever in zip(pack.stages, LEVER_CURRICULUM, strict=True):
        unlocked.append(lever)
        assert tuple(stage.unlocked_levers) == tuple(unlocked)
    assert tuple(tier.id for tier in pack.severity_tiers) == SEVERITY_TIERS


def test_metrics_functions_deterministic() -> None:
    drawdown_series = (100.0, 110.0, 99.0, 121.0, 80.0)
    # Constant ΔNW ⇒ expanding σ = 0 ⇒ Sharpe = mean(ΔNW) / 1 cr.
    flat_sigma = (100.0, 102.0, 104.0, 106.0)
    flags = (False, False, False, False, True)
    a_nw = net_worth(drawdown_series)
    b_nw = net_worth(drawdown_series)
    a_sh = sharpe(flat_sigma)
    b_sh = sharpe(flat_sigma)
    a_dd = max_drawdown(drawdown_series)
    b_dd = max_drawdown(drawdown_series)
    a_bk = bankruptcies(flags)
    b_bk = bankruptcies(flags)
    assert (a_nw, a_sh, a_dd, a_bk) == (b_nw, b_sh, b_dd, b_bk)
    assert a_nw == 80.0
    assert a_dd == pytest.approx((121.0 - 80.0) / 121.0)
    assert a_bk == 1
    assert a_sh == pytest.approx(2.0)
    assert published_market_share({}) == 0.0
    assert published_market_share({"reports": [{"firm_id": "acme", "books": {"market_share": 0.2}}]}, firm_id="acme") == 0.2
    assert published_market_share({"goods": {}}) == 0.0
