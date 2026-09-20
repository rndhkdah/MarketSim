"""T4.07 — publication lags, GDP revisions, observe() vintages."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.events.releases import (
    CPI_DAY,
    GDP_NOISE_SD,
    DataReleases,
    following_month_day_tick,
    gdp_release_tick,
)
from marketsim.real.economy import make_real_world
from marketsim.world import World


def _record(rel: DataReleases, month: int, *, cpi: float = 1.02, gdp: float = 100.0, rng=None) -> None:
    rel.record_month(
        month,
        cpi=cpi,
        unemployment=0.05,
        output=np.ones(18),
        gdp=gdp,
        trade_balance=1.0,
        govt_balance=-0.5,
        policy_rate=0.02,
        month_end_tick=month * 21 + 20,
        rng=rng,
    )


def test_cpi_not_visible_before_day_10() -> None:
    rel = DataReleases()
    _record(rel, 0, cpi=1.07)
    day9 = following_month_day_tick(0, CPI_DAY) - 1
    day10 = following_month_day_tick(0, CPI_DAY)
    vis_early = rel.visible(day9)
    assert "cpi" not in vis_early
    vis = rel.visible(day10)
    assert vis["cpi"]["m:0"] == pytest.approx(1.07)
    assert vis["cpi_latest"] == pytest.approx(1.07)
    assert "gdp" not in vis_early
    assert 0 not in vis  # unpublished truth table is private


def test_gdp_revisions_converge_to_truth() -> None:
    rel = DataReleases()
    rng = np.random.default_rng(4)
    for m in range(3):
        _record(rel, m, gdp=10.0 + m, rng=rng)
    truth = rel._quarter_gdp_truth[0]
    eps = rel._quarter_gdp_eps[0]
    assert abs(eps) > 0.0
    t0 = gdp_release_tick(0, 0)
    t1 = gdp_release_tick(0, 1)
    t2 = gdp_release_tick(0, 2)
    v0 = rel.visible(t0)["gdp"]["q:0"]
    v1 = rel.visible(t1)["gdp"]["q:0"]
    v2 = rel.visible(t2)["gdp"]["q:0"]
    assert v0 == pytest.approx(truth * (1.0 + eps))
    assert v1 == pytest.approx(truth * (1.0 + eps / 2.0))
    assert v2 == pytest.approx(truth * (1.0 + eps / 4.0))
    assert abs(v2 - truth) < abs(v1 - truth) < abs(v0 - truth)
    stored = rel.gdp_vintages(0)
    assert [v.revision for v in stored] == [0, 1, 2]
    assert len(stored) == 3


def test_vintages_stored_and_unemployment_day() -> None:
    rel = DataReleases()
    _record(rel, 0, cpi=1.0)
    rel.visible(following_month_day_tick(0, 5))
    assert any(v.series == "unemployment" for v in rel.vintages)
    assert rel.visible(following_month_day_tick(0, 4)).get("unemployment") is None


def test_observe_hides_unpublished_cpi(config_dir) -> None:
    w = make_real_world(config_dir, seed=0, check_sfc=True)
    rel = DataReleases(calendar=w.clock.calendar, codes=w.cfg.codes)
    w.modules.append(rel)
    # Process through February day 9 (tick 29); CPI of January publishes on tick 30.
    w.step(30)
    obs = w.observe("agent0")
    releases = obs.get("releases") or {}
    assert "cpi" not in releases
    w.step(1)
    obs = w.observe("agent0")
    assert "cpi" in obs["releases"]
    assert "m:0" in obs["releases"]["cpi"]
    # Poison the true series; published vintage is unchanged.
    published = obs["releases"]["cpi"]["m:0"]
    rel._month_truth[0]["cpi"] = published + 50.0
    again = w.observe("agent0")
    assert again["releases"]["cpi"]["m:0"] == pytest.approx(published)


def test_gdp_noise_scale_is_spec() -> None:
    assert GDP_NOISE_SD == pytest.approx(0.003)


def test_world_observe_without_releases_has_no_truth_leak(config_dir) -> None:
    w = World.create(config_dir, seed=1)
    w.step(5)
    obs = w.observe("x")
    assert obs["tick"] == 5
    assert "cpi" not in obs
    assert "_month_truth" not in obs
