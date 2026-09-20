"""T3.01 — regional geometry."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from marketsim.core.config import Config, load_config
from marketsim.core.errors import ConfigError
from marketsim.regions.geometry import RegionsConfig, build_geometry, target_capacity_share


def test_capacity_shares_sum_to_one(cfg: Config) -> None:
    assert cfg.regions is not None
    geo = build_geometry(cfg.regions, cfg.codes)
    assert geo.capacity_share.shape == (geo.n_regions, len(cfg.codes))
    assert np.allclose(geo.capacity_share.sum(axis=0), 1.0, atol=1e-12)
    # Omitted LQ is 1.0 — CAPITAL SOFTWARE > INDUSTRIAL SOFTWARE.
    si = geo.sector_codes.index("SOFTWARE")
    ri_cap = geo.codes.index("CAPITAL")
    ri_ind = geo.codes.index("INDUSTRIAL")
    assert geo.location_quotient[ri_cap, si] == pytest.approx(1.6)
    assert geo.location_quotient[ri_ind, si] == pytest.approx(1.0)
    assert geo.capacity_share[ri_cap, si] > geo.capacity_share[ri_ind, si]


def test_links_are_symmetric(cfg: Config) -> None:
    assert cfg.regions is not None
    geo = build_geometry(cfg.regions, cfg.codes)
    assert np.allclose(geo.cost, geo.cost.T, atol=1e-15)
    assert np.allclose(geo.capacity_mult, geo.capacity_mult.T, atol=1e-15)
    assert np.allclose(np.diag(geo.cost), 0.0)
    i, j = geo.codes.index("CAPITAL"), geo.codes.index("INDUSTRIAL")
    assert geo.cost[i, j] == pytest.approx(0.04)
    assert geo.capacity_mult[i, j] == pytest.approx(2.0)


def test_masks(cfg: Config) -> None:
    assert cfg.regions is not None
    geo = build_geometry(cfg.regions, cfg.codes)
    full = geo.mask()
    assert full.shape == (3, 18)
    assert bool(full.all())
    one = geo.mask(regions=["RESOURCE"], sectors=["ENERGY", "MATERIALS"])
    assert one.sum() == 2
    assert one[geo.codes.index("RESOURCE"), geo.sector_codes.index("ENERGY")]
    assert not one[geo.codes.index("CAPITAL"), geo.sector_codes.index("ENERGY")]


def test_r1_file_accepted(tmp_path, cfg: Config) -> None:
    raw = {
        "regions": [{"code": "NATIONAL", "population_share": 1.0, "wage_level": 1.0}],
        "location_quotient": {},
        "tradability": {},
        "armington_sigma": {"default": 2.0},
        "links": [],
        "trade": {"tau_share_m": 6, "home_bias": 1.5, "gravity_theta": 8.0, "availability_kappa": 1.0},
        "migration": {"enabled": False, "rate_per_pp_per_year": 0.0},
    }
    path = tmp_path / "regions.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    parsed = RegionsConfig.model_validate(raw)
    geo = build_geometry(parsed, cfg.codes)
    assert geo.n_regions == 1
    assert geo.codes == ("NATIONAL",)
    assert np.allclose(geo.capacity_share, 1.0)
    assert geo.cost.shape == (1, 1)


def test_unknown_lq_sector_rejected(cfg: Config) -> None:
    assert cfg.regions is not None
    dumped = cfg.regions.model_dump()
    dumped["location_quotient"] = {"CAPITAL": {"NOTASECTOR": 1.2}}
    bad = RegionsConfig.model_validate(dumped)
    with pytest.raises(ConfigError, match="NOTASECTOR"):
        build_geometry(bad, cfg.codes)


def test_target_capacity_share_formula() -> None:
    pop = np.array([0.5, 0.5])
    lq = np.array([[2.0, 1.0], [1.0, 1.0]])
    share = target_capacity_share(pop, lq)
    assert share[0, 0] == pytest.approx(2.0 / 3.0)
    assert share[1, 0] == pytest.approx(1.0 / 3.0)
    assert np.allclose(share.sum(0), 1.0)


def test_load_config_builds_shipped_regions(config_dir) -> None:
    loaded = load_config(config_dir)
    assert loaded.regions is not None
    assert [r.code for r in loaded.regions.regions] == ["CAPITAL", "INDUSTRIAL", "RESOURCE"]
