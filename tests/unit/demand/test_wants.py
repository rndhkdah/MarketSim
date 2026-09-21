"""T3.09 — want layer allocation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from marketsim.demand.wants import allocate_within_want, load_wants
from marketsim.layer1.build_io import CODES
from marketsim.real.steady_state import compute_real_baseline


def _layer(config_dir: Path):
    raw = yaml.safe_load((config_dir / "wants.yaml").read_text(encoding="utf-8"))
    return load_wants(raw, CODES)


def test_shares_in_simplex_and_bounds(config_dir: Path) -> None:
    layer = _layer(config_dir)
    p = np.ones(len(CODES))
    av = np.ones(len(CODES))
    sh = allocate_within_want(layer, p, av)
    for q in range(layer.n_wants):
        material = layer.m[q] >= 1e-8
        row = sh[q, material]
        assert float(sh[q].sum()) == pytest.approx(1.0, abs=1e-12)
        if row.size:
            # RAS may park a structural weight outside [min, max]; clip applies
            # only when the prior itself already lives in the box.
            if np.all(layer.m[q, material] <= layer.max_share + 1e-12):
                assert np.all(row <= layer.max_share + 1e-12)
            if np.all(layer.m[q, material] >= layer.min_share - 1e-12):
                assert np.all(row >= layer.min_share - 1e-12)


def test_cheaper_good_gains_share(config_dir: Path) -> None:
    layer = _layer(config_dir)
    food = layer.names.index("FOOD_HOME")
    staples = CODES.index("STAPLES")
    agri = CODES.index("AGRIFOOD")
    p = np.ones(len(CODES))
    av = np.ones(len(CODES))
    base = allocate_within_want(layer, p, av)
    p[staples] = 0.8
    cheap = allocate_within_want(layer, p, av)
    assert cheap[food, staples] > base[food, staples]
    assert cheap[food, agri] < base[food, agri]


def test_availability_collapse_preserves_want(config_dir: Path) -> None:
    layer = _layer(config_dir)
    food = layer.names.index("FOOD_HOME")
    staples = CODES.index("STAPLES")
    p = np.ones(len(CODES))
    av = np.ones(len(CODES))
    av[staples] = 0.05
    sh = allocate_within_want(layer, p, av)
    assert sh[food, staples] < 0.20
    assert float(sh[food].sum()) == pytest.approx(1.0, abs=1e-12)


def test_every_hh_sector_in_a_want(cfg, io, config_dir: Path) -> None:
    real = compute_real_baseline(io, cfg)
    c0 = real.flat(real.C0)
    layer = _layer(config_dir)
    present = layer.m.sum(axis=0) > 0
    for i, code in enumerate(CODES):
        if c0[i] > 1e-12:
            assert present[i], f"{code} has household demand but no want"


def test_semis_in_household_goods(config_dir: Path) -> None:
    layer = _layer(config_dir)
    hh = layer.names.index("HOUSEHOLD_GOODS")
    assert layer.m[hh, CODES.index("SEMIS")] > 0


def test_two_shape_rule(cfg, io, config_dir: Path) -> None:
    """Every HH-facing sector sits in ≥ 2 wants whose YAML shape keys differ."""
    real = compute_real_baseline(io, cfg)
    c0 = real.flat(real.C0)
    layer = _layer(config_dir)
    for i, code in enumerate(CODES):
        if c0[i] <= 1e-12:
            continue
        shapes = {layer.shapes[q] for q in range(layer.n_wants) if layer.m[q, i] > 0}
        assert len(shapes) >= 2, f"{code} shapes={sorted(shapes)}"
