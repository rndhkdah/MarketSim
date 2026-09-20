"""T3.06 — national LF conserved; slump loses ≈0.1 % of LF per year per pp gap."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.regions.geometry import MigrationCfg
from marketsim.regions.labour import migrate_labour_force


def test_national_lf_conserved() -> None:
    lf = np.array([45.0, 35.0, 20.0])
    u = np.array([0.04, 0.05, 0.08])
    out = migrate_labour_force(lf, u, MigrationCfg())
    assert float(out.sum()) == pytest.approx(float(lf.sum()), abs=1e-12)


def test_slump_loses_labour_slowly() -> None:
    lf = np.array([50.0, 50.0])
    u = np.array([0.04, 0.06])  # Ū = 0.05 → slump is +1 pp vs national
    cfg = MigrationCfg(enabled=True, rate_per_pp_per_year=0.001)
    cur = lf.copy()
    for _ in range(12):
        cur = migrate_labour_force(cur, u, cfg)
    annual = (cur[1] - lf[1]) / lf[1]
    assert annual == pytest.approx(-0.001, abs=2e-4)
    assert cur[1] < lf[1]
    assert cur[0] > lf[0]


def test_disabled_is_identity() -> None:
    lf = np.array([0.4, 0.6])
    out = migrate_labour_force(lf, np.array([0.1, 0.0]), MigrationCfg(enabled=False))
    assert np.array_equal(out, lf)
