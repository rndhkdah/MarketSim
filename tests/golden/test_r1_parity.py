"""T3.02 gate 1: regional code at R = 1 matches the Phase-2 golden series (1e-12)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marketsim.scenarios.irf import make_economy, run_irf

GOLDEN = Path(__file__).resolve().parent / "data" / "r1_phase2.npz"
TOL = 1e-12
NS_MONTHS = 36
IRF_MONTHS = 24
PASSTHROUGH = {"dynamics.banks.mode": "passthrough"}


def _golden() -> np.lib.npyio.NpzFile:
    if not GOLDEN.is_file():
        pytest.fail(f"missing Phase-2 golden {GOLDEN}")
    return np.load(GOLDEN)


def test_live_state_is_regional(config_dir: Path) -> None:
    eco = make_economy(config_dir, pi_star=0.0, overrides=PASSTHROUGH, check_sfc=True)
    s = eco.real.S
    assert eco.R == 1
    assert eco.real.R == 1
    assert eco._x.shape == (1, s)
    assert eco._p.shape == (1, s)
    assert eco._n.shape == (1, s)
    assert eco._s_in.shape == (1, s, s)
    assert eco._w.shape == (1,)
    assert eco._lf.shape == (1,)
    assert eco._yd_e.shape == (1,)
    assert eco._wealth.shape == (1,)
    assert eco.x.shape == (s,)
    assert eco.geom.n_regions == 1
    assert eco.geom.codes == ("NATIONAL",)
    assert np.allclose(eco.geom.capacity_share, 1.0)
    assert eco.real.lf_r.shape == (1,)
    assert eco.real.lf_r[0] == pytest.approx(eco.real.LF, abs=0.0)


def test_r1_noshock_matches_phase2_golden(config_dir: Path) -> None:
    g = _golden()
    eco = make_economy(config_dir, pi_star=0.0, overrides=PASSTHROUGH, check_sfc=True)
    assert tuple(g["codes"]) == eco.codes
    gdp = np.zeros(NS_MONTHS)
    cpi = np.zeros(NS_MONTHS)
    u = np.zeros(NS_MONTHS)
    r = np.zeros(NS_MONTHS)
    x = np.zeros((NS_MONTHS, eco.real.S))
    p = np.zeros((NS_MONTHS, eco.real.S))
    w = np.zeros(NS_MONTHS)
    for t in range(NS_MONTHS):
        rec = eco.step_month()
        gdp[t] = rec["gdp"]
        cpi[t] = rec["cpi"]
        u[t] = rec["U"]
        r[t] = rec["r"]
        x[t] = rec["x"]
        p[t] = eco.p
        w[t] = float(eco.w)
    assert np.max(np.abs(gdp - g["ns_gdp"])) < TOL
    assert np.max(np.abs(cpi - g["ns_cpi"])) < TOL
    assert np.max(np.abs(u - g["ns_u"])) < TOL
    assert np.max(np.abs(r - g["ns_r"])) < TOL
    assert np.max(np.abs(x - g["ns_x"])) < TOL
    assert np.max(np.abs(p - g["ns_p"])) < TOL
    assert np.max(np.abs(w - g["ns_w"])) < TOL


def test_r1_demand_irf_matches_phase2_golden(config_dir: Path) -> None:
    g = _golden()
    irf = run_irf(
        "demand",
        0.02,
        months=IRF_MONTHS,
        config_dir=config_dir,
        pi_star=0.0,
        overrides=PASSTHROUGH,
        check_sfc=True,
    )
    assert np.max(np.abs(irf.gap - g["irf_gap"])) < TOL
    assert np.max(np.abs(irf.lvl - g["irf_lvl"])) < TOL
    assert np.max(np.abs(irf.x - g["irf_x"])) < TOL
    assert np.max(np.abs(irf.cpi - g["irf_cpi"])) < TOL
    assert np.max(np.abs(irf.u - g["irf_u"])) < TOL
    assert np.max(np.abs(irf.r - g["irf_r"])) < TOL


def test_r1_pi_star_2pct_matches_phase2_golden(config_dir: Path) -> None:
    g = _golden()
    eco = make_economy(config_dir, pi_star=0.02, overrides=PASSTHROUGH, check_sfc=True)
    gdp = np.zeros(12)
    cpi = np.zeros(12)
    for t in range(12):
        rec = eco.step_month()
        gdp[t] = rec["gdp"]
        cpi[t] = rec["cpi"]
    assert np.max(np.abs(gdp - g["pi2_gdp"])) < TOL
    assert np.max(np.abs(cpi - g["pi2_cpi"])) < TOL
