"""T5.07 — autopilot reuses Phase-2 functions; one-firm cell matches NPC."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import Config
from marketsim.firms.autopilot import Autopilot
from marketsim.real.labour import step_labour
from marketsim.real.production import expected_sales, plan_output


def test_plan_and_sales_and_labour_match_phase2(cfg: Config) -> None:
    ap = Autopilot(cfg)
    se, sales, inv, cover, leak, tau, backlog, k = 10.0, 9.5, 2.0, 0.5, 0.008, 3.0, 0.4, 12.0
    got = ap.plan(se, inv, cover, leak, tau, backlog, 3.0, True, k)
    want = float(
        plan_output(
            np.array([se]),
            np.array([inv]),
            np.array([cover]),
            np.array([leak]),
            np.array([tau]),
            np.array([backlog]),
            3.0,
            np.array([True]),
            np.array([k]),
        )[0]
    )
    assert got == pytest.approx(want, abs=1e-12)
    assert ap.expect_sales(se, sales) == pytest.approx(
        float(expected_sales(np.array([se]), np.array([sales]), cfg.dynamics.expectations.tau_sales_m)[0]),
        abs=1e-12,
    )
    n, w, u = ap.labour(3.0, 1.0, ell=0.4, fc=0.2, k=k, ustar=0.8, plan=got, z_sup=0.0, lf=20.0, pi_e=0.0)
    n2, w2, u2 = step_labour(
        np.array([3.0]),
        1.0,
        ell=np.array([0.4]),
        fc=np.array([0.2]),
        k=np.array([k]),
        ustar=np.array([0.8]),
        plan=np.array([got]),
        z_sup=np.array([0.0]),
        lf=20.0,
        cfg=cfg,
        pi_e=0.0,
    )
    assert n == pytest.approx(float(n2[0]), abs=1e-12)
    assert w == pytest.approx(float(w2), abs=1e-12)
    assert u == pytest.approx(float(u2), abs=1e-12)


def test_one_firm_cell_reproduces_npc_plan(cfg: Config) -> None:
    """A firm that owns the whole cell uses the same plan as the NPC mass."""
    ap = Autopilot(cfg)
    npc = (8.0, 1.5, 0.25, 0.006, 2.0, 0.0, 3.0, True, 10.0)
    assert ap.plan(*npc) == pytest.approx(
        float(
            plan_output(
                np.array([npc[0]]),
                np.array([npc[1]]),
                np.array([npc[2]]),
                np.array([npc[3]]),
                np.array([npc[4]]),
                np.array([npc[5]]),
                npc[6],
                np.array([npc[7]]),
                np.array([npc[8]]),
            )[0]
        ),
        abs=1e-9,
    )
    assert ap.payout(12.0, 40.0, 40.0, 0.5) == pytest.approx(6.0)
