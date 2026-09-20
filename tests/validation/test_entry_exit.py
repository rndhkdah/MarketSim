"""T3.14 — NPC entry/exit (§3.5, gate 5)."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.entry_exit import step_entry_exit
from marketsim.scenarios.irf import make_economy

PASSTHROUGH = {"dynamics.banks.mode": "passthrough"}


def test_no_entry_exit_at_baseline_bitwise(config_dir) -> None:
    on = make_economy(config_dir, overrides=PASSTHROUGH, check_sfc=True)
    off = make_economy(
        config_dir,
        overrides={**PASSTHROUGH, "dynamics.entry_exit.enabled": False},
        check_sfc=True,
    )
    k_on = []
    k_off = []
    for _ in range(12):
        on.step_month()
        off.step_month()
        k_on.append(on.k.copy())
        k_off.append(off.k.copy())
        assert np.max(np.abs(on._last_scrap)) == 0.0
    for a, b in zip(k_on, k_off, strict=True):
        assert np.array_equal(a, b)


def test_exit_writeoff_passes_sfc(config_dir) -> None:
    eco = make_economy(config_dir, overrides=PASSTHROUGH, check_sfc=True)
    eco.eb_s[:] = 0.0
    eco._excess_sm[:] = 0.4
    eco.step_month()
    assert float(eco._last_scrap.sum()) > 0.0
    assert eco.month == 1


@pytest.mark.validation
def test_entry_closes_excess_within_8_years(config_dir) -> None:
    """Gate 5: permanent +10 % demand to AUTOS → excess < θ_e within 96 months."""
    eco = make_economy(config_dir, overrides=PASSTHROUGH, check_sfc=True)
    autos = eco.codes.index("AUTOS")
    bump = 0.10 * float(eco.real.flat(eco.real.C0)[autos])
    last = 0.0
    for _ in range(96):
        eco.bus.recon_demand[autos] += bump
        eco.step_month()
        last = float(eco._excess_sm[autos] - 1.0)
    assert last < eco.cfg.dynamics.entry_exit.theta_entry


def test_step_formula_matches_spec() -> None:
    from marketsim.core.config import EntryExitCfg

    cfg = EntryExitCfg()
    k = np.array([10.0, 10.0])
    ebitda = np.array([2.0, 0.2])
    rate0 = np.array([0.1, 0.1])
    sm = np.ones(2)
    out = step_entry_exit(ebitda, k, rate0, sm, cfg)
    # rates 0.2 and 0.02; first pass of smoother: s = 1 + (2-1)/12 = 1.083, excess 0.083 < 0.15
    assert out.entry_rate[0] == 0.0
    assert out.exit_rate[1] == 0.0
