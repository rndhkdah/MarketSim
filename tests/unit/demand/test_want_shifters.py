"""T3.13 — want shifters (events interface)."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import load_config
from marketsim.demand.system import TiersWantsDemand
from marketsim.demand.wants import WantShifts
from marketsim.layer1.build_io import CODES
from marketsim.real.households import household_basket
from marketsim.real.shocks import ShockBus, ar1_rho
from marketsim.real.steady_state import compute_financial_baseline, compute_real_baseline


def _tw(config_dir, io) -> tuple[object, TiersWantsDemand]:
    cfg = load_config(config_dir, {"dynamics.households.demand_mode": "tiers_wants"})
    real = compute_real_baseline(io, cfg)
    fin = compute_financial_baseline(real, cfg, io)
    tw = household_basket(real, cfg, fin)
    assert isinstance(tw, TiersWantsDemand)
    return cfg, tw


def test_eating_out_cut_spreads_and_saving_unchanged(config_dir, io) -> None:
    _, tw = _tw(config_dir, io)
    p = np.ones(len(CODES))
    q0 = tw.allocate(1.0, p, 1.0, 0.0)
    spend0 = float((p * q0).sum())
    shifts = WantShifts.at_rest(1, tw.layer.names, rho=0.9)
    shifts.inject("EATING_OUT_LEISURE", -0.5)
    tw.shifts = shifts
    q1 = tw.allocate(1.0, p, 1.0, 0.0)
    d = CODES.index("DISCRET")
    assert q1[d] < q0[d]
    assert float((p * q1).sum()) == pytest.approx(spend0, abs=1e-12)
    assert np.any(q1 > q0 + 1e-12)


def test_shock_bus_want_inject_and_decay(config_dir, io) -> None:
    cfg, tw = _tw(config_dir, io)
    bus = ShockBus.from_config(cfg)
    shifts = WantShifts.at_rest(1, tw.layer.names, ar1_rho(tw.layer.persistence_q))
    bus.attach_want_shifts(shifts)
    bus.inject("want", -0.5, targets=["EATING_OUT_LEISURE"], day=0)
    q = tw.layer.names.index("EATING_OUT_LEISURE")
    assert shifts.values[0, q] == pytest.approx(0.5, abs=1e-12)
    bus.decay()
    rho = shifts.rho
    assert shifts.values[0, q] == pytest.approx(1.0 + rho * (0.5 - 1.0), abs=1e-12)


def test_regional_inject_is_local(config_dir, io) -> None:
    _, tw = _tw(config_dir, io)
    shifts = WantShifts.at_rest(3, tw.layer.names, 0.9)
    q = tw.layer.names.index("EATING_OUT_LEISURE")
    shifts.inject("EATING_OUT_LEISURE", -0.5, regions=np.array([1]))
    assert shifts.values[0, q] == pytest.approx(1.0)
    assert shifts.values[1, q] == pytest.approx(0.5)
    assert shifts.values[2, q] == pytest.approx(1.0)
    nat = shifts.national(np.array([0.45, 0.35, 0.20]))
    assert nat[q] == pytest.approx(0.45 * 1.0 + 0.35 * 0.5 + 0.20 * 1.0)
