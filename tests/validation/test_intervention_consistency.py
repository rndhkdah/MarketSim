"""T8.10 — intervention-consistency experiments (identified in-sim only).

These tests hold other agents' actions fixed (the do-operator). They do not
identify a real-world policy counterfactual. See
``claude/plan/reports/tier3-counterfactual.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marketsim.market.impact import ImpactKernel, delta_impact
from marketsim.market.instruments import ImpactCfg
from marketsim.world import World

pytestmark = pytest.mark.validation

# §6.4 / markets.yaml defaults. Named so the test does not retune.
_Y = 0.8
_DELTA = 0.5
_SIGMA = 0.02  # daily vol (decimal)
_ADV = 100.0  # cr / day
_Q_SMALL = 4.0  # cr
_Q_LARGE = 16.0  # cr


def _cfg() -> ImpactCfg:
    return ImpactCfg(Y=_Y, delta=_DELTA)


def test_unrelated_name_invariant_when_only_one_symbol_is_traded() -> None:
    """Same seed/state; q on name 0 must not move name 1's impact state."""
    treated = ImpactKernel(_cfg(), shape=(2,))
    control = ImpactKernel(_cfg(), shape=(2,))
    q_treat = np.array([_Q_SMALL, 0.0])
    q_ctrl = np.array([0.0, 0.0])
    adv = np.full(2, _ADV)
    sig = np.full(2, _SIGMA)
    xi_t = np.asarray(treated.step(q_treat, adv, sig), dtype=float)
    xi_c = np.asarray(control.step(q_ctrl, adv, sig), dtype=float)
    assert xi_t[0] != pytest.approx(xi_c[0], abs=1e-12)
    assert xi_t[1] == pytest.approx(xi_c[1], abs=1e-12)
    np.testing.assert_allclose(treated.I[1], control.I[1], atol=1e-12)


def test_dose_response_monotonic_in_signed_volume() -> None:
    """Larger |q| → larger |ΔI| at fixed ADV (concave, still monotone)."""
    d_small = float(delta_impact(_Q_SMALL, _ADV, _SIGMA, Y=_Y, delta=_DELTA))
    d_large = float(delta_impact(_Q_LARGE, _ADV, _SIGMA, Y=_Y, delta=_DELTA))
    d_sell = float(delta_impact(-_Q_LARGE, _ADV, _SIGMA, Y=_Y, delta=_DELTA))
    assert d_small > 0.0
    assert d_large > d_small
    assert d_sell == pytest.approx(-d_large, abs=1e-12)
    # Concavity: 4× volume is not 4× impact when δ = 0.5.
    assert d_large < 4.0 * d_small


def test_same_seed_empty_world_hash_ignores_unrouted_orders(config_dir: Path) -> None:
    """Empty ``World`` has no book: submit-then-step equals idle-step after PUBLISH.

    This is a negative result — it is *not* evidence that orders have no price
    impact. Impact lives on a venue (T8.10 report §3).
    """
    idle = World.create(config_dir, seed=21)
    treated = World.create(config_dir, seed=21)
    treated.submit("alice", {"symbol": "EQ:NPC:AUTOS", "qty": _Q_SMALL, "side": "buy"})
    idle.step(1)
    treated.step(1)
    assert idle.state_hash() == treated.state_hash()


def test_same_seed_same_actions_same_hash(config_dir: Path) -> None:
    """Baseline determinism: same seed + same (empty) actions → same hash."""
    a = World.create(config_dir, seed=3)
    b = World.create(config_dir, seed=3)
    a.step(4)
    b.step(4)
    assert a.state_hash() == b.state_hash()
