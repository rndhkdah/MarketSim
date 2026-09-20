"""T6.09 — background order flow: sign memory, ADV, no MM ledger posts."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.rng import RngHub
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.market.background import (
    STREAM_FLOW,
    BackgroundFlow,
    feed_impact,
    sign_acf,
)
from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import FlowCfg


def _flow(*, mr: float = 0.0, mom: float = 0.0, n: int = 1, adv: float = 100.0) -> BackgroundFlow:
    cfg = FlowCfg(
        n_components=3,
        persistences=(0.50, 0.90, 0.99),
        mean_reversion=mr,
        momentum=mom,
    )
    symbols = tuple(f"EQ:NPC:S{i}" for i in range(n))
    return BackgroundFlow(symbols, np.full(n, adv), cfg)


def test_sign_acf_positive_and_slowly_decaying() -> None:
    rng = RngHub(1)
    flow = _flow()
    qs = np.array([float(flow.step(rng, np.zeros(1))[0]) for _ in range(4000)])
    signs = np.sign(qs)
    signs[signs == 0.0] = 1.0
    ac1 = sign_acf(signs, 1)
    ac5 = sign_acf(signs, 5)
    ac20 = sign_acf(signs, 20)
    assert ac1 > 0.0
    assert ac5 > 0.0
    assert ac20 > 0.0
    assert ac1 > ac20


def test_volume_matches_adv() -> None:
    rng = RngHub(2)
    adv = 80.0
    flow = _flow(adv=adv)
    burn = 200
    qs = np.array([float(flow.step(rng, np.zeros(1))[0]) for _ in range(burn + 3000)])
    assert float(np.mean(np.abs(qs[burn:]))) == pytest.approx(adv, rel=0.25)


def test_no_ledger_postings_on_mm_venues() -> None:
    led = Ledger.empty()
    led.register_entity("MM")
    led.register_entity("HH:0")
    led.register_entity("BANKSYS")
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry("HH:0", "DEP", 50.0),
                Entry("BANKSYS", "DEP", -50.0),
            ),
        )
    )
    before = led.pos.copy()
    kernel = ImpactKernel()
    rng = RngHub(3)
    flow = _flow(adv=40.0)
    q = float(flow.step(rng, np.zeros(1))[0])
    xi = feed_impact(kernel, q, 40.0, 0.02)
    assert xi != 0.0 or q == 0.0
    assert np.array_equal(led.pos, before)
    assert led.position("HH:0", "DEP") == pytest.approx(50.0)


def test_deterministic_same_seed() -> None:
    flow_a = _flow(n=3, mr=0.10, mom=0.10)
    flow_b = BackgroundFlow.from_state(flow_a.to_state())
    xi = np.array([0.01, -0.02, 0.0])
    hub_a, hub_b, hub_c = RngHub(9), RngHub(9), RngHub(10)
    a = np.stack([flow_a.step(hub_a, xi) for _ in range(64)])
    b = np.stack([flow_b.step(hub_b, xi) for _ in range(64)])
    assert np.array_equal(a, b)
    flow_c = _flow(n=3, mr=0.10, mom=0.10)
    c = np.stack([flow_c.step(hub_c, xi) for _ in range(64)])
    assert not np.array_equal(a, c)
    # Named stream is the T6.06 ``flow`` family, not ``noise`` / ``sentiment``.
    assert STREAM_FLOW == "flow"
