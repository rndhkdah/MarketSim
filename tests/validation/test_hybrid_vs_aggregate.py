"""T5.16 — N autopilot firms on a cell reproduce the NPC plan (gate 1 identity)."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import Config
from marketsim.firms.autopilot import Autopilot
from marketsim.real.production import plan_output


def _npc_plan(se, inv, cover, leak, tau, backlog, tau_b, k) -> float:
    return float(
        plan_output(
            np.array([se]),
            np.array([inv]),
            np.array([cover]),
            np.array([leak]),
            np.array([tau]),
            np.array([backlog]),
            tau_b,
            np.array([True]),
            np.array([k]),
        )[0]
    )


@pytest.mark.validation
@pytest.mark.parametrize("n_firms", [1, 5])
def test_split_firms_match_npc_plan(cfg: Config, n_firms: int) -> None:
    ap = Autopilot(cfg)
    se, inv, cover, leak, tau, backlog, tau_b, k = 12.0, 3.0, 0.25, 0.008, 2.0, 0.6, 3.0, 15.0
    npc = _npc_plan(se, inv, cover, leak, tau, backlog, tau_b, k)
    parts = [
        ap.plan(se / n_firms, inv / n_firms, cover, leak, tau, backlog / n_firms, tau_b, True, k / n_firms)
        for _ in range(n_firms)
    ]
    assert sum(parts) == pytest.approx(npc, abs=1e-9)


@pytest.mark.validation
@pytest.mark.slow
@pytest.mark.parametrize("n_firms", [1, 5])
def test_hybrid_path_stays_within_gate1_band(cfg: Config, n_firms: int) -> None:
    """10-year cell replica: output path vs NPC within 1 % (gate 1 GDP proxy)."""
    ap = Autopilot(cfg)
    se = inv = 10.0
    k = 20.0
    npc_x = []
    hyb_x = []
    for _ in range(120):
        npc = ap.plan(se, inv, 0.25, 0.008, 2.0, 0.0, 3.0, True, k)
        hyb = sum(ap.plan(se / n_firms, inv / n_firms, 0.25, 0.008, 2.0, 0.0, 3.0, True, k / n_firms) for _ in range(n_firms))
        npc_x.append(npc)
        hyb_x.append(hyb)
        se = 0.9 * se + 0.1 * npc
        inv = inv + npc - se
    npc_a = np.asarray(npc_x)
    hyb_a = np.asarray(hyb_x)
    assert np.max(np.abs(hyb_a / np.maximum(npc_a, 1e-12) - 1.0)) < 0.01
