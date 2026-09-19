from __future__ import annotations

import pytest

from sweep import all_cells, run_bounded, run_stochastic

SEEDS = (1, 2, 3)
TOP5_MUST = {"CONSTRUCT", "CAPGOODS"}
BOTTOM6_MUST = {"INSURANCE", "TELECOM", "BANKS"}


@pytest.mark.validation
def test_sweep_cells_bounded(cfg, config_dir) -> None:
    for cell in all_cells(cfg):
        rec = run_bounded(config_dir, cell, months=120, pi_star=0.0)
        assert rec["finite"] == 1.0
        assert rec["max_abs_gap"] < 1e-6


@pytest.mark.slow
@pytest.mark.validation
def test_stochastic_100y_bounded(config_dir) -> None:
    for seed in SEEDS:
        rec = run_stochastic(config_dir, seed, months=1200, pi_star=0.02)
        assert rec["finite"]
        assert rec["max_abs_gap"] < 0.10
        assert 0.01 < rec["u_min"] and rec["u_max"] < 0.12
        assert rec["envelope_ratio"] < 2.0


@pytest.mark.slow
@pytest.mark.validation
@pytest.mark.xfail(
    strict=True,
    reason="QUESTIONS T2.24: SEMIS not #1 sd(yoy); CONSTRUCT not top-5; TELECOM not always bottom-6",
)
def test_volatility_ranking(config_dir) -> None:
    ranks = []
    for seed in SEEDS:
        rec = run_stochastic(config_dir, seed, months=1200, pi_star=0.02)
        ranks.append(rec["rank"])
        assert rec["rank"][0] == "SEMIS"
        assert TOP5_MUST <= set(rec["rank"][:5])
        assert BOTTOM6_MUST <= set(rec["rank"][-6:])
    assert all(r[0] == "SEMIS" for r in ranks)
