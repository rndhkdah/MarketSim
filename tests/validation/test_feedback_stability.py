"""T6.19 — feedback edges stay bounded; World IRF hook is QUESTIONS T6.19."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.real.feedbacks import (
    ICR_REF,
    MPC_WEALTH_M,
    FeedbackState,
    capex_q_term,
    derp_to_cc_gap,
    icr_hire_scale,
    scale_employment_target,
    wealth_consumption,
)


def test_q_term_clipped_and_derp_in_100bp() -> None:
    raw = capex_q_term(np.e**2, q_scale=0.1, q_tobin_coef=0.3, q_clip=0.5)
    assert raw == pytest.approx(0.1 * 0.3 * 100.0 * 0.5)
    assert derp_to_cc_gap(0.01) == pytest.approx(1.0)


def test_wealth_mpc_and_smoother_leaks() -> None:
    st = FeedbackState(wealth_s=0.0)
    for _ in range(24):
        st.step_wealth(120.0)
    assert st.wealth_s == pytest.approx(120.0, rel=0.15)
    assert wealth_consumption(120.0) == pytest.approx(MPC_WEALTH_M * 120.0)
    # Leak: a one-off mark does not integrate without bound.
    st2 = FeedbackState(wealth_s=0.0)
    st2.step_wealth(1_000.0)
    for _ in range(60):
        st2.step_wealth(0.0)
    assert st2.wealth_s < 10.0


def test_icr_scales_hiring_only_when_tight() -> None:
    assert icr_hire_scale(2.0) == pytest.approx(1.0)
    assert icr_hire_scale(ICR_REF) == pytest.approx(1.0)
    tight = float(icr_hire_scale(0.75))
    assert 0.0 < tight < 1.0
    n0 = np.array([10.0, 10.0])
    n1 = scale_employment_target(n0, np.array([2.0, 0.75]))
    assert n1[0] == pytest.approx(10.0)
    assert n1[1] < 10.0


@pytest.mark.validation
def test_three_switches_together_stay_bounded() -> None:
    """Gate 7 stand-in on the maps (World IRF needs QUESTIONS T6.19)."""
    st = FeedbackState()
    q = np.ones(18)
    n_star = np.full(18, 5.0)
    for t in range(360):
        mark = 100.0 + 2.0 * np.sin(t / 12.0)
        w = st.step_wealth(mark)
        c = wealth_consumption(w)
        q_term = capex_q_term(q, q_scale=0.1, q_tobin_coef=0.3, q_clip=0.5)
        n = scale_employment_target(n_star, 1.4 + 0.2 * np.sin(t / 8.0))
        assert np.isfinite(c) and abs(c) < 10.0
        assert np.all(np.isfinite(q_term))
        assert np.all(n <= n_star + 1e-12)
        assert st.wealth_s < 200.0
