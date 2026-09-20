from __future__ import annotations

import numpy as np
import pytest

from marketsim.pricing.bonds import BOND_INDEX_DURATION, bond_index_return
from marketsim.pricing.curve import (
    HORIZON_M,
    LAMBDA_M,
    TermPremium,
    expected_policy_path,
    y10,
)


@pytest.mark.parametrize("r_n", [0.0, 0.01])
@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_plus_100bp_policy_move_passes_35bp_into_y10(r_n: float, pi_star: float) -> None:
    # Layer-1 DISC_PASS is reproduced by mean(λ^h), not imported or asserted.
    dy = y10(r_n + pi_star + 0.01, r_n, pi_star, tp=0.0) - y10(r_n + pi_star, r_n, pi_star, tp=0.0)
    assert dy == pytest.approx(0.0035, abs=0.0005)


def test_bond_index_return_duration_times_dy_plus_carry() -> None:
    y_prev = 0.04
    dy = 0.01
    carry = y_prev / 12.0
    ret = bond_index_return(y_prev, y_prev + dy)
    assert -BOND_INDEX_DURATION * dy == pytest.approx(-0.07)
    assert ret == pytest.approx(-BOND_INDEX_DURATION * dy + carry)
    assert bond_index_return(y_prev, y_prev + dy, carry=0.002) == pytest.approx(-0.07 + 0.002)


def test_carry_accrues_when_yield_unchanged() -> None:
    y = 0.042
    carry = y / 12.0
    ret = bond_index_return(y, y)
    assert ret == pytest.approx(carry)
    assert ret > 0.0
    assert bond_index_return(y, y, carry=0.003) == pytest.approx(0.003)


def test_expected_policy_path_length_is_120() -> None:
    path = expected_policy_path(0.05, 0.01, 0.02)
    assert path.shape == (HORIZON_M,)
    assert HORIZON_M == 120
    assert path[0] == pytest.approx(0.05)


def test_expected_policy_path_flat_at_neutral() -> None:
    r_n, pi_star = 0.01, 0.02
    r_t = r_n + pi_star
    path = expected_policy_path(r_t, r_n, pi_star)
    assert np.allclose(path, r_t)
    h = np.arange(HORIZON_M, dtype=float)
    expected = (r_n + pi_star) + (0.05 - r_n - pi_star) * (LAMBDA_M**h)
    assert np.allclose(expected_policy_path(0.05, r_n, pi_star), expected)


def test_y10_adds_term_premium_one_for_one() -> None:
    r_n, pi_star, tp = 0.01, 0.02, 0.005
    r_t = r_n + pi_star
    assert y10(r_t, r_n, pi_star, tp=tp) - y10(r_t, r_n, pi_star, tp=0.0) == pytest.approx(tp)


def test_term_premium_ar1_step_and_state() -> None:
    tp = TermPremium(tp=0.01)
    out = tp.step(0.02, phi=0.9, loading=0.5, shock=0.001)
    assert out == pytest.approx(0.9 * 0.01 + 0.5 * 0.02 + 0.001)
    restored = TermPremium.from_state(tp.to_state())
    assert restored.tp == pytest.approx(out)
