"""T3.10 — need shapes and budget scaling."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.demand.packages import (
    evaluate_shape,
    luxury,
    normal,
    plateau,
    scale_budget,
    survival,
    vanish,
)


def test_survival_formula_three_points() -> None:
    v_max, y_s = 0.12, 0.35
    for y in (0.0, y_s, 2.0):
        got = float(survival(y, v_max, y_s))
        assert got == pytest.approx(v_max * (1.0 - np.exp(-y / y_s)), abs=1e-12)
        assert float(evaluate_shape("survival", {"v_max": v_max, "y_s": y_s}, y)) == pytest.approx(
            got, abs=1e-12
        )


def test_plateau_formula_three_points() -> None:
    v_max, y_p = 0.10, 1.00
    for y in (0.5, 1.0, 2.0):
        got = float(plateau(y, v_max, y_p))
        assert got == pytest.approx(v_max * min(1.0, y / y_p), abs=1e-12)
        assert float(evaluate_shape("plateau", {"v_max": v_max, "y_p": y_p}, y)) == pytest.approx(
            got, abs=1e-12
        )


def test_normal_formula_three_points() -> None:
    b = 0.08
    for y in (0.5, 1.0, 1.5):
        got = float(normal(y, b))
        assert got == pytest.approx(b * y, abs=1e-12)
        assert float(evaluate_shape("normal", {"b": b}, y)) == pytest.approx(got, abs=1e-12)


def test_luxury_formula_three_points() -> None:
    b, y_th, gamma = 0.06, 0.80, 1.5
    for y in (0.5, 0.80, 2.0):
        got = float(luxury(y, b, y_th, gamma))
        assert got == pytest.approx(b * max(0.0, y - y_th) ** gamma, abs=1e-12)
        assert float(
            evaluate_shape("luxury", {"b": b, "y_th": y_th, "gamma": gamma}, y)
        ) == pytest.approx(got, abs=1e-12)


def test_vanish_peaks_at_ypk_and_vanishes() -> None:
    v_pk, y_pk = 0.04, 0.40
    grid = np.linspace(1e-6, 8.0, 801)
    vals = vanish(grid, v_pk, y_pk)
    peak_y = float(grid[int(np.argmax(vals))])
    assert peak_y == pytest.approx(y_pk, abs=grid[1] - grid[0])
    assert float(vanish(y_pk, v_pk, y_pk)) == pytest.approx(v_pk, abs=1e-12)
    assert float(vanish(20.0 * y_pk, v_pk, y_pk)) < 0.05 * v_pk
    assert float(evaluate_shape("vanish", {"v_pk": v_pk, "y_pk": y_pk}, y_pk)) == pytest.approx(
        v_pk, abs=1e-12
    )


def test_budget_exhausted_exactly() -> None:
    values = np.array([0.20, 0.05, 0.10, 0.08, 0.03])
    shapes = ("survival", "vanish", "plateau", "normal", "luxury")
    for budget in (0.05, 0.30, 0.80, 1.50):
        out = scale_budget(values, shapes, budget)
        assert float(out.sum()) == pytest.approx(budget, abs=1e-12)
        assert np.all(out >= -1e-15)


def test_survival_served_first() -> None:
    values = np.array([0.40, 0.30, 0.20])
    shapes = ("survival", "normal", "luxury")
    tight = scale_budget(values, shapes, 0.25)
    assert tight[0] == pytest.approx(0.25, abs=1e-12)
    assert tight[1] == pytest.approx(0.0, abs=1e-12)
    assert tight[2] == pytest.approx(0.0, abs=1e-12)

    mid = scale_budget(values, shapes, 0.60)
    assert mid[0] == pytest.approx(0.40, abs=1e-12)
    assert mid[1] + mid[2] == pytest.approx(0.20, abs=1e-12)
    assert mid[1] / mid[2] == pytest.approx(values[1] / values[2], abs=1e-12)
