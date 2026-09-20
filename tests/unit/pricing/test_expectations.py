from __future__ import annotations

import numpy as np
import pytest

from marketsim.layer1.build_io import CODES
from marketsim.pricing.expectations import (
    G_LR_ANCHOR,
    TAU_EE_M,
    EarningsExpectations,
    anchored_g_lr,
)


class _PoisonBooks:
    """Published snapshot plus a truth field the updater must never touch."""

    def __init__(
        self,
        published_profit: np.ndarray,
        published_growth: np.ndarray,
        truth_profit: np.ndarray,
        truth_growth: np.ndarray | None = None,
    ) -> None:
        self.published_profit = np.asarray(published_profit, dtype=float)
        self.published_growth = np.asarray(published_growth, dtype=float)
        self._truth_profit = np.asarray(truth_profit, dtype=float)
        self._truth_growth = (
            np.zeros_like(self.published_growth)
            if truth_growth is None
            else np.asarray(truth_growth, dtype=float)
        )
        self.truth_reads = 0

    @property
    def truth_profit(self) -> np.ndarray:
        self.truth_reads += 1
        return self._truth_profit

    @property
    def truth_growth(self) -> np.ndarray:
        self.truth_reads += 1
        return self._truth_growth

    @property
    def profit(self) -> np.ndarray:
        self.truth_reads += 1
        return self._truth_profit

    @property
    def growth(self) -> np.ndarray:
        self.truth_reads += 1
        return self._truth_growth


class _PoisonTape:
    """Published vintage map plus a hidden truth table (never a published key)."""

    def __init__(
        self,
        published: dict[int, np.ndarray],
        published_growth: dict[int, np.ndarray] | np.ndarray,
        truth: dict[int, np.ndarray],
    ) -> None:
        self.published = {int(m): np.asarray(v, dtype=float) for m, v in published.items()}
        if isinstance(published_growth, dict):
            self.published_growth = {int(m): np.asarray(v, dtype=float) for m, v in published_growth.items()}
        else:
            self.published_growth = np.asarray(published_growth, dtype=float)
        n = next(iter(truth.values())).shape[0]
        table = np.full((max(truth) + 1, n), np.nan, dtype=float)
        for m, v in truth.items():
            table[int(m)] = np.asarray(v, dtype=float)
        self._truth_table = table
        self.truth_reads = 0

    @property
    def truth(self) -> np.ndarray:
        self.truth_reads += 1
        return self._truth_table


def test_poison_unpublished_truth_profit_leaves_ee_unchanged() -> None:
    n = len(CODES)
    published = np.linspace(1.0, 1.7, n)
    growth = np.full(n, 0.03)
    books = _PoisonBooks(published, growth, truth_profit=published * 50.0, truth_growth=np.full(n, 0.9))
    a = EarningsExpectations(published)
    a.update_from_books(books, 0.02)
    books._truth_profit[:] = 1e9
    books._truth_growth[:] = 8.0
    b = EarningsExpectations(published)
    b.update_from_books(books, 0.02)
    assert books.truth_reads == 0
    assert np.array_equal(a.ee, b.ee)
    assert np.array_equal(a.g_lr, b.g_lr)
    assert np.allclose(a.g_lr, G_LR_ANCHOR * growth)
    books.published_profit = published * 2.0
    c = EarningsExpectations(published)
    c.update_from_books(books, 0.02)
    assert not np.allclose(c.ee, a.ee)
    assert books.truth_reads == 0


def test_unpublished_months_absent_cannot_leak() -> None:
    n = len(CODES)
    x0 = np.linspace(1.0, 2.0, n)
    x1 = x0 * float(np.exp(0.02 / 12.0))
    unpublished = x0 * 80.0
    growth = np.zeros(n)
    tape = _PoisonTape(
        published={0: x0, 1: x1},
        published_growth={0: growth, 1: growth},
        truth={0: x0, 1: x1, 2: unpublished},
    )
    exp = EarningsExpectations(x0)
    assert exp.update_from_vintages(tape, 0, 0.02) is not None
    assert exp.update_from_vintages(tape, 1, 0.02) is not None
    ee_pub = exp.ee.copy()
    g_pub = exp.g_lr.copy()
    assert exp.update_from_vintages(tape, 2, 0.02) is None
    tape._truth_table[2] = 1e9
    assert exp.update_from_vintages(tape, 2, 0.02) is None
    assert tape.truth_reads == 0
    assert np.array_equal(exp.ee, ee_pub)
    assert np.array_equal(exp.g_lr, g_pub)
    leaked = EarningsExpectations.from_state(exp.to_state())
    leaked.update(unpublished, growth, 0.02)
    assert not np.allclose(leaked.ee, ee_pub)


@pytest.mark.parametrize("pi_star", [0.0, 0.02])
def test_exact_under_steady_pi_star(pi_star: float) -> None:
    n = len(CODES)
    x0 = np.linspace(1.0, 2.0, n)
    exp = EarningsExpectations(x0)
    growth = np.zeros(n)
    assert np.array_equal(exp.ee, x0)
    for t in range(1, int(TAU_EE_M) * 2 + 1):
        xt = x0 * float(np.exp(pi_star * t / 12.0))
        ee, g_lr = exp.update(xt, growth, pi_star)
        assert np.allclose(ee, xt, atol=1e-12, rtol=0.0)
        real = ee / float(np.exp(pi_star * t / 12.0))
        assert np.allclose(real, x0, atol=1e-12, rtol=0.0)
        assert np.array_equal(g_lr, np.zeros(n))
        assert np.array_equal(anchored_g_lr(growth), np.zeros(n))


def test_state_round_trip() -> None:
    n = len(CODES)
    exp = EarningsExpectations(np.linspace(1.0, 2.0, n))
    exp.update(np.linspace(1.1, 2.2, n), np.full(n, 0.04), 0.02)
    restored = EarningsExpectations.from_state(exp.to_state())
    assert np.array_equal(restored.ee, exp.ee)
    assert np.array_equal(restored.g_lr, exp.g_lr)
    assert restored.ee.shape == (n,)


def test_vector_length_default_18_or_caller_n() -> None:
    assert EarningsExpectations().ee.shape == (len(CODES),)
    assert EarningsExpectations().ee.shape == (18,)
    assert EarningsExpectations(n=5).ee.shape == (5,)
    assert EarningsExpectations(np.ones(7)).ee.shape == (7,)
    assert EarningsExpectations(n=5).g_lr.shape == (5,)
