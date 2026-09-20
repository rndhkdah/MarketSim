"""T6.06 — ξ = I + s + n, limits to arbitrage, calm index vol, stream isolation."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.rng import RngHub
from marketsim.pricing.mispricing import (
    A_0,
    A_LEAK,
    A_RATIO_MAX,
    A_RATIO_MIN,
    CALM_INDEX_VOL_MAX,
    CALM_INDEX_VOL_MIN,
    DAYS_PER_YEAR,
    KAPPA_A,
    KAPPA_MOM,
    STREAM_FLOW,
    STREAM_NAMES,
    STREAM_NOISE,
    STREAM_SENTIMENT,
    THETA_0,
    Mispricing,
    annualised_vol,
    arb_speed,
    cap_weighted_index,
    ou_decay,
)


def _book(
    n: int = 6,
    *,
    capital: float | None = None,
    noise_scale: float | None = None,
    kappa_a: float = KAPPA_A,
    kappa_mom: float = KAPPA_MOM,
    a_leak: float = A_LEAK,
    stream_noise: str = STREAM_NOISE,
) -> Mispricing:
    idio = np.linspace(0.12, 0.18, n)
    weights = np.linspace(0.25, 0.08, n)
    return Mispricing(
        idio,
        weights,
        capital=capital,
        noise_scale=noise_scale,
        kappa_a=kappa_a,
        kappa_mom=kappa_mom,
        a_leak=a_leak,
        stream_noise=stream_noise,
    )


def test_xi_stationary_without_flow() -> None:
    hub = RngHub(1)
    book = _book()
    burn = DAYS_PER_YEAR
    live_n = DAYS_PER_YEAR * 7
    path = np.stack([book.step(hub, impact=0.0) for _ in range(burn + live_n)])
    live = path[burn:]
    assert np.allclose(live.mean(axis=0), 0.0, atol=0.08)
    mid = live.shape[0] // 2
    v1 = float(live[:mid].var())
    v2 = float(live[mid:].var())
    assert 0.5 < v1 / v2 < 2.0
    assert np.isfinite(live).all()
    assert float(np.max(np.abs(live))) < 2.0
    mid_state = book.to_state()
    hub_mid = hub.to_state()
    rest = np.stack([book.step(hub, impact=0.0) for _ in range(16)])
    clone = Mispricing.from_state(mid_state)
    hub2 = RngHub.from_state(hub_mid)
    cont = np.stack([clone.step(hub2, impact=0.0) for _ in range(16)])
    assert np.allclose(rest, cont)


def test_reversion_slows_when_arbitrage_capital_is_low() -> None:
    assert arb_speed(A_0) == pytest.approx(THETA_0)
    assert arb_speed(A_RATIO_MIN * A_0) == pytest.approx(THETA_0 * A_RATIO_MIN)
    assert arb_speed(10.0 * A_0) == pytest.approx(THETA_0 * A_RATIO_MAX)

    def remaining(capital: float) -> float:
        hub = RngHub(0)
        book = _book(3, capital=capital, noise_scale=0.0, kappa_a=0.0, kappa_mom=0.0, a_leak=0.0)
        book.n[:] = 0.5
        book.s[:] = 0.0
        for _ in range(30):
            book.step(hub, impact=0.0)
        return float(np.mean(np.abs(book.xi)))

    hi = remaining(A_0)
    lo = remaining(A_RATIO_MIN * A_0)
    assert lo > hi * 1.5
    assert hi == pytest.approx(0.5 * ou_decay(THETA_0) ** 30, rel=1e-12)
    assert lo == pytest.approx(0.5 * ou_decay(THETA_0 * A_RATIO_MIN) ** 30, rel=1e-12)

    # A loses when |ξ| widens against the short-ξ book (impact pulse, frozen n).
    hub = RngHub(0)
    book = _book(3, noise_scale=0.0, kappa_mom=0.0, kappa_a=KAPPA_A)
    book.n[:] = 0.2
    book.step(hub, impact=0.0)
    a_before = book.capital
    book.step(hub, impact=0.15)
    assert book.capital < a_before


def test_calm_regime_index_volatility_15_to_18_pct() -> None:
    hub = RngHub(2)
    book = _book()
    burn = DAYS_PER_YEAR
    n_live = DAYS_PER_YEAR * 8
    levels: list[float] = []
    for i in range(burn + n_live):
        xi = book.step(hub, impact=0.0, news=0.0, z_risk=0.0)
        if i >= burn - 1:
            levels.append(cap_weighted_index(xi, book.weights))
    lv = np.asarray(levels, dtype=float)
    rets = lv[1:] / lv[:-1] - 1.0
    vol = annualised_vol(rets)
    assert CALM_INDEX_VOL_MIN <= vol <= CALM_INDEX_VOL_MAX


def test_streams_isolated() -> None:
    assert STREAM_NAMES == (STREAM_FLOW, STREAM_SENTIMENT, STREAM_NOISE)
    hub = RngHub(123)
    draws = {name: hub.stream(name).standard_normal(32) for name in STREAM_NAMES}
    assert not np.allclose(draws[STREAM_FLOW], draws[STREAM_SENTIMENT])
    assert not np.allclose(draws[STREAM_SENTIMENT], draws[STREAM_NOISE])
    assert not np.allclose(draws[STREAM_FLOW], draws[STREAM_NOISE])

    replay = RngHub(123)
    for name, expected in draws.items():
        assert np.array_equal(replay.stream(name).standard_normal(32), expected)

    def path(*, extra: bool, stream_noise: str = STREAM_NOISE, seed: int = 7) -> np.ndarray:
        rng = RngHub(seed)
        book = _book(4, stream_noise=stream_noise)
        rows = []
        for _ in range(40):
            if extra:
                rng.stream(STREAM_FLOW).standard_normal()
                rng.stream(STREAM_SENTIMENT).standard_normal()
            rows.append(book.step(rng, impact=0.0))
        return np.stack(rows)

    quiet = path(extra=False)
    busy = path(extra=True)
    assert np.array_equal(quiet, busy)
    again = path(extra=False)
    assert np.array_equal(quiet, again)
    other_name = path(extra=False, stream_noise=STREAM_SENTIMENT)
    assert not np.allclose(quiet, other_name)
    other_seed = path(extra=False, seed=8)
    assert not np.allclose(quiet, other_seed)
