"""T8.03 — quality / brand in the goods-market share function."""

from __future__ import annotations

import pytest

from marketsim.firms.firm import FirmsFile
from marketsim.firms.goods_market import (
    QUALITY_DECAY_M,
    QUALITY_GAIN,
    QUALITY_MAX,
    QUALITY_MIN,
    SellerQuote,
    _target_shares,
    allocate,
    evolve_quality,
)


def _cfg() -> FirmsFile:
    return FirmsFile()


def _pair(quality: float) -> list[SellerQuote]:
    return [
        SellerQuote("npc", 80.0, 1.0, 80.0, 0.8, quality=quality),
        SellerQuote("a", 20.0, 1.0, 20.0, 0.2, quality=quality),
    ]


def _converge(quotes: list[SellerQuote], demand: float = 100.0, months: int = 40) -> list[float]:
    cfg = _cfg()
    for _ in range(months):
        for q in quotes:
            q.avail = q.capacity
        allocate(quotes, demand, cfg)
    return [q.share for q in quotes]


def test_seller_quote_quality_defaults_to_one() -> None:
    q = SellerQuote("npc", 80.0, 1.0, 80.0, 0.8)
    assert q.quality == 1.0


@pytest.mark.parametrize("quality", [1.0, 1.5])
def test_equal_quality_shares_match_today(quality: float) -> None:
    """Equal Q factors out of σ*, so hybrid ≈ aggregate is unchanged vs Q = 1."""
    today = _pair(1.0)
    equal = _pair(quality)
    gm = _cfg().goods_market
    star_today = _target_shares(today, 1.0, gm.epsilon_s, gm.availability_kappa)
    star_equal = _target_shares(equal, 1.0, gm.epsilon_s, gm.availability_kappa)
    assert star_equal == pytest.approx(star_today, abs=1e-12)
    assert _converge(equal) == pytest.approx(_converge(today), abs=1e-9)
    assert equal[0].share == pytest.approx(0.8, abs=1e-6)
    assert equal[1].share == pytest.approx(0.2, abs=1e-6)


def test_higher_quality_gains_share_at_equal_price() -> None:
    quotes = [
        SellerQuote("lo", 50.0, 1.0, 50.0, 0.5, quality=1.0),
        SellerQuote("hi", 50.0, 1.0, 50.0, 0.5, quality=1.5),
    ]
    gm = _cfg().goods_market
    star = _target_shares(quotes, 1.0, gm.epsilon_s, gm.availability_kappa)
    assert star[1] > star[0]
    shares = _converge(quotes, demand=80.0)
    assert shares[1] > shares[0]
    assert shares[1] > 0.5
    # φ = 1 → σ*_hi / σ*_lo = Q_hi / Q_lo = 1.5 at equal capacity and price.
    assert star[1] / star[0] == pytest.approx(1.5, abs=1e-12)


def test_quality_decays_toward_one_without_spend() -> None:
    q = evolve_quality(2.0, rnd_spend=0.0, marketing_spend=0.0)
    assert q == pytest.approx(1.0 + (2.0 - 1.0) * (1.0 - QUALITY_DECAY_M))
    assert 1.0 < q < 2.0
    q2 = evolve_quality(q)
    assert q2 < q
    assert evolve_quality(1.0) == pytest.approx(1.0)


def test_rnd_and_marketing_spend_raise_quality() -> None:
    from_rnd = evolve_quality(1.0, rnd_spend=10.0, marketing_spend=0.0)
    from_mkt = evolve_quality(1.0, rnd_spend=0.0, marketing_spend=10.0)
    from_both = evolve_quality(1.0, rnd_spend=10.0, marketing_spend=10.0)
    assert from_rnd == pytest.approx(1.0 + QUALITY_GAIN * 10.0)
    assert from_rnd == pytest.approx(from_mkt)
    assert from_both > from_rnd
    assert from_both == pytest.approx(1.0 + QUALITY_GAIN * 20.0)


def test_quality_clips_to_bounds() -> None:
    assert evolve_quality(QUALITY_MAX, rnd_spend=1.0e9, marketing_spend=1.0e9) == QUALITY_MAX
    assert evolve_quality(0.0) == QUALITY_MIN
    assert QUALITY_MIN <= evolve_quality(QUALITY_MIN) <= QUALITY_MAX
