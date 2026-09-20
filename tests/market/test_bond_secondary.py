"""T6.27 — secondary bond market, NPC holders, gate 11 hook."""

from __future__ import annotations

import numpy as np
import pytest

from marketsim.core.config import Config
from marketsim.core.rng import RngHub
from marketsim.market.bond_market import (
    INSURANCE_TARGET_SHARES,
    BondSecondaryMarket,
    daily_holding_return,
    gb_bond_total_return,
    yield_from_price,
)
from marketsim.market.mm import EngineMM
from marketsim.market.venue import Order, OrderType, Side, TimeInForce
from marketsim.pricing.bond_buckets import BUCKET_ORDER, DURATION_REF_YIELD, GOVT_BUCKETS


def _face(n: float = 1_000_000.0) -> dict[str, float]:
    return {name: float(n) for name in BUCKET_ORDER}


def _market(
    cfg: Config,
    *,
    face: dict[str, float] | None = None,
    holdings: dict[str, dict[str, float]] | None = None,
    targets: dict[str, dict[str, float]] | None = None,
    turnover: float | None = None,
    sigma: float = 0.01,
    rebalance_gain: float | None = None,
) -> BondSecondaryMarket:
    return BondSecondaryMarket.from_config(
        cfg,
        face=face or _face(),
        holdings=holdings,
        targets=targets,
        fair_yield=DURATION_REF_YIELD,
        sigma=sigma,
        turnover=turnover,
        rebalance_gain=rebalance_gain,
    )


def test_engine_mm_venue_lists_four_buckets(cfg: Config) -> None:
    mkt = _market(cfg)
    assert mkt.symbols() == BUCKET_ORDER
    assert isinstance(mkt.mm, EngineMM)
    assert set(mkt.mm.symbols()) == set(BUCKET_ORDER)
    for symbol in BUCKET_ORDER:
        qte = mkt.quote(symbol)
        assert qte.bid > 0.0
        assert qte.ask > qte.bid
        assert mkt.price(symbol) == pytest.approx(1.0)
        assert mkt.bucket_yield(symbol) == pytest.approx(DURATION_REF_YIELD)


def test_agent_trades_move_yields_through_impact_and_revert(cfg: Config) -> None:
    mkt = _market(cfg)
    y0 = mkt.bucket_yield("GB_BOND")
    p0 = mkt.price("GB_BOND")
    # participation_cap × ADV = 0.25 × 0.004 × face = 1_000 units at P=1 (§6.5 / §6.2).
    qty = 1_000
    parked = mkt.submit(
        Order(
            agent_id="alice",
            side=Side.BUY,
            qty=qty,
            symbol="GB_BOND",
            order_type=OrderType.MARKET,
            tif=TimeInForce.IOC,
        )
    )
    assert parked.fills == ()
    fills = mkt.step()
    assert len(fills) == 1
    assert fills[0].taker == "alice"
    assert fills[0].qty == qty
    y1 = mkt.bucket_yield("GB_BOND")
    p1 = mkt.price("GB_BOND")
    assert p1 > p0
    assert y1 < y0
    assert y1 == pytest.approx(yield_from_price(p1, mkt.kappa["GB_BOND"], mkt.decay["GB_BOND"]))

    for _ in range(80):
        mkt.step()
    y2 = mkt.bucket_yield("GB_BOND")
    assert y2 > y1
    assert abs(y2 - y0) < abs(y1 - y0)


def test_adv_proportional_to_outstanding_face(cfg: Config) -> None:
    assert cfg.markets is not None
    turnover = cfg.markets.turnover
    face = {"GB_BILL": 1_000.0, "GB_NOTE": 2_000.0, "GB_BOND": 4_000.0, "CORP_POOL": 8_000.0}
    mkt = _market(cfg, face=face)
    for symbol, units in face.items():
        assert mkt.adv(symbol) == pytest.approx(turnover * units)
    assert mkt.adv("GB_BOND") / mkt.adv("GB_NOTE") == pytest.approx(2.0)
    mkt.set_face("GB_BOND", 8_000.0)
    assert mkt.adv("GB_BOND") == pytest.approx(turnover * 8_000.0)
    assert mkt.adv("GB_BOND") / mkt.adv("GB_BILL") == pytest.approx(8.0)


def test_npc_rebalancing_converges_to_target_shares(cfg: Config) -> None:
    face = _face(100_000.0)
    holdings = {
        "INSURANCE": {
            "GB_BILL": 0.0,
            "GB_NOTE": 10_000.0,
            "GB_BOND": 90_000.0,
            "CORP_POOL": 0.0,
        },
    }
    mkt = _market(
        cfg,
        face=face,
        holdings=holdings,
        targets={"INSURANCE": dict(INSURANCE_TARGET_SHARES)},
        turnover=0.02,
    )
    start = mkt.holder_shares("INSURANCE")
    assert start["GB_BOND"] == pytest.approx(0.90)
    for _ in range(200):
        mkt.step()
    got = mkt.holder_shares("INSURANCE")
    for symbol, share in INSURANCE_TARGET_SHARES.items():
        assert got[symbol] == pytest.approx(share, abs=0.05)
    assert got["GB_BILL"] == pytest.approx(0.0, abs=0.02)
    total = sum(mkt.holdings["INSURANCE"][b] + mkt.holdings["MM"][b] for b in BUCKET_ORDER)
    assert total == pytest.approx(sum(face.values()))


def test_row_selloff_raises_yields(cfg: Config) -> None:
    holdings = {
        "ROW": {name: 200_000.0 if name in GOVT_BUCKETS else 0.0 for name in BUCKET_ORDER},
    }
    mkt = _market(cfg, holdings=holdings)
    y0 = {name: mkt.bucket_yield(name) for name in GOVT_BUCKETS}
    mkt.apply_row_selloff(0.50)
    for name in GOVT_BUCKETS:
        assert mkt.bucket_yield(name) > y0[name]
    assert mkt.holdings["ROW"]["GB_BOND"] == pytest.approx(100_000.0)
    assert mkt.holdings["MM"]["GB_BOND"] == pytest.approx(900_000.0)


def test_gate_11_hook_is_actual_gb_bond_total_return(cfg: Config) -> None:
    mkt = _market(cfg)
    mkt.submit(
        Order(
            agent_id="bob",
            side=Side.SELL,
            qty=1_000,
            symbol="GB_BOND",
            order_type=OrderType.MARKET,
            tif=TimeInForce.IOC,
        )
    )
    p_prev = mkt.price("GB_BOND")
    mkt.step()
    expected = daily_holding_return(
        p_prev,
        mkt.price("GB_BOND"),
        mkt.kappa["GB_BOND"],
        mkt.decay["GB_BOND"],
    )
    assert gb_bond_total_return(mkt) == pytest.approx(expected)
    assert mkt.gb_bond_total_return() == pytest.approx(expected)
    assert expected < 0.0


def test_gate_11_regime_flip_standin(cfg: Config) -> None:
    """Fast stand-in: corr(eq, actual GB_BOND total return) flips demand vs supply (§6.1 / gate 11)."""

    def _episode(sign: float, seed: int) -> float:
        mkt = _market(cfg)
        y0 = mkt.bucket_yield("GB_BOND")
        rng = RngHub(seed)
        eq: list[float] = []
        br: list[float] = []
        for _ in range(64):
            z = float(rng.stream("flow").standard_normal())
            mkt.set_fair_yield("GB_BOND", max(1e-6, y0 + 0.002 * z))
            mkt.step()
            eq.append(sign * z)
            br.append(gb_bond_total_return(mkt))
        return float(np.corrcoef(eq, br)[0, 1])

    # Demand: growth + / Taylor yields + → bonds down. Supply: both risk assets down.
    assert _episode(+1.0, 11) < -0.15
    assert _episode(-1.0, 12) > +0.15
