"""T6.28 — QE / QT postings, gates 13–14, remittance, credit easing."""

from __future__ import annotations

import math

import pytest

from marketsim.core.config import Config
from marketsim.ledger.journal import Ledger
from marketsim.ledger.sfc import assert_consistent
from marketsim.market.bond_market import BondSecondaryMarket
from marketsim.market.corp_pool import CorpPool
from marketsim.market.mm import MM_ACCOUNT
from marketsim.pricing.bond_buckets import BUCKET_ORDER, DURATION_REF_YIELD
from marketsim.pricing.curve import KAPPA_QE, term_premium_bucket
from marketsim.real.feedbacks import capex_q_term
from marketsim.real.policy.cb_operations import (
    BANK,
    CASH,
    CB,
    GOVT,
    QE_GDP_SHARE,
    QE_MONTHS,
    RES,
    CBOperations,
    opening_mm_book,
    purchase_postings,
)

# Gate 13: 10 % of GDP over 12 months → GB_BOND yield −30 to −80 bp (§6.11).
_YIELD_BP_LO = 30.0
_YIELD_BP_HI = 80.0
# Outstanding face / GDP so monthly slices stay well inside ADV (impact modest).
_FACE_TO_GDP = 50.0
_GDP = 1_000_000.0


def _market(cfg: Config, *, face: float) -> BondSecondaryMarket:
    return BondSecondaryMarket.from_config(
        cfg,
        face={b: face for b in BUCKET_ORDER},
        holdings=opening_mm_book(face),
        fair_yield=DURATION_REF_YIELD,
        sigma=0.01,
    )


def _desk(cfg: Config, *, face: float | None = None, pool: CorpPool | None = None) -> CBOperations:
    led = Ledger.empty(debug_journal=True)
    mkt = _market(cfg, face=face if face is not None else _FACE_TO_GDP * _GDP)
    return CBOperations(led, mkt, gdp=_GDP, pool=pool, s_t=0.01)


def test_purchase_four_postings_by_hand(cfg: Config) -> None:
    ops = _desk(cfg)
    x = 1_000.0
    y0 = ops.market.bucket_yield("GB_BOND")
    p0 = ops.market.price("GB_BOND")
    mm_b0 = ops.market.holdings[MM_ACCOUNT]["GB_BOND"]
    legs = purchase_postings(x, x / p0, "GB_BOND")
    assert len(legs) == 3
    assert {e.entity for e in legs[0]} == {MM_ACCOUNT, CB}
    assert {e.instrument for e in legs[0]} == {"GB_BOND"}
    assert {e.entity for e in legs[1]} == {CB, BANK}
    assert {e.instrument for e in legs[1]} == {RES}
    assert {e.entity for e in legs[2]} == {BANK, MM_ACCOUNT}
    assert {e.instrument for e in legs[2]} == {CASH}

    face = ops.purchase(x, "GB_BOND", tick=1)
    assert face == pytest.approx(x / p0)
    assert_consistent(ops.ledger, 1)
    assert ops.ledger.position(CB, "GB_BOND") == pytest.approx(face)
    assert ops.market.holdings[MM_ACCOUNT]["GB_BOND"] == pytest.approx(mm_b0 - face)
    assert ops.ledger.position(BANK, RES) == pytest.approx(x)
    assert ops.ledger.position(CB, RES) == pytest.approx(-x)
    assert ops.ledger.position(MM_ACCOUNT, CASH) == pytest.approx(x)
    assert ops.ledger.position(BANK, CASH) == pytest.approx(-x)
    assert ops.market.bucket_yield("GB_BOND") < y0


def test_gate13_qe_then_qt_reserves_and_yield(cfg: Config) -> None:
    ops = _desk(cfg)
    y0 = ops.market.bucket_yield("GB_BOND")
    res0 = ops.ledger.position(BANK, RES)
    ops.programme("GB_BOND", gdp_share=QE_GDP_SHARE, months=QE_MONTHS, start_tick=1)
    assert_consistent(ops.ledger)
    injected = QE_GDP_SHARE * _GDP
    assert ops.ledger.position(BANK, RES) == pytest.approx(res0 + injected)
    # Purchases are 10 % of GDP in market value; face/GDP is a touch lower as P rises.
    assert ops.h_cb == pytest.approx(QE_GDP_SHARE, abs=0.01)
    dy_bp = (ops.market.bucket_yield("GB_BOND") - y0) * 10_000.0
    assert dy_bp <= -_YIELD_BP_LO
    assert dy_bp >= -_YIELD_BP_HI

    ops.unwind("GB_BOND", tick=20)
    assert_consistent(ops.ledger)
    assert ops.h_cb == pytest.approx(0.0, abs=1e-6)
    assert ops.ledger.position(BANK, RES) == pytest.approx(res0, abs=0.15 * injected)
    # QT undoes the stock; leftover ξ may keep a small concession.
    assert ops.market.bucket_yield("GB_BOND") > y0 + (dy_bp / 10_000.0) * 0.25


def test_gate14_bond_financed_raises_tp_and_lowers_capex(cfg: Config) -> None:
    """Same Δb under bond finance vs QE: tp↑ / capex↓ vs QE; QE injects reserves (§6.11.14)."""
    assert cfg.edges is not None and cfg.dynamics is not None
    d = 7.07
    b0, db = 0.60, 0.05
    tp_bond = term_premium_bucket(
        d, debt_to_gdp=b0 + db, debt_to_gdp_star=b0, h_cb=0.0, h_cb0=0.0, z_risk=0.0
    )
    tp_qe = term_premium_bucket(
        d, debt_to_gdp=b0 + db, debt_to_gdp_star=b0, h_cb=db, h_cb0=0.0, z_risk=0.0
    )
    assert tp_bond > tp_qe
    assert tp_bond - tp_qe == pytest.approx((d / 7.0) * KAPPA_QE * db, abs=1e-12)

    q_kwargs = dict(
        q_scale=cfg.edges.capex.q_scale,
        q_tobin_coef=cfg.edges.capex.coefficients.q_tobin,
        q_clip=cfg.dynamics.capex.q_clip,
    )
    cap_bond = float(capex_q_term(math.exp(-d * tp_bond), **q_kwargs))
    cap_qe = float(capex_q_term(math.exp(-d * tp_qe), **q_kwargs))
    assert cap_bond < cap_qe

    qe = _desk(cfg)
    qe.debt_to_gdp = b0 + db
    qe.purchase(db * _GDP, "GB_BOND", tick=1)
    bond = _desk(cfg)
    bond.debt_to_gdp = b0 + db
    bond.refresh_fair_yields()
    assert qe.ledger.position(BANK, RES) > bond.ledger.position(BANK, RES)
    assert qe.market.bucket_yield("GB_BOND") < bond.market.bucket_yield("GB_BOND")


def test_coupon_remittance_to_govt(cfg: Config) -> None:
    ops = _desk(cfg)
    ops.purchase(21_000.0, "GB_BOND", tick=1)
    cb_dep0 = ops.ledger.position(CB, CASH)
    govt0 = ops.ledger.position(GOVT, CASH)
    remitted = ops.remit_coupons(tick=2)
    assert remitted > 0.0
    assert ops.ledger.position(CB, CASH) == pytest.approx(cb_dep0 - remitted)
    assert ops.ledger.position(GOVT, CASH) == pytest.approx(govt0 + remitted)
    assert_consistent(ops.ledger, 2)


def test_corp_pool_purchase_lowers_common_spread(cfg: Config) -> None:
    pool = CorpPool()
    s0 = pool.borrowing_spread(0.01, 0.0)
    ops = _desk(cfg, pool=pool)
    ops.purchase(5_000.0, "CORP_POOL", tick=1)
    assert_consistent(ops.ledger, 1)
    assert pool.borrowing_spread(0.01, 0.0) < s0
    assert ops.market.price("CORP_POOL") > 1.0
