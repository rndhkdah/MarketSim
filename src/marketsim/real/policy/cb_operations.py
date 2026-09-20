"""Central-bank QE / QT and open-market operations (T6.28 / §6.11).

A purchase of ``x`` (cr, market value) on the secondary venue:

1. bonds ``MM → CB`` (face = ``x / P``);
2. reserves created ``(CB, RES, −x), (BANKSYS, RES, +x)``;
3. seller paid ``(BANKSYS, DEP, −x), (MM, DEP, +x)``.

Flow hits the §6.4 kernel; stock enters ``h_cb`` in ``tp_b``. Coupon income
on the CB book is remitted to ``GOVT``. ``CORP_POOL`` purchases are credit
easing (same postings) and lower the common borrowing spread via ``ξ``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marketsim.core.errors import ConfigError, StateError
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.market.bond_market import BondSecondaryMarket
from marketsim.market.corp_pool import CorpPool, y_match
from marketsim.market.mm import MM_ACCOUNT
from marketsim.pricing.bond_buckets import (
    BUCKET_ORDER,
    DURATION_REF_YIELD,
    GOVT_BUCKETS,
    bucket_duration_years,
    month_face_flows,
)
from marketsim.pricing.curve import term_premium_bucket

CB = "CB"
BANK = "BANKSYS"
GOVT = "GOVT"
CASH = "DEP"
RES = "RES"

BOND_TAG = "equity_trade"
RESERVE_TAG = "capital_transfer"
PAY_TAG = "capital_transfer"
REMIT_TAG = "cb_remittance"

# §6.11 gate 13: 10 % of GDP over 12 months.
QE_GDP_SHARE = 0.10
QE_MONTHS = 12
_EPS = 1e-15


def purchase_postings(amount: float, face: float, bucket: str) -> tuple[tuple[Entry, ...], ...]:
    """The four §6.11 purchase legs as three balanced entry-tuples.

    ``amount`` is cr (market value); ``face`` is remaining-face units.
    Order: bonds MM→CB; reserves created; seller paid.
    """
    x = float(amount)
    q = float(face)
    return (
        (Entry(MM_ACCOUNT, bucket, -q), Entry(CB, bucket, q)),
        (Entry(CB, RES, -x), Entry(BANK, RES, x)),
        (Entry(BANK, CASH, -x), Entry(MM_ACCOUNT, CASH, x)),
    )


class CBOperations:
    """QE / QT desk. ``amount`` arguments are cr; ``h_cb`` is face / GDP."""

    def __init__(
        self,
        ledger: Ledger,
        market: BondSecondaryMarket,
        *,
        gdp: float,
        debt_to_gdp: float = 0.60,
        debt_to_gdp_star: float = 0.60,
        z_risk: float = 0.0,
        y_ref: float = DURATION_REF_YIELD,
        pool: CorpPool | None = None,
        s_t: float = 0.0,
    ) -> None:
        if gdp <= 0.0:
            raise ConfigError("gdp must be > 0 (cr / year)")
        self.ledger = ledger
        self.market = market
        self.gdp = float(gdp)
        self.debt_to_gdp = float(debt_to_gdp)
        self.debt_to_gdp_star = float(debt_to_gdp_star)
        self.z_risk = float(z_risk)
        self.y_ref = float(y_ref)
        self.pool = pool
        self.s_t = float(s_t)
        self._ensure_entities()
        self.h_cb0 = self.h_cb
        self.refresh_fair_yields()

    def _ensure_entities(self) -> None:
        for name in (CB, BANK, GOVT, MM_ACCOUNT):
            if name not in self.ledger.entities:
                self.ledger.register_entity(name)
        for inst in (*BUCKET_ORDER, CASH, RES):
            if inst not in self.ledger.instruments:
                self.ledger.register_instrument(inst, financial=True)

    @property
    def h_cb(self) -> float:
        """Central-bank government-bond face / GDP (dimensionless)."""
        book = self.market.holdings.get(CB, {b: 0.0 for b in BUCKET_ORDER})
        face = sum(float(book.get(b, 0.0)) for b in GOVT_BUCKETS)
        return face / self.gdp

    def duration_y(self, symbol: str) -> float:
        """Bucket Macaulay duration (years) at the current fair yield."""
        return bucket_duration_years(self.market.fair_yield[symbol], self.market.decay[symbol])

    def refresh_fair_yields(self) -> None:
        """``y_b = y_ref + tp_b(h_cb)`` on government buckets; then publish marks."""
        for symbol in GOVT_BUCKETS:
            tp = term_premium_bucket(
                self.duration_y(symbol),
                debt_to_gdp=self.debt_to_gdp,
                debt_to_gdp_star=self.debt_to_gdp_star,
                h_cb=self.h_cb,
                h_cb0=self.h_cb0,
                z_risk=self.z_risk,
            )
            self.market.set_fair_yield(symbol, max(0.0, self.y_ref + tp))
        self.market.publish_marks()

    def purchase(self, amount: float, bucket: str, *, tick: int = 0) -> float:
        """Buy ``amount`` cr of ``bucket`` from ``MM``. Returns face units taken."""
        return self._trade(+float(amount), bucket, tick=tick)

    def sale(self, amount: float, bucket: str, *, tick: int = 0) -> float:
        """Sell ``amount`` cr of ``bucket`` to ``MM`` (QT). Returns face units sold."""
        return self._trade(-float(amount), bucket, tick=tick)

    def _trade(self, signed_amount: float, bucket: str, *, tick: int) -> float:
        if bucket not in BUCKET_ORDER:
            raise ConfigError(f"unknown bucket {bucket!r}")
        if abs(signed_amount) < _EPS:
            return 0.0
        buying = signed_amount > 0.0
        x = abs(float(signed_amount))
        px = float(self.market.price(bucket))
        if px <= 0.0:
            raise ConfigError("secondary price must be > 0 (cr/unit)")
        face = x / px
        src, dst = (MM_ACCOUNT, CB) if buying else (CB, MM_ACCOUNT)
        held = float(self.market.holdings.get(src, {}).get(bucket, 0.0))
        if held + _EPS < face:
            raise ConfigError(f"{src} holds {held} < {face} {bucket} for official flow")
        signed_q = x if buying else -x
        self.market.official_flow(src, dst, bucket, face, signed_notional=signed_q)
        self._post_purchase(x, face, bucket, tick=tick, reverse=not buying)
        if bucket in GOVT_BUCKETS:
            self.refresh_fair_yields()
        else:
            self.market.publish_marks()
        if bucket == "CORP_POOL" and self.pool is not None:
            self._mark_pool()
        return face

    def _post_purchase(
        self,
        amount: float,
        face: float,
        bucket: str,
        *,
        tick: int,
        reverse: bool,
    ) -> None:
        sign = -1.0 if reverse else 1.0
        legs = purchase_postings(sign * amount, sign * face, bucket)
        self.ledger.post(Tx(int(tick), BOND_TAG, legs[0], memo=f"omo {bucket}"))
        self.ledger.post(Tx(int(tick), RESERVE_TAG, legs[1], memo="omo reserves"))
        self.ledger.post(Tx(int(tick), PAY_TAG, legs[2], memo="omo pay seller"))

    def _mark_pool(self) -> None:
        assert self.pool is not None
        ym = y_match(self.market.bucket_yield("GB_NOTE"), self.market.bucket_yield("GB_BOND"))
        self.pool.mark_day(
            xi=float(self.market.kernels["CORP_POOL"].xi),
            nav=1.0,
            y_match=ym,
            s_t=self.s_t,
            z_risk=self.z_risk,
        )

    def remit_coupons(self, *, tick: int) -> float:
        """Remit this month's coupon on CB holdings to ``GOVT``. Returns cr remitted."""
        book = self.market.holdings.get(CB, {})
        total = 0.0
        for symbol in BUCKET_ORDER:
            face = float(book.get(symbol, 0.0))
            if face <= _EPS:
                continue
            coupon, _, _ = month_face_flows(face, self.market.kappa[symbol], self.market.decay[symbol])
            total += coupon
        if abs(total) < _EPS:
            return 0.0
        self.ledger.post(
            Tx(
                int(tick),
                REMIT_TAG,
                (Entry(CB, CASH, -total), Entry(GOVT, CASH, total)),
                memo="cb coupon remittance",
            )
        )
        return float(total)

    def unwind(self, bucket: str, *, tick: int = 0) -> float:
        """Sell the entire CB book in ``bucket`` (QT the residual stock). Returns face sold."""
        face = float(self.market.holdings.get(CB, {}).get(bucket, 0.0))
        if face <= _EPS:
            return 0.0
        return self.sale(face * float(self.market.price(bucket)), bucket, tick=tick)

    def programme(
        self,
        bucket: str,
        *,
        gdp_share: float = QE_GDP_SHARE,
        months: int = QE_MONTHS,
        start_tick: int = 0,
        reverse: bool = False,
    ) -> list[float]:
        """Split ``gdp_share · GDP`` across ``months`` equal purchases (or sales if ``reverse``)."""
        slice_amt = float(gdp_share) * self.gdp / float(months)
        faces: list[float] = []
        for i in range(int(months)):
            if reverse:
                faces.append(self.sale(slice_amt, bucket, tick=start_tick + i))
            else:
                faces.append(self.purchase(slice_amt, bucket, tick=start_tick + i))
        return faces

    def to_state(self) -> dict[str, Any]:
        return {
            "gdp": self.gdp,
            "debt_to_gdp": self.debt_to_gdp,
            "debt_to_gdp_star": self.debt_to_gdp_star,
            "z_risk": self.z_risk,
            "y_ref": self.y_ref,
            "s_t": self.s_t,
            "h_cb0": self.h_cb0,
        }

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        *,
        ledger: Ledger,
        market: BondSecondaryMarket,
        pool: CorpPool | None = None,
    ) -> CBOperations:
        needed = ("gdp", "h_cb0")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"cb operations state missing {missing}")
        obj = cls(
            ledger,
            market,
            gdp=float(state["gdp"]),
            debt_to_gdp=float(state.get("debt_to_gdp", 0.60)),
            debt_to_gdp_star=float(state.get("debt_to_gdp_star", 0.60)),
            z_risk=float(state.get("z_risk", 0.0)),
            y_ref=float(state.get("y_ref", DURATION_REF_YIELD)),
            pool=pool,
            s_t=float(state.get("s_t", 0.0)),
        )
        obj.h_cb0 = float(state["h_cb0"])
        obj.refresh_fair_yields()
        return obj


def opening_mm_book(face: float) -> dict[str, Mapping[str, float]]:
    """All outstanding face sits with ``MM`` (tests / isolated desk)."""
    return {MM_ACCOUNT: {b: float(face) for b in BUCKET_ORDER}}
