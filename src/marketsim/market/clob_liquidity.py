"""Thin engine quote + queue-reactive-lite flow on the CLOB (T6.11 / §6.5–§6.6).

The engine posts only a two-sided quote at ``V·(1 ± w)``, size
``thin_quote_size × float``. Queue-reactive-lite adds extra MM limits whose
intensities depend on the spread and top-of-book imbalance, sized to float so
the book is never empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marketsim.core.rng import RngHub
from marketsim.market.clob import (
    CLOB,
    Order,
    OrderType,
    Side,
    TimeInForce,
    ticks_to_price,
)
from marketsim.market.instruments import ClobCfg
from marketsim.market.mm import MM_ACCOUNT
from marketsim.pricing.firm_value import thin_quote_feed

# §6.5 QR-lite: base limit intensity (1/tick) when the book is balanced.
LAM_LIMIT_0 = 0.35
# §6.5 QR-lite: extra cancel intensity (1/tick) when the book is imbalanced.
LAM_CANCEL_0 = 0.15
# §6.5 QR-lite: market intensity (1/tick) when the opposite side is thick.
LAM_MKT_0 = 0.05
# Isolated from valuation / mispricing streams.
STREAM_CLOB_FLOW = "flow.clob"
# Master plan §6: 252 ticks / year.
DAYS_PER_YEAR = 252


def snap_tick(price: float, tick: float, *, side: Side) -> float:
    """Quantise onto the tick grid. Bids floor, asks ceil. ``price`` is cr/share."""
    step = float(tick)
    n = float(price) / step
    if side is Side.BUY:
        k = int(np.floor(n + 1e-12))
    else:
        k = int(np.ceil(n - 1e-12))
    return ticks_to_price(max(k, 1), step)


def tob_imbalance(bid_qty: int, ask_qty: int) -> float:
    """``(bid − ask) / (bid + ask)``. Dimensionless; 0 if both sides empty."""
    tot = int(bid_qty) + int(ask_qty)
    if tot <= 0:
        return 0.0
    return (int(bid_qty) - int(ask_qty)) / float(tot)


@dataclass
class ClobLiquidity:
    """Engine quote + QR-lite for one ``EQ:FIRM:*`` name.

    ``float_shares`` is the tradable share count. ``value`` is cr/share.
    """

    clob: CLOB
    symbol: str
    float_shares: int
    cfg: ClobCfg
    bid_id: int | None = None
    ask_id: int | None = None
    extra_ids: tuple[int, ...] = ()
    last_value: float = 0.0
    quote_qty: int = 0

    @classmethod
    def attach(
        cls,
        clob: CLOB,
        symbol: str,
        *,
        float_shares: int,
        cfg: ClobCfg,
    ) -> ClobLiquidity:
        if int(float_shares) < 1:
            raise ValueError("float_shares must be >= 1")
        qty = max(int(cfg.lot), int(round(float(cfg.thin_quote_size) * float(float_shares))))
        qty = max(qty, int(cfg.lot))
        qty = (qty // int(cfg.lot)) * int(cfg.lot)
        return cls(clob=clob, symbol=symbol, float_shares=int(float_shares), cfg=cfg, quote_qty=qty)

    def depth_shares(self) -> int:
        """Resting engine size on both sides (shares)."""
        return 2 * int(self.quote_qty)

    def refresh_quote(self, value: float) -> None:
        """Replace the thin quote around ``value`` (cr/share)."""
        book = self.clob.book(self.symbol)
        if self.bid_id is not None:
            book.cancel(self.bid_id)
            self.bid_id = None
        if self.ask_id is not None:
            book.cancel(self.ask_id)
            self.ask_id = None
        raw_bid, raw_ask = thin_quote_feed(value, w=self.cfg.thin_quote_w)
        bid = snap_tick(raw_bid, self.cfg.tick, side=Side.BUY)
        ask = snap_tick(raw_ask, self.cfg.tick, side=Side.SELL)
        if bid >= ask:
            ask = ticks_to_price(int(round(bid / self.cfg.tick)) + 1, self.cfg.tick)
        qty = self.quote_qty
        bid_res = self.clob.submit(
            Order(
                agent_id=MM_ACCOUNT,
                side=Side.BUY,
                qty=qty,
                symbol=self.symbol,
                order_type=OrderType.LIMIT,
                tif=TimeInForce.GTC,
                price=bid,
            )
        )
        ask_res = self.clob.submit(
            Order(
                agent_id=MM_ACCOUNT,
                side=Side.SELL,
                qty=qty,
                symbol=self.symbol,
                order_type=OrderType.LIMIT,
                tif=TimeInForce.GTC,
                price=ask,
            )
        )
        self.bid_id = bid_res.order_id
        self.ask_id = ask_res.order_id
        self.last_value = float(value)

    def quote_inside_band(self, value: float) -> bool:
        """True if best bid/ask sit inside ``V·(1 ± w)`` after tick snap."""
        book = self.clob.book(self.symbol)
        if book.best_bid is None or book.best_ask is None:
            return False
        lo, hi = thin_quote_feed(value, w=self.cfg.thin_quote_w)
        # Tick snap can move one increment outside the raw band.
        pad = float(self.cfg.tick)
        return book.best_bid + pad >= lo and book.best_ask - pad <= hi

    def step_qr(self, rng: RngHub, value: float) -> None:
        """One tick of queue-reactive-lite intensity (limit / cancel / market)."""
        book = self.clob.book(self.symbol)
        bid_q = self.quote_qty
        ask_q = self.quote_qty
        imb = tob_imbalance(bid_q, ask_q)
        bb, ba = book.best_bid, book.best_ask
        if bb is None or ba is None or bb <= 0.0:
            self.refresh_quote(value)
            return
        spread = (ba - bb) / ((ba + bb) / 2.0)
        u = rng.stream(STREAM_CLOB_FLOW).random(3)
        # Wider spread → more limits; imbalance cancels the thick side.
        p_limit = min(1.0, LAM_LIMIT_0 * (1.0 + spread))
        p_cancel = min(1.0, LAM_CANCEL_0 * abs(imb))
        p_mkt = min(1.0, LAM_MKT_0 * max(spread, 0.0))
        extra_qty = max(self.cfg.lot, self.quote_qty // 4)
        extra_qty = (extra_qty // self.cfg.lot) * self.cfg.lot
        if u[0] < p_limit:
            side = Side.BUY if imb <= 0.0 else Side.SELL
            px = bb if side is Side.BUY else ba
            res = self.clob.submit(
                Order(
                    agent_id=MM_ACCOUNT,
                    side=side,
                    qty=extra_qty,
                    symbol=self.symbol,
                    order_type=OrderType.LIMIT,
                    tif=TimeInForce.DAY,
                    price=px,
                )
            )
            if res.order_id is not None:
                self.extra_ids = (*self.extra_ids, res.order_id)
        if u[1] < p_cancel and self.extra_ids:
            oid = self.extra_ids[0]
            book.cancel(oid)
            self.extra_ids = self.extra_ids[1:]
        if u[2] < p_mkt:
            side = Side.SELL if imb > 0.0 else Side.BUY
            self.clob.submit(
                Order(
                    agent_id=MM_ACCOUNT,
                    side=side,
                    qty=self.cfg.lot,
                    symbol=self.symbol,
                    order_type=OrderType.MARKET,
                    tif=TimeInForce.IOC,
                )
            )
        # Re-post the thin quote if a market ate it.
        if book.best_bid is None or book.best_ask is None:
            self.refresh_quote(value)

    def to_state(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "float_shares": self.float_shares,
            "bid_id": self.bid_id,
            "ask_id": self.ask_id,
            "extra_ids": list(self.extra_ids),
            "last_value": self.last_value,
            "quote_qty": self.quote_qty,
        }
