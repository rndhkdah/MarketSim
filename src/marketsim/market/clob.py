"""Price-time CLOB for ``EQ:FIRM:*`` (T6.10 / §6.6).

Pure matching engine: fills are structured records for T6.12 settlement. No ledger
postings, no thin engine quote (T6.11), no background flow.

Units
-----
- ``tick``: price increment (currency / share).
- ``lot``: minimum size and size multiple (shares).
- ``price`` / ``stop_price`` / ``last_price`` / ``reference_price``: currency / share.
- ``qty`` / ``remaining`` / auction ``volume``: shares.
- ``notional``: currency (``price * qty``), fee-ready.
- ``halt_band``: dimensionless fraction of the daily reference (``markets.yaml``).
- ``clock_tick``: matching-session index. One ``end_tick()`` is one trading day.

Tick boundary / DAY expiries
----------------------------
DAY orders are queued as ``ExpiryItem(tick, order_id)`` on accept. They die when
``end_tick()`` runs on the tick they were accepted (after a pending call auction).
GTC orders survive. This is the documented day boundary.

Determinism
-----------
``ingest`` processes a same-tick batch in ``(agent_id, sequence, order_id)`` order.
Triggered stops flush in that order. Books on a venue are visited by sorted symbol.
No wall-clock, no unordered dict iteration, no RNG.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from sortedcontainers import SortedDict

from marketsim.core.errors import ConfigError, StateError
from marketsim.market.instruments import ClobCfg


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class TimeInForce(StrEnum):
    IOC = "ioc"
    GTC = "gtc"
    DAY = "day"


class BookStatus(StrEnum):
    CONTINUOUS = "continuous"
    AUCTION = "auction"


class OrderStatus(StrEnum):
    RESTING = "resting"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


def price_to_ticks(price: float, tick: float) -> int:
    """Quantise ``price`` (currency / share) onto the tick grid. Raises ``ValueError`` if off-grid."""
    if tick <= 0:
        raise ValueError("tick must be > 0 (price increment)")
    n = int(round(float(price) / float(tick)))
    if abs(n * tick - float(price)) > min(1e-9, float(tick) * 1e-6):
        raise ValueError(f"price {price} is not a multiple of tick {tick}")
    return n


def ticks_to_price(price_ticks: int, tick: float) -> float:
    """Tick count → currency / share."""
    return int(price_ticks) * float(tick)


@dataclass(frozen=True, slots=True)
class Order:
    """Instruction. ``qty`` is shares; ``price`` / ``stop_price`` are currency / share."""

    agent_id: str
    side: Side
    qty: int
    symbol: str = ""
    order_type: OrderType = OrderType.LIMIT
    tif: TimeInForce = TimeInForce.GTC
    price: float | None = None
    stop_price: float | None = None
    sequence: int | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """One match. ``price`` currency / share; ``qty`` shares; ``notional`` currency (``price * qty``)."""

    symbol: str
    price: float
    qty: int
    maker: str
    taker: str
    maker_order_id: int
    taker_order_id: int
    notional: float
    tick: int


@dataclass(frozen=True, slots=True)
class SubmitResult:
    """Outcome of submit / cancel / replace. ``remaining`` is unfilled shares (0 if dead)."""

    order_id: int | None
    status: OrderStatus
    remaining: int
    fills: tuple[Fill, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class AuctionResult:
    """Call-auction uncross. ``price`` is currency / share; ``volume`` / ``imbalance`` are shares."""

    symbol: str
    price: float | None
    volume: int
    imbalance: int
    fills: tuple[Fill, ...] = ()


@dataclass(frozen=True, slots=True)
class ExpiryItem:
    """DAY-order expiry queue item. Dies at ``end_tick`` of ``tick`` (matching-session index)."""

    tick: int
    order_id: int


@dataclass(slots=True)
class LiveOrder:
    order_id: int
    agent_id: str
    sequence: int
    side: Side
    qty: int
    remaining: int
    order_type: OrderType
    tif: TimeInForce
    price_ticks: int | None
    stop_ticks: int | None
    priority: int
    status: OrderStatus
    symbol: str
    triggered: bool = False

    def to_state(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "agent_id": self.agent_id,
            "sequence": self.sequence,
            "side": str(self.side),
            "qty": self.qty,
            "remaining": self.remaining,
            "order_type": str(self.order_type),
            "tif": str(self.tif),
            "price_ticks": self.price_ticks,
            "stop_ticks": self.stop_ticks,
            "priority": self.priority,
            "status": str(self.status),
            "symbol": self.symbol,
            "triggered": self.triggered,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> LiveOrder:
        return cls(
            order_id=int(state["order_id"]),
            agent_id=str(state["agent_id"]),
            sequence=int(state["sequence"]),
            side=Side(state["side"]),
            qty=int(state["qty"]),
            remaining=int(state["remaining"]),
            order_type=OrderType(state["order_type"]),
            tif=TimeInForce(state["tif"]),
            price_ticks=None if state["price_ticks"] is None else int(state["price_ticks"]),
            stop_ticks=None if state["stop_ticks"] is None else int(state["stop_ticks"]),
            priority=int(state["priority"]),
            status=OrderStatus(state["status"]),
            symbol=str(state["symbol"]),
            triggered=bool(state["triggered"]),
        )


def _level_qty(level: deque[LiveOrder]) -> int:
    return sum(o.remaining for o in level)


def _reject(reason: str, *, order_id: int | None = None) -> SubmitResult:
    return SubmitResult(order_id=order_id, status=OrderStatus.REJECTED, remaining=0, reason=reason)


class OrderBook:
    """Single-instrument price-time book.

    Bids / asks are ``SortedDict`` price-tick → FIFO ``deque``. Stops sit off-book
    until ``last_price`` crosses the stop (buy: last ≥ stop; sell: last ≤ stop).
    """

    def __init__(
        self,
        symbol: str,
        *,
        tick: float,
        lot: int,
        halt_band: float,
        reference_price: float | None = None,
        clock_tick: int = 0,
    ) -> None:
        if tick <= 0:
            raise ConfigError("tick must be > 0 (price increment)")
        if lot < 1:
            raise ConfigError("lot must be >= 1 (shares)")
        if not 0.0 <= float(halt_band) <= 1.0:
            raise ConfigError("halt_band must be in [0, 1]")
        if not symbol:
            raise ConfigError("symbol must be non-empty")
        self.symbol = symbol
        self.tick = float(tick)
        self.lot = int(lot)
        self.halt_band = float(halt_band)
        self.clock_tick = int(clock_tick)
        self.status = BookStatus.AUCTION if reference_price is None else BookStatus.CONTINUOUS
        self._halt_pending = False
        self._bids: SortedDict[int, deque[LiveOrder]] = SortedDict()
        self._asks: SortedDict[int, deque[LiveOrder]] = SortedDict()
        self._buy_stops: SortedDict[int, deque[LiveOrder]] = SortedDict()
        self._sell_stops: SortedDict[int, deque[LiveOrder]] = SortedDict()
        self._mkt_buys: deque[LiveOrder] = deque()
        self._mkt_sells: deque[LiveOrder] = deque()
        self._orders: dict[int, LiveOrder] = {}
        self._expiry: list[ExpiryItem] = []
        self._triggered: list[LiveOrder] = []
        self._next_order_id = 1
        self._next_priority = 1
        self._next_sequence = 1
        self._reference_ticks: int | None = None
        self._last_ticks: int | None = None
        if reference_price is not None:
            self._set_reference(float(reference_price))

    @classmethod
    def from_clob_cfg(
        cls,
        cfg: ClobCfg,
        symbol: str,
        *,
        reference_price: float | None = None,
    ) -> OrderBook:
        """Build from ``markets.yaml`` ``clob:`` (tick, lot, halt_band). Thin quote is T6.11."""
        return cls(
            symbol,
            tick=cfg.tick,
            lot=cfg.lot,
            halt_band=cfg.halt_band,
            reference_price=reference_price,
        )

    @property
    def reference_price(self) -> float | None:
        """Daily band centre (currency / share)."""
        if self._reference_ticks is None:
            return None
        return ticks_to_price(self._reference_ticks, self.tick)

    @property
    def last_price(self) -> float | None:
        """Last print (currency / share)."""
        if self._last_ticks is None:
            return None
        return ticks_to_price(self._last_ticks, self.tick)

    @property
    def best_bid(self) -> float | None:
        """Highest resting limit bid (currency / share), or None."""
        if not self._bids:
            return None
        return ticks_to_price(self._bids.peekitem(-1)[0], self.tick)

    @property
    def best_ask(self) -> float | None:
        """Lowest resting limit ask (currency / share), or None."""
        if not self._asks:
            return None
        return ticks_to_price(self._asks.peekitem(0)[0], self.tick)

    @property
    def is_halted(self) -> bool:
        """True after a band breach while the book is still in call-auction mode."""
        return self.status is BookStatus.AUCTION and self._halt_pending

    def is_crossed(self) -> bool:
        """True if limit best bid ≥ best ask. Auction books may be crossed; continuous must not."""
        if not self._bids or not self._asks:
            return False
        return int(self._bids.peekitem(-1)[0]) >= int(self._asks.peekitem(0)[0])

    def level_order_ids(self, side: Side, price: float) -> tuple[int, ...]:
        """FIFO order ids at ``price`` (currency / share); front has time priority."""
        px = price_to_ticks(price, self.tick)
        book = self._bids if side is Side.BUY else self._asks
        level = book.get(px)
        if level is None:
            return ()
        return tuple(o.order_id for o in level)

    def resting_ids(self) -> tuple[int, ...]:
        """Live order ids, sorted (deterministic)."""
        return tuple(sorted(self._orders))

    def submit(self, order: Order) -> SubmitResult:
        """Accept one order and match immediately (continuous) or rest for the call (auction)."""
        live, err = self._accept(order)
        if err is not None:
            return err
        return self._process(live)

    def ingest(self, orders: Sequence[Order]) -> list[SubmitResult]:
        """Same-tick batch: accept, then process in ``(agent_id, sequence, order_id)`` order."""
        items: list[tuple[str, int, int, LiveOrder | SubmitResult]] = []
        for raw in orders:
            live, err = self._accept(raw)
            if live is not None:
                items.append((live.agent_id, live.sequence, live.order_id, live))
            else:
                seq = 0 if raw.sequence is None else int(raw.sequence)
                items.append((raw.agent_id, seq, 0, err if err is not None else _reject("invalid")))
        items.sort(key=lambda row: (row[0], row[1], row[2]))
        out: list[SubmitResult] = []
        for _, _, _, payload in items:
            if isinstance(payload, LiveOrder):
                out.append(self._process(payload))
            else:
                out.append(payload)
        return out

    def cancel(self, order_id: int) -> SubmitResult:
        """Cancel a live order. Unknown ids are rejected."""
        order = self._orders.get(int(order_id))
        if order is None:
            return _reject("unknown order", order_id=int(order_id))
        self._pull(order)
        self._retire(order, OrderStatus.CANCELLED)
        return SubmitResult(
            order_id=order.order_id,
            status=OrderStatus.CANCELLED,
            remaining=0,
            reason="cancel",
        )

    def replace(self, order_id: int, *, qty: int | None = None, price: float | None = None) -> SubmitResult:
        """Cancel-replace. ``qty`` is the new remaining size (shares).

        Priority is kept only when the limit/stop price is unchanged **and** size
        does not increase. Size-up or any price change loses time priority.
        A price change re-matches as an aggressor.
        """
        order = self._orders.get(int(order_id))
        if order is None:
            return _reject("unknown order", order_id=int(order_id))
        if qty is not None:
            try:
                self._check_qty(int(qty), allow_zero=True)
            except ValueError as exc:
                return _reject(str(exc), order_id=order.order_id)
            new_qty = int(qty)
        else:
            new_qty = order.remaining
        if new_qty == 0:
            return self.cancel(order.order_id)
        new_price_ticks = order.price_ticks
        if price is not None:
            try:
                new_price_ticks = self._check_price(float(price))
            except ValueError as exc:
                return _reject(str(exc), order_id=order.order_id)
        price_changed = new_price_ticks != order.price_ticks
        size_up = new_qty > order.remaining
        if not price_changed and not size_up:
            order.remaining = new_qty
            order.qty = max(order.qty, new_qty)
            status = OrderStatus.PARTIAL if order.remaining < order.qty else OrderStatus.RESTING
            order.status = status
            return SubmitResult(order_id=order.order_id, status=status, remaining=order.remaining)
        if not price_changed and size_up:
            self._pull(order)
            order.remaining = new_qty
            order.qty = new_qty
            self._rest(order)
            return SubmitResult(order_id=order.order_id, status=order.status, remaining=order.remaining)
        self._pull(order)
        order.price_ticks = new_price_ticks
        order.remaining = new_qty
        order.qty = new_qty
        if order.order_type is OrderType.STOP and not order.triggered:
            if self._stop_should_fire(order):
                order.triggered = True
                if order.price_ticks is None:
                    order.order_type = OrderType.MARKET
                fills = self._match_live(order)
                fills.extend(self._flush_stops())
                return self._result(order, fills)
            self._rest(order)
            return SubmitResult(order_id=order.order_id, status=order.status, remaining=order.remaining)
        fills = self._match_live(order)
        fills.extend(self._flush_stops())
        return self._result(order, fills)

    def start_auction(self) -> None:
        """Switch to call-auction mode (IPO / halt). Resting limits stay."""
        self.status = BookStatus.AUCTION

    def uncross(self) -> AuctionResult:
        """Volume-maximising call auction. No-op if already continuous.

        Clearing price: maximise executable volume; then minimise |imbalance|
        (buy qty − sell qty at that price, shares); then the tick closest to
        the prior/reference price; then the higher tick.
        After a non-empty uncross, last and the daily band reference become the
        clearing price so continuous trading can resume. Leftover markets and
        IOC qty are cancelled; GTC/DAY limits rest.
        """
        if self.status is BookStatus.CONTINUOUS:
            return AuctionResult(symbol=self.symbol, price=None, volume=0, imbalance=0)
        picked = self._auction_clearing_ticks()
        if picked is None:
            self._cancel_auction_leftovers(markets_only=True)
            if self._reference_ticks is not None:
                self.status = BookStatus.CONTINUOUS
                self._halt_pending = False
            return AuctionResult(symbol=self.symbol, price=None, volume=0, imbalance=0)
        price_ticks, volume, imbalance = picked
        fills = self._auction_match(price_ticks, volume)
        self._last_ticks = price_ticks
        self._reference_ticks = price_ticks
        self._cancel_auction_leftovers(markets_only=False)
        self.status = BookStatus.CONTINUOUS
        self._halt_pending = False
        fills.extend(self._flush_stops())
        return AuctionResult(
            symbol=self.symbol,
            price=ticks_to_price(price_ticks, self.tick),
            volume=volume,
            imbalance=imbalance,
            fills=tuple(fills),
        )

    def end_tick(self) -> tuple[AuctionResult | None, tuple[int, ...]]:
        """Day boundary: uncross a pending auction, expire DAY orders, advance ``clock_tick``.

        DAY orders accepted on this ``clock_tick`` are cancelled (expiry queue drain
        in ``order_id`` order). Returns ``(auction_or_None, expired_order_ids)``.
        """
        auction: AuctionResult | None = None
        if self.status is BookStatus.AUCTION:
            auction = self.uncross()
        expired = self._expire_day_orders()
        self.clock_tick += 1
        return auction, expired

    def set_reference(self, price: float) -> None:
        """Set the daily band centre (currency / share) without changing book status."""
        self._set_reference(float(price))

    def to_state(self) -> dict[str, Any]:
        orders = [self._orders[i].to_state() for i in sorted(self._orders)]
        expiry = sorted(self._expiry, key=lambda e: (e.tick, e.order_id))
        triggered = sorted(o.order_id for o in self._triggered)
        return {
            "symbol": self.symbol,
            "tick": self.tick,
            "lot": self.lot,
            "halt_band": self.halt_band,
            "clock_tick": self.clock_tick,
            "status": str(self.status),
            "halt_pending": self._halt_pending,
            "next_order_id": self._next_order_id,
            "next_priority": self._next_priority,
            "next_sequence": self._next_sequence,
            "reference_ticks": self._reference_ticks,
            "last_ticks": self._last_ticks,
            "orders": orders,
            "expiry": [{"tick": e.tick, "order_id": e.order_id} for e in expiry],
            "triggered_ids": triggered,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> OrderBook:
        if "symbol" not in state or "tick" not in state:
            raise StateError("order book state needs symbol and tick")
        book = cls(
            str(state["symbol"]),
            tick=float(state["tick"]),
            lot=int(state["lot"]),
            halt_band=float(state["halt_band"]),
            reference_price=None,
            clock_tick=int(state["clock_tick"]),
        )
        book.status = BookStatus(state["status"])
        book._halt_pending = bool(state["halt_pending"])
        book._next_order_id = int(state["next_order_id"])
        book._next_priority = int(state["next_priority"])
        book._next_sequence = int(state["next_sequence"])
        book._reference_ticks = None if state["reference_ticks"] is None else int(state["reference_ticks"])
        book._last_ticks = None if state["last_ticks"] is None else int(state["last_ticks"])
        lives = [LiveOrder.from_state(row) for row in state["orders"]]
        lives.sort(key=lambda o: (o.priority, o.order_id))
        for live in lives:
            book._orders[live.order_id] = live
            book._place(live)
        book._expiry = [
            ExpiryItem(tick=int(row["tick"]), order_id=int(row["order_id"]))
            for row in state.get("expiry", ())
        ]
        trig_ids = {int(i) for i in state.get("triggered_ids", ())}
        book._triggered = [book._orders[i] for i in sorted(trig_ids) if i in book._orders]
        return book

    def _set_reference(self, price: float) -> None:
        self._reference_ticks = self._check_price(price)

    def _check_qty(self, qty: int, *, allow_zero: bool = False) -> int:
        if qty < 0 or (qty == 0 and not allow_zero):
            raise ValueError("qty must be a positive multiple of lot (shares)")
        if qty % self.lot != 0:
            raise ValueError("qty must be a positive multiple of lot (shares)")
        return qty

    def _check_price(self, price: float) -> int:
        if float(price) <= 0:
            raise ValueError("price must be > 0 (currency / share)")
        return price_to_ticks(float(price), self.tick)

    def _accept(self, order: Order) -> tuple[LiveOrder | None, SubmitResult | None]:
        symbol = order.symbol or self.symbol
        if symbol != self.symbol:
            return None, _reject(f"symbol {order.symbol!r} does not match book {self.symbol!r}")
        try:
            qty = self._check_qty(int(order.qty))
        except ValueError as exc:
            return None, _reject(str(exc))
        price_ticks: int | None = None
        stop_ticks: int | None = None
        if order.order_type is OrderType.LIMIT:
            if order.price is None:
                return None, _reject("limit order requires price (currency / share)")
            try:
                price_ticks = self._check_price(float(order.price))
            except ValueError as exc:
                return None, _reject(str(exc))
        elif order.order_type is OrderType.STOP:
            if order.stop_price is None:
                return None, _reject("stop order requires stop_price (currency / share)")
            try:
                stop_ticks = self._check_price(float(order.stop_price))
                if order.price is not None:
                    price_ticks = self._check_price(float(order.price))
            except ValueError as exc:
                return None, _reject(str(exc))
        elif order.order_type is OrderType.MARKET:
            if order.price is not None:
                try:
                    price_ticks = self._check_price(float(order.price))
                except ValueError as exc:
                    return None, _reject(str(exc))
        else:
            return None, _reject(f"unknown order_type {order.order_type!r}")
        if order.sequence is None:
            sequence = self._next_sequence
            self._next_sequence += 1
        else:
            sequence = int(order.sequence)
            if sequence >= self._next_sequence:
                self._next_sequence = sequence + 1
        oid = self._next_order_id
        self._next_order_id += 1
        live = LiveOrder(
            order_id=oid,
            agent_id=str(order.agent_id),
            sequence=sequence,
            side=Side(order.side),
            qty=qty,
            remaining=qty,
            order_type=OrderType(order.order_type),
            tif=TimeInForce(order.tif),
            price_ticks=price_ticks,
            stop_ticks=stop_ticks,
            priority=-1,
            status=OrderStatus.RESTING,
            symbol=self.symbol,
        )
        self._orders[oid] = live
        if live.tif is TimeInForce.DAY:
            self._expiry.append(ExpiryItem(self.clock_tick, oid))
        return live, None

    def _process(self, live: LiveOrder) -> SubmitResult:
        if live.order_type is OrderType.STOP:
            if self._stop_should_fire(live):
                live.triggered = True
                if live.price_ticks is None:
                    live.order_type = OrderType.MARKET
            else:
                self._rest(live)
                return SubmitResult(
                    order_id=live.order_id,
                    status=OrderStatus.RESTING,
                    remaining=live.remaining,
                )
        fills = self._match_live(live)
        fills.extend(self._flush_stops())
        return self._result(live, fills)

    def _result(self, live: LiveOrder, fills: list[Fill]) -> SubmitResult:
        return SubmitResult(
            order_id=live.order_id,
            status=live.status,
            remaining=live.remaining if live.order_id in self._orders else 0,
            fills=tuple(fills),
            reason=None
            if live.status not in (OrderStatus.CANCELLED, OrderStatus.EXPIRED)
            else live.status.value,
        )

    def _match_live(self, order: LiveOrder) -> list[Fill]:
        if self.status is BookStatus.AUCTION:
            self._rest(order)
            return []
        fills: list[Fill] = []
        opposite = self._asks if order.side is Side.BUY else self._bids
        while order.remaining > 0 and opposite:
            opp_px = int(opposite.peekitem(0 if order.side is Side.BUY else -1)[0])
            if not self._price_crosses(order, opp_px):
                break
            level = opposite[opp_px]
            while order.remaining > 0 and level:
                rest = level[0]
                if rest.agent_id == order.agent_id:
                    order.remaining = 0
                    self._retire(order, OrderStatus.CANCELLED)
                    return fills
                if self._outside_band(opp_px):
                    self._enter_halt()
                    self._park_after_halt(order)
                    return fills
                qty = min(order.remaining, rest.remaining)
                fills.append(self._execute(rest, order, qty, opp_px))
                if rest.remaining == 0:
                    level.popleft()
                    self._retire(rest, OrderStatus.FILLED)
            if not level:
                del opposite[opp_px]
        self._finish_aggressor(order)
        return fills

    def _price_crosses(self, order: LiveOrder, opp_px: int) -> bool:
        if order.order_type is OrderType.MARKET or order.price_ticks is None:
            return True
        if order.side is Side.BUY:
            return opp_px <= order.price_ticks
        return opp_px >= order.price_ticks

    def _finish_aggressor(self, order: LiveOrder) -> None:
        if order.order_id not in self._orders:
            return
        if order.remaining <= 0:
            self._retire(order, OrderStatus.FILLED)
            return
        if order.tif is TimeInForce.IOC or order.order_type is OrderType.MARKET:
            self._retire(order, OrderStatus.CANCELLED)
            return
        self._rest(order)

    def _park_after_halt(self, order: LiveOrder) -> None:
        if order.order_id not in self._orders:
            return
        if order.remaining <= 0:
            self._retire(order, OrderStatus.FILLED)
            return
        if order.tif is TimeInForce.IOC:
            self._retire(order, OrderStatus.CANCELLED)
            return
        self._rest(order)

    def _enter_halt(self) -> None:
        self.status = BookStatus.AUCTION
        self._halt_pending = True

    def _outside_band(self, price_ticks: int) -> bool:
        if self._reference_ticks is None:
            return False
        ref = ticks_to_price(self._reference_ticks, self.tick)
        px = ticks_to_price(int(price_ticks), self.tick)
        lo = ref * (1.0 - self.halt_band)
        hi = ref * (1.0 + self.halt_band)
        return px < lo - 1e-12 or px > hi + 1e-12

    def _execute(self, maker: LiveOrder, taker: LiveOrder, qty: int, price_ticks: int) -> Fill:
        maker.remaining -= qty
        taker.remaining -= qty
        if maker.remaining:
            maker.status = OrderStatus.PARTIAL
        if taker.remaining:
            taker.status = OrderStatus.PARTIAL
        px = ticks_to_price(int(price_ticks), self.tick)
        self._note_last(int(price_ticks))
        return Fill(
            symbol=self.symbol,
            price=px,
            qty=int(qty),
            maker=maker.agent_id,
            taker=taker.agent_id,
            maker_order_id=maker.order_id,
            taker_order_id=taker.order_id,
            notional=px * int(qty),
            tick=self.clock_tick,
        )

    def _note_last(self, price_ticks: int) -> None:
        self._last_ticks = int(price_ticks)
        self._activate_stops(int(price_ticks))

    def _stop_should_fire(self, order: LiveOrder) -> bool:
        if order.stop_ticks is None or self._last_ticks is None:
            return False
        if order.side is Side.BUY:
            return self._last_ticks >= order.stop_ticks
        return self._last_ticks <= order.stop_ticks

    def _activate_stops(self, last: int) -> None:
        while self._buy_stops and int(self._buy_stops.peekitem(0)[0]) <= last:
            _px, level = self._buy_stops.peekitem(0)
            order = level.popleft()
            if not level:
                del self._buy_stops[_px]
            if order.order_id not in self._orders:
                continue
            order.triggered = True
            if order.price_ticks is None:
                order.order_type = OrderType.MARKET
            self._triggered.append(order)
        while self._sell_stops and int(self._sell_stops.peekitem(-1)[0]) >= last:
            _px, level = self._sell_stops.peekitem(-1)
            order = level.popleft()
            if not level:
                del self._sell_stops[_px]
            if order.order_id not in self._orders:
                continue
            order.triggered = True
            if order.price_ticks is None:
                order.order_type = OrderType.MARKET
            self._triggered.append(order)

    def _flush_stops(self) -> list[Fill]:
        fills: list[Fill] = []
        while self._triggered:
            self._triggered.sort(key=lambda o: (o.agent_id, o.sequence, o.order_id))
            order = self._triggered.pop(0)
            if order.order_id not in self._orders:
                continue
            fills.extend(self._match_live(order))
        return fills

    def _rest(self, order: LiveOrder) -> None:
        order.priority = self._next_priority
        self._next_priority += 1
        if order.remaining < order.qty:
            order.status = OrderStatus.PARTIAL
        else:
            order.status = OrderStatus.RESTING
        self._place(order)

    def _place(self, order: LiveOrder) -> None:
        if order.order_type is OrderType.STOP and not order.triggered:
            book = self._buy_stops if order.side is Side.BUY else self._sell_stops
            key = int(order.stop_ticks) if order.stop_ticks is not None else 0
            if key not in book:
                book[key] = deque()
            book[key].append(order)
            return
        if order.order_type is OrderType.MARKET or order.price_ticks is None:
            bucket = self._mkt_buys if order.side is Side.BUY else self._mkt_sells
            bucket.append(order)
            return
        book = self._bids if order.side is Side.BUY else self._asks
        key = int(order.price_ticks)
        if key not in book:
            book[key] = deque()
        book[key].append(order)

    def _pull(self, order: LiveOrder) -> None:
        if order.order_type is OrderType.STOP and not order.triggered:
            self._remove_from(
                self._buy_stops if order.side is Side.BUY else self._sell_stops, order.stop_ticks, order
            )
            return
        if order.order_type is OrderType.MARKET or order.price_ticks is None:
            bucket = self._mkt_buys if order.side is Side.BUY else self._mkt_sells
            self._remove_from_deque(bucket, order)
            return
        book = self._bids if order.side is Side.BUY else self._asks
        self._remove_from(book, order.price_ticks, order)

    def _remove_from(
        self,
        book: SortedDict[int, deque[LiveOrder]],
        key: int | None,
        order: LiveOrder,
    ) -> None:
        if key is None:
            return
        level = book.get(int(key))
        if level is None:
            return
        self._remove_from_deque(level, order)
        if not level:
            del book[int(key)]

    @staticmethod
    def _remove_from_deque(bucket: deque[LiveOrder], order: LiveOrder) -> None:
        try:
            bucket.remove(order)
        except ValueError:
            return

    def _retire(self, order: LiveOrder, status: OrderStatus) -> None:
        order.status = status
        order.remaining = 0 if status is not OrderStatus.PARTIAL else order.remaining
        if status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
            self._orders.pop(order.order_id, None)

    def _expire_day_orders(self) -> tuple[int, ...]:
        keep: list[ExpiryItem] = []
        expired: list[int] = []
        pending = sorted(self._expiry, key=lambda e: (e.tick, e.order_id))
        for item in pending:
            if item.tick > self.clock_tick:
                keep.append(item)
                continue
            order = self._orders.get(item.order_id)
            if order is None or order.tif is not TimeInForce.DAY:
                continue
            self._pull(order)
            self._retire(order, OrderStatus.EXPIRED)
            expired.append(item.order_id)
        self._expiry = keep
        return tuple(expired)

    def _mkt_qty(self, side: Side) -> int:
        bucket = self._mkt_buys if side is Side.BUY else self._mkt_sells
        return sum(o.remaining for o in bucket)

    def _demand_supply(self, price_ticks: int) -> tuple[int, int]:
        demand = self._mkt_qty(Side.BUY)
        supply = self._mkt_qty(Side.SELL)
        for px, level in self._bids.items():
            if int(px) >= price_ticks:
                demand += _level_qty(level)
        for px, level in self._asks.items():
            if int(px) <= price_ticks:
                supply += _level_qty(level)
        return demand, supply

    def _limit_prices(self) -> list[int]:
        prices = {int(p) for p in self._bids} | {int(p) for p in self._asks}
        return sorted(prices)

    def _auction_clearing_ticks(self) -> tuple[int, int, int] | None:
        """Return ``(price_ticks, volume, imbalance)`` or None if nothing trades."""
        prices = self._limit_prices()
        mkt_buy = self._mkt_qty(Side.BUY)
        mkt_sell = self._mkt_qty(Side.SELL)
        if not prices:
            if mkt_buy <= 0 or mkt_sell <= 0:
                return None
            ref = self._reference_ticks if self._reference_ticks is not None else self._last_ticks
            if ref is None:
                return None
            vol = min(mkt_buy, mkt_sell)
            return ref, vol, mkt_buy - mkt_sell

        records: list[tuple[int, int, int, int]] = []

        # Each record is (lo, hi, volume, imbalance) inclusive tick range.
        def _push(lo: int, hi: int, demand: int, supply: int) -> None:
            vol = min(demand, supply)
            if vol <= 0:
                return
            records.append((lo, hi, vol, demand - supply))

        for p in prices:
            d, s = self._demand_supply(p)
            _push(p, p, d, s)
        for i in range(len(prices) - 1):
            a, b = prices[i], prices[i + 1]
            if b - a <= 1:
                continue
            # Interior (a, b): D = D(b) (bids at a drop), S = S(a) (asks at b not yet in).
            d_hi, _s_hi = self._demand_supply(b)
            _d_lo, s_lo = self._demand_supply(a)
            _push(a + 1, b - 1, d_hi, s_lo)

        if not records:
            return None
        max_v = max(r[2] for r in records)
        top = [r for r in records if r[2] == max_v]
        min_abs = min(abs(r[3]) for r in top)
        top = [r for r in top if abs(r[3]) == min_abs]
        ref = self._reference_ticks if self._reference_ticks is not None else self._last_ticks
        picked = self._closest_tick([(r[0], r[1]) for r in top], ref)
        # Recover volume / imbalance from the winning range.
        for lo, hi, vol, imb in top:
            if lo <= picked <= hi:
                return picked, vol, imb
        lo, hi, vol, imb = top[0]
        return picked, vol, imb

    def _closest_tick(self, ranges: list[tuple[int, int]], ref: int | None) -> int:
        best: int | None = None
        best_key: tuple[int, int] | None = None
        for lo, hi in ranges:
            if ref is None:
                cand = hi
                key = (0, -cand)
            elif lo <= ref <= hi:
                cand = ref
                key = (0, -cand)
            elif ref < lo:
                cand = lo
                key = (abs(cand - ref), -cand)
            else:
                cand = hi
                key = (abs(cand - ref), -cand)
            if best_key is None or key < best_key:
                best_key = key
                best = cand
        assert best is not None
        return best

    def _eligible(self, side: Side, price_ticks: int) -> list[LiveOrder]:
        out: list[LiveOrder] = []
        markets = self._mkt_buys if side is Side.BUY else self._mkt_sells
        out.extend(o for o in markets if o.remaining > 0)
        book = self._bids if side is Side.BUY else self._asks
        for px, level in book.items():
            if side is Side.BUY and int(px) < price_ticks:
                continue
            if side is Side.SELL and int(px) > price_ticks:
                continue
            out.extend(o for o in level if o.remaining > 0)
        if side is Side.BUY:
            out.sort(
                key=lambda o: (
                    0 if o.order_type is OrderType.MARKET or o.price_ticks is None else 1,
                    -(o.price_ticks or 0),
                    o.priority,
                    o.order_id,
                )
            )
        else:
            out.sort(
                key=lambda o: (
                    0 if o.order_type is OrderType.MARKET or o.price_ticks is None else 1,
                    o.price_ticks or 0,
                    o.priority,
                    o.order_id,
                )
            )
        return out

    def _auction_match(self, price_ticks: int, volume: int) -> list[Fill]:
        buys = self._eligible(Side.BUY, price_ticks)
        sells = self._eligible(Side.SELL, price_ticks)
        fills: list[Fill] = []
        bi = 0
        si = 0
        left = volume
        while left > 0 and bi < len(buys) and si < len(sells):
            buy = buys[bi]
            sell = sells[si]
            if buy.remaining <= 0:
                bi += 1
                continue
            if sell.remaining <= 0:
                si += 1
                continue
            if buy.agent_id == sell.agent_id:
                newest = buy if (buy.sequence, buy.order_id) >= (sell.sequence, sell.order_id) else sell
                self._pull(newest)
                self._retire(newest, OrderStatus.CANCELLED)
                if newest is buy:
                    bi += 1
                else:
                    si += 1
                continue
            qty = min(left, buy.remaining, sell.remaining)
            if buy.priority <= sell.priority:
                maker, taker = buy, sell
            else:
                maker, taker = sell, buy
            fills.append(self._execute(maker, taker, qty, price_ticks))
            left -= qty
            if buy.remaining <= 0:
                self._pull(buy)
                self._retire(buy, OrderStatus.FILLED)
                bi += 1
            if sell.remaining <= 0:
                self._pull(sell)
                self._retire(sell, OrderStatus.FILLED)
                si += 1
        return fills

    def _cancel_auction_leftovers(self, *, markets_only: bool) -> None:
        doomed: list[LiveOrder] = []
        doomed.extend(self._mkt_buys)
        doomed.extend(self._mkt_sells)
        if not markets_only:
            for oid in sorted(self._orders):
                o = self._orders[oid]
                if o.tif is TimeInForce.IOC:
                    doomed.append(o)
        seen: set[int] = set()
        for order in doomed:
            if order.order_id in seen or order.order_id not in self._orders:
                continue
            seen.add(order.order_id)
            self._pull(order)
            self._retire(order, OrderStatus.CANCELLED)


class CLOB:
    """Multi-instrument CLOB venue. Iteration / uncross / end_tick visit symbols sorted."""

    def __init__(self, *, tick: float, lot: int, halt_band: float) -> None:
        if tick <= 0:
            raise ConfigError("tick must be > 0 (price increment)")
        if lot < 1:
            raise ConfigError("lot must be >= 1 (shares)")
        if not 0.0 <= float(halt_band) <= 1.0:
            raise ConfigError("halt_band must be in [0, 1]")
        self.tick = float(tick)
        self.lot = int(lot)
        self.halt_band = float(halt_band)
        self.clock_tick = 0
        self._books: dict[str, OrderBook] = {}
        self._agent_seq: dict[str, int] = {}

    @classmethod
    def from_clob_cfg(cls, cfg: ClobCfg) -> CLOB:
        """Venue from ``markets.yaml`` ``clob:`` (tick, lot, halt_band). Thin quote is T6.11."""
        return cls(tick=cfg.tick, lot=cfg.lot, halt_band=cfg.halt_band)

    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._books))

    def add_book(self, symbol: str, *, reference_price: float | None = None) -> OrderBook:
        """List ``EQ:FIRM:*``. No reference → opening call auction (IPO)."""
        if symbol in self._books:
            raise ConfigError(f"duplicate book {symbol!r}")
        book = OrderBook(
            symbol,
            tick=self.tick,
            lot=self.lot,
            halt_band=self.halt_band,
            reference_price=reference_price,
            clock_tick=self.clock_tick,
        )
        self._books[symbol] = book
        return book

    def book(self, symbol: str) -> OrderBook:
        try:
            return self._books[symbol]
        except KeyError as exc:
            raise ConfigError(f"unknown book {symbol!r}") from exc

    def submit(self, order: Order) -> SubmitResult:
        if not order.symbol:
            return _reject("order.symbol is required")
        if order.symbol not in self._books:
            return _reject(f"unknown book {order.symbol!r}")
        return self._books[order.symbol].submit(order)

    def ingest(self, orders: Sequence[Order]) -> list[SubmitResult]:
        """Lockstep intake: assign missing sequences, process by ``(agent_id, sequence, symbol)``."""
        ready: list[Order] = []
        for raw in orders:
            if raw.sequence is None:
                n = self._agent_seq.get(raw.agent_id, 0) + 1
                self._agent_seq[raw.agent_id] = n
                raw = replace(raw, sequence=n)
            ready.append(raw)
        ready.sort(key=lambda o: (o.agent_id, int(o.sequence or 0), o.symbol))
        return [self.submit(o) for o in ready]

    def cancel(self, symbol: str, order_id: int) -> SubmitResult:
        return self.book(symbol).cancel(order_id)

    def replace(
        self,
        symbol: str,
        order_id: int,
        *,
        qty: int | None = None,
        price: float | None = None,
    ) -> SubmitResult:
        return self.book(symbol).replace(order_id, qty=qty, price=price)

    def uncross(self, symbol: str | None = None) -> list[AuctionResult]:
        if symbol is not None:
            return [self.book(symbol).uncross()]
        return [self._books[s].uncross() for s in self.symbols()]

    def end_tick(self) -> dict[str, tuple[AuctionResult | None, tuple[int, ...]]]:
        """Advance every book at the day boundary (sorted symbols)."""
        out: dict[str, tuple[AuctionResult | None, tuple[int, ...]]] = {}
        for symbol in self.symbols():
            out[symbol] = self._books[symbol].end_tick()
        self.clock_tick += 1
        return out

    def to_state(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "lot": self.lot,
            "halt_band": self.halt_band,
            "clock_tick": self.clock_tick,
            "agent_seq": {k: self._agent_seq[k] for k in sorted(self._agent_seq)},
            "books": {s: self._books[s].to_state() for s in self.symbols()},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CLOB:
        if "tick" not in state or "books" not in state:
            raise StateError("CLOB state needs tick and books")
        venue = cls(tick=float(state["tick"]), lot=int(state["lot"]), halt_band=float(state["halt_band"]))
        venue.clock_tick = int(state.get("clock_tick", 0))
        venue._agent_seq = {str(k): int(v) for k, v in state.get("agent_seq", {}).items()}
        raw_books = state["books"]
        for symbol in sorted(raw_books):
            venue._books[str(symbol)] = OrderBook.from_state(raw_books[symbol])
        return venue
