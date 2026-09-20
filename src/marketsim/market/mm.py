"""Engine market maker for NPC equity, bonds and commodities (T6.08 / §6.5).

Quotes ``P·(1 ∓ s/2)`` around an inventory-skewed mid. Orders accepted on
submit fill on the next ``end_tick`` (next-tick fills). Per-tick size is
capped at ``participation_cap × ADV``; leftover quantity rests (GTC/DAY) or
cancels (IOC). The counterparty on every fill is ``MM``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from marketsim.core.errors import ConfigError, StateError
from marketsim.market.impact import ImpactKernel, fill_price, post_impact_mid
from marketsim.market.instruments import ImpactCfg, MMCfg
from marketsim.market.venue import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    SubmitResult,
    TimeInForce,
)

# Ledger entity for the NPC-investor book — a household sub-account (§6.5).
MM_ACCOUNT = "MM"
# Synthetic maker id on every engine quote (CLOB ``Fill.maker_order_id`` is int).
MM_QUOTE_ID = 0


@dataclass(frozen=True, slots=True)
class Quote:
    """Two-sided engine quote.

    ``mid`` / ``skewed_mid`` / ``bid`` / ``ask`` are cr/unit. ``spread`` is the
    dimensionless quoted ``s``; ``skew`` is the dimensionless mid shift.
    """

    symbol: str
    mid: float
    skewed_mid: float
    spread: float
    skew: float
    bid: float
    ask: float


@dataclass(slots=True)
class _Live:
    order_id: int
    agent_id: str
    sequence: int
    side: Side
    qty: int
    remaining: int
    order_type: OrderType
    tif: TimeInForce
    price: float | None
    stop_price: float | None
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
            "price": self.price,
            "stop_price": self.stop_price,
            "status": str(self.status),
            "symbol": self.symbol,
            "triggered": self.triggered,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> _Live:
        return cls(
            order_id=int(state["order_id"]),
            agent_id=str(state["agent_id"]),
            sequence=int(state["sequence"]),
            side=Side(state["side"]),
            qty=int(state["qty"]),
            remaining=int(state["remaining"]),
            order_type=OrderType(state["order_type"]),
            tif=TimeInForce(state["tif"]),
            price=None if state["price"] is None else float(state["price"]),
            stop_price=None if state["stop_price"] is None else float(state["stop_price"]),
            status=OrderStatus(state["status"]),
            symbol=str(state["symbol"]),
            triggered=bool(state["triggered"]),
        )


@dataclass(slots=True)
class _Book:
    symbol: str
    mid: float
    sigma: float
    cap: float
    adv: float
    inventory: float
    kernel: ImpactKernel
    last_price: float | None = None


def quoted_spread(sigma: float, inventory: float, cap: float, cfg: MMCfg) -> float:
    """Dimensionless quoted spread ``s = s0 + k_σ·σ + k_inv·|inv|/cap`` (§6.5).

    ``sigma`` is daily vol (decimal). ``inventory`` and ``cap`` are cr.
    """
    if cap <= 0.0:
        raise ValueError("cap must be > 0 (cr)")
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0 (daily vol, decimal)")
    return float(cfg.s0) + float(cfg.k_sigma) * float(sigma) + float(cfg.k_inv) * abs(float(inventory)) / float(cap)


def inventory_skew(inventory: float, cap: float, cfg: MMCfg) -> float:
    """Mid skew ``−k_skew·inv/cap`` (§6.5). Dimensionless.

    Long inventory (``inv > 0``, cr) skews the mid down so the MM fades the long.
    """
    if cap <= 0.0:
        raise ValueError("cap must be > 0 (cr)")
    return -float(cfg.k_skew) * float(inventory) / float(cap)


def quote_prices(
    mid: float,
    sigma: float,
    inventory: float,
    cap: float,
    cfg: MMCfg,
    *,
    symbol: str = "",
) -> Quote:
    """Bid/ask around the skewed mid (§6.5).

    ``skewed_mid = P · (1 + skew)``; quotes ``skewed_mid · (1 ∓ s/2)``.
    ``mid`` / bid / ask are cr/unit.
    """
    if mid <= 0.0:
        raise ValueError("mid must be > 0 (cr/unit)")
    spread = quoted_spread(sigma, inventory, cap, cfg)
    skew = inventory_skew(inventory, cap, cfg)
    skewed = float(mid) * (1.0 + skew)
    half = spread / 2.0  # §6.5 quotes P·(1 ∓ s/2)
    return Quote(
        symbol=symbol,
        mid=float(mid),
        skewed_mid=skewed,
        spread=spread,
        skew=skew,
        bid=skewed * (1.0 - half),
        ask=skewed * (1.0 + half),
    )


def liquidity_budget(adv: float, cfg: MMCfg) -> float:
    """Per-tick fill budget ``participation_cap × ADV`` (cr). §6.5."""
    if adv < 0.0:
        raise ValueError("ADV must be >= 0 (cr/day)")
    return float(cfg.participation_cap) * float(adv)


def _reject(reason: str, *, order_id: int | None = None) -> SubmitResult:
    return SubmitResult(order_id=order_id, status=OrderStatus.REJECTED, remaining=0, reason=reason)


def _side_sign(side: Side) -> float:
    return 1.0 if side is Side.BUY else -1.0


class EngineMM:
    """Engine-MM venue. ``submit`` parks; ``end_tick`` fills (next-tick)."""

    def __init__(self, cfg: MMCfg | None = None, *, impact_cfg: ImpactCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else MMCfg()
        self.impact_cfg = impact_cfg if impact_cfg is not None else ImpactCfg()
        self.clock_tick = 0
        self.last_fills: tuple[Fill, ...] = ()
        self._books: dict[str, _Book] = {}
        self._orders: dict[int, _Live] = {}
        self._agent_seq: dict[str, int] = {}
        self._next_order_id = 1

    @classmethod
    def from_mm_cfg(cls, cfg: MMCfg, *, impact_cfg: ImpactCfg | None = None) -> EngineMM:
        """Venue from ``markets.yaml`` ``mm:`` (spread/budget). Impact defaults from ``ImpactCfg``."""
        return cls(cfg, impact_cfg=impact_cfg)

    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._books))

    def add_book(
        self,
        symbol: str,
        *,
        mid: float,
        sigma: float,
        cap: float,
        adv: float,
        inventory: float = 0.0,
    ) -> None:
        """List one MM name. ``mid`` cr/unit; ``sigma`` daily vol; ``cap`` / ``adv`` / ``inventory`` cr."""
        if not symbol:
            raise ConfigError("symbol must be non-empty")
        if symbol in self._books:
            raise ConfigError(f"duplicate book {symbol!r}")
        if mid <= 0.0:
            raise ConfigError("mid must be > 0 (cr/unit)")
        if sigma < 0.0:
            raise ConfigError("sigma must be >= 0 (daily vol, decimal)")
        if cap <= 0.0:
            raise ConfigError("cap must be > 0 (cr)")
        if adv < 0.0:
            raise ConfigError("ADV must be >= 0 (cr/day)")
        self._books[symbol] = _Book(
            symbol=symbol,
            mid=float(mid),
            sigma=float(sigma),
            cap=float(cap),
            adv=float(adv),
            inventory=float(inventory),
            kernel=ImpactKernel(self.impact_cfg),
        )

    def set_mark(
        self,
        symbol: str,
        *,
        mid: float | None = None,
        sigma: float | None = None,
        cap: float | None = None,
        adv: float | None = None,
        inventory: float | None = None,
    ) -> None:
        """Update the public mark used for the next quote / match (units as ``add_book``)."""
        book = self._book(symbol)
        if mid is not None:
            if mid <= 0.0:
                raise ConfigError("mid must be > 0 (cr/unit)")
            book.mid = float(mid)
        if sigma is not None:
            if sigma < 0.0:
                raise ConfigError("sigma must be >= 0 (daily vol, decimal)")
            book.sigma = float(sigma)
        if cap is not None:
            if cap <= 0.0:
                raise ConfigError("cap must be > 0 (cr)")
            book.cap = float(cap)
        if adv is not None:
            if adv < 0.0:
                raise ConfigError("ADV must be >= 0 (cr/day)")
            book.adv = float(adv)
        if inventory is not None:
            book.inventory = float(inventory)

    def quote(self, symbol: str) -> Quote:
        """Current bid/ask from the stored mark and inventory (cr/unit)."""
        book = self._book(symbol)
        return quote_prices(book.mid, book.sigma, book.inventory, book.cap, self.cfg, symbol=symbol)

    def inventory(self, symbol: str) -> float:
        """Signed MM book (cr). Positive = long the instrument."""
        return self._book(symbol).inventory

    def resting_ids(self, symbol: str | None = None) -> tuple[int, ...]:
        """Live order ids, sorted. Optional ``symbol`` filter."""
        ids = [
            oid
            for oid, o in self._orders.items()
            if symbol is None or o.symbol == symbol
        ]
        return tuple(sorted(ids))

    def submit(self, order: Order) -> SubmitResult:
        """Accept and park. Fills happen on the next ``end_tick``."""
        live, err = self._accept(order)
        if err is not None:
            return err
        return SubmitResult(order_id=live.order_id, status=live.status, remaining=live.remaining)

    def ingest(self, orders: Sequence[Order]) -> list[SubmitResult]:
        """Lockstep intake: assign missing sequences, then ``(agent_id, sequence, symbol)``."""
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
        order = self._orders.get(int(order_id))
        if order is None or order.symbol != symbol:
            return _reject("unknown order", order_id=int(order_id))
        self._retire(order, OrderStatus.CANCELLED)
        return SubmitResult(
            order_id=order.order_id,
            status=OrderStatus.CANCELLED,
            remaining=0,
            reason="cancel",
        )

    def replace(
        self,
        symbol: str,
        order_id: int,
        *,
        qty: int | None = None,
        price: float | None = None,
    ) -> SubmitResult:
        """Amend a parked order. ``qty`` is the new remaining size (units)."""
        order = self._orders.get(int(order_id))
        if order is None or order.symbol != symbol:
            return _reject("unknown order", order_id=int(order_id))
        if qty is not None:
            if int(qty) < 0:
                return _reject("qty must be >= 0 (units)", order_id=order.order_id)
            if int(qty) == 0:
                return self.cancel(symbol, order.order_id)
            order.remaining = int(qty)
            order.qty = max(order.qty, int(qty))
        if price is not None:
            if float(price) <= 0.0:
                return _reject("price must be > 0 (cr/unit)", order_id=order.order_id)
            order.price = float(price)
        order.status = OrderStatus.PARTIAL if order.remaining < order.qty else OrderStatus.RESTING
        return SubmitResult(order_id=order.order_id, status=order.status, remaining=order.remaining)

    def end_tick(self) -> tuple[Fill, ...]:
        """Match parked orders against this tick's quotes (next-tick fills).

        Applies one net-signed impact step per name, then DAY expiries, then
        advances ``clock_tick``. Returns the fills; also stored as ``last_fills``.
        """
        fills: list[Fill] = []
        for symbol in self.symbols():
            fills.extend(self._match_symbol(symbol))
            self._expire_day(symbol)
        self.last_fills = tuple(fills)
        self.clock_tick += 1
        return self.last_fills

    def to_state(self) -> dict[str, Any]:
        return {
            "cfg": {
                "s0": self.cfg.s0,
                "k_sigma": self.cfg.k_sigma,
                "k_inv": self.cfg.k_inv,
                "k_skew": self.cfg.k_skew,
                "participation_cap": self.cfg.participation_cap,
            },
            "impact_cfg": {
                "delta": self.impact_cfg.delta,
                "beta": self.impact_cfg.beta,
                "Y": self.impact_cfg.Y,
                "tau0_d": self.impact_cfg.tau0_d,
                "half_lives_d": list(self.impact_cfg.half_lives_d),
            },
            "clock_tick": self.clock_tick,
            "next_order_id": self._next_order_id,
            "agent_seq": {k: self._agent_seq[k] for k in sorted(self._agent_seq)},
            "orders": [self._orders[i].to_state() for i in sorted(self._orders)],
            "books": {
                s: {
                    "symbol": b.symbol,
                    "mid": b.mid,
                    "sigma": b.sigma,
                    "cap": b.cap,
                    "adv": b.adv,
                    "inventory": b.inventory,
                    "last_price": b.last_price,
                    "kernel": b.kernel.to_state(),
                }
                for s, b in ((sym, self._books[sym]) for sym in self.symbols())
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> EngineMM:
        needed = ("cfg", "books", "clock_tick")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"engine MM state missing {missing}")
        impact_raw = dict(state.get("impact_cfg") or {})
        if "half_lives_d" in impact_raw:
            impact_raw["half_lives_d"] = tuple(impact_raw["half_lives_d"])
        impact_cfg = ImpactCfg(**impact_raw) if impact_raw else ImpactCfg()
        mm = cls(MMCfg(**state["cfg"]), impact_cfg=impact_cfg)
        mm.clock_tick = int(state["clock_tick"])
        mm._next_order_id = int(state.get("next_order_id", 1))
        mm._agent_seq = {str(k): int(v) for k, v in state.get("agent_seq", {}).items()}
        raw_books = state["books"]
        for symbol in sorted(raw_books):
            row = raw_books[symbol]
            book = _Book(
                symbol=str(row["symbol"]),
                mid=float(row["mid"]),
                sigma=float(row["sigma"]),
                cap=float(row["cap"]),
                adv=float(row["adv"]),
                inventory=float(row["inventory"]),
                kernel=ImpactKernel.from_state(row["kernel"]),
                last_price=None if row.get("last_price") is None else float(row["last_price"]),
            )
            mm._books[book.symbol] = book
        for row in state.get("orders", ()):
            live = _Live.from_state(row)
            mm._orders[live.order_id] = live
        return mm

    def _book(self, symbol: str) -> _Book:
        try:
            return self._books[symbol]
        except KeyError as exc:
            raise ConfigError(f"unknown book {symbol!r}") from exc

    def _accept(self, order: Order) -> tuple[_Live | None, SubmitResult | None]:
        if not order.symbol:
            return None, _reject("order.symbol is required")
        if order.symbol not in self._books:
            return None, _reject(f"unknown book {order.symbol!r}")
        if order.agent_id == MM_ACCOUNT:
            return None, _reject("MM cannot take its own quote")
        qty = int(order.qty)
        if qty <= 0:
            return None, _reject("qty must be > 0 (units)")
        if order.order_type is OrderType.LIMIT and order.price is None:
            return None, _reject("limit order requires price (cr/unit)")
        if order.order_type is OrderType.STOP and order.stop_price is None:
            return None, _reject("stop order requires stop_price (cr/unit)")
        if order.price is not None and float(order.price) <= 0.0:
            return None, _reject("price must be > 0 (cr/unit)")
        if order.stop_price is not None and float(order.stop_price) <= 0.0:
            return None, _reject("stop_price must be > 0 (cr/unit)")
        if order.sequence is None:
            sequence = self._agent_seq.get(order.agent_id, 0) + 1
            self._agent_seq[order.agent_id] = sequence
        else:
            sequence = int(order.sequence)
            if sequence >= self._agent_seq.get(order.agent_id, 0):
                self._agent_seq[order.agent_id] = sequence
        oid = self._next_order_id
        self._next_order_id += 1
        live = _Live(
            order_id=oid,
            agent_id=str(order.agent_id),
            sequence=sequence,
            side=Side(order.side),
            qty=qty,
            remaining=qty,
            order_type=OrderType(order.order_type),
            tif=TimeInForce(order.tif),
            price=None if order.price is None else float(order.price),
            stop_price=None if order.stop_price is None else float(order.stop_price),
            status=OrderStatus.RESTING,
            symbol=order.symbol,
        )
        self._orders[oid] = live
        return live, None

    def _match_symbol(self, symbol: str) -> list[Fill]:
        book = self._books[symbol]
        qte = quote_prices(book.mid, book.sigma, book.inventory, book.cap, self.cfg, symbol=symbol)
        budget_left = liquidity_budget(book.adv, self.cfg)
        lives = [
            o
            for o in self._orders.values()
            if o.symbol == symbol and o.status in (OrderStatus.RESTING, OrderStatus.PARTIAL)
        ]
        lives.sort(key=lambda o: (o.agent_id, o.sequence, o.order_id))

        planned: dict[int, int] = {}
        for order in lives:
            if not self._is_marketable(order, book, qte):
                continue
            max_qty = int(budget_left / book.mid) if book.mid > 0.0 else 0
            take = min(order.remaining, max_qty)
            if take > 0:
                planned[order.order_id] = take
                budget_left -= take * book.mid

        q_net = 0.0
        for order in lives:
            take = planned.get(order.order_id, 0)
            if take:
                q_net += _side_sign(order.side) * take * book.mid

        xi_pre = book.kernel.xi
        book.kernel.step(q_net, book.adv, book.sigma)
        mid_post = float(post_impact_mid(qte.skewed_mid, xi_pre, book.kernel.xi))
        buy_px = float(fill_price(mid_post, qte.spread, 1.0))
        sell_px = float(fill_price(mid_post, qte.spread, -1.0))

        fills: list[Fill] = []
        for order in lives:
            take = planned.get(order.order_id, 0)
            if take > 0:
                px = buy_px if order.side is Side.BUY else sell_px
                order.remaining -= take
                # Agent buy → MM sells → inventory (cr) falls.
                book.inventory -= _side_sign(order.side) * take * px
                book.last_price = px
                fills.append(
                    Fill(
                        symbol=symbol,
                        price=px,
                        qty=take,
                        maker=MM_ACCOUNT,
                        taker=order.agent_id,
                        maker_order_id=MM_QUOTE_ID,
                        taker_order_id=order.order_id,
                        notional=px * take,
                        tick=self.clock_tick,
                    )
                )
            self._finish_aggressor(order)
        return fills

    def _is_marketable(self, order: _Live, book: _Book, qte: Quote) -> bool:
        if order.order_type is OrderType.STOP and not order.triggered:
            ref = book.last_price if book.last_price is not None else book.mid
            stop = float(order.stop_price) if order.stop_price is not None else 0.0
            fire = ref >= stop if order.side is Side.BUY else ref <= stop
            if not fire:
                return False
            order.triggered = True
            if order.price is None:
                order.order_type = OrderType.MARKET
        if order.order_type is OrderType.MARKET or order.price is None:
            return True
        px = float(order.price)
        if order.side is Side.BUY:
            return px >= qte.ask
        return px <= qte.bid

    def _finish_aggressor(self, order: _Live) -> None:
        if order.order_id not in self._orders:
            return
        if order.remaining <= 0:
            self._retire(order, OrderStatus.FILLED)
            return
        if order.tif is TimeInForce.IOC:
            self._retire(order, OrderStatus.CANCELLED)
            return
        order.status = OrderStatus.PARTIAL if order.remaining < order.qty else OrderStatus.RESTING

    def _expire_day(self, symbol: str) -> tuple[int, ...]:
        expired: list[int] = []
        for oid in list(self._orders):
            order = self._orders.get(oid)
            if order is None or order.symbol != symbol or order.tif is not TimeInForce.DAY:
                continue
            self._retire(order, OrderStatus.EXPIRED)
            expired.append(oid)
        return tuple(expired)

    def _retire(self, order: _Live, status: OrderStatus) -> None:
        order.status = status
        if status is not OrderStatus.PARTIAL:
            order.remaining = 0
        self._orders.pop(order.order_id, None)
