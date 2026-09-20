"""T6.10 — price-time CLOB for EQ:FIRM:* (§6.6)."""

from __future__ import annotations

import time

import pytest

from marketsim.core.config import Config
from marketsim.market.clob import (
    CLOB,
    BookStatus,
    Order,
    OrderBook,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)

SYM = "EQ:FIRM:acme"


def _venue(cfg: Config, *, ref: float | None = 10.0) -> tuple[CLOB, OrderBook]:
    assert cfg.markets is not None
    clob = CLOB.from_clob_cfg(cfg.markets.clob)
    book = clob.add_book(SYM, reference_price=ref)
    return clob, book


def _limit(
    agent: str,
    side: Side,
    qty: int,
    price: float,
    *,
    tif: TimeInForce = TimeInForce.GTC,
    sequence: int | None = None,
) -> Order:
    return Order(
        agent_id=agent,
        side=side,
        qty=qty,
        symbol=SYM,
        order_type=OrderType.LIMIT,
        tif=tif,
        price=price,
        sequence=sequence,
    )


def _market(
    agent: str,
    side: Side,
    qty: int,
    *,
    tif: TimeInForce = TimeInForce.GTC,
    sequence: int | None = None,
) -> Order:
    return Order(
        agent_id=agent,
        side=side,
        qty=qty,
        symbol=SYM,
        order_type=OrderType.MARKET,
        tif=tif,
        sequence=sequence,
    )


def _stop(
    agent: str,
    side: Side,
    qty: int,
    stop_price: float,
    *,
    tif: TimeInForce = TimeInForce.GTC,
    sequence: int | None = None,
) -> Order:
    return Order(
        agent_id=agent,
        side=side,
        qty=qty,
        symbol=SYM,
        order_type=OrderType.STOP,
        tif=tif,
        stop_price=stop_price,
        sequence=sequence,
    )


def test_price_time_priority(cfg: Config) -> None:
    _clob, book = _venue(cfg)
    first = book.submit(_limit("alice", Side.SELL, 10, 10.00, sequence=1))
    second = book.submit(_limit("bob", Side.SELL, 10, 10.00, sequence=1))
    assert book.level_order_ids(Side.SELL, 10.00) == (first.order_id, second.order_id)
    hit = book.submit(_market("carol", Side.BUY, 10))
    assert len(hit.fills) == 1
    assert hit.fills[0].maker == "alice"
    assert hit.fills[0].taker == "carol"
    assert hit.fills[0].qty == 10
    assert hit.fills[0].price == pytest.approx(10.00)
    assert hit.fills[0].notional == pytest.approx(100.00)
    assert book.level_order_ids(Side.SELL, 10.00) == (second.order_id,)


def test_partial_fills(cfg: Config) -> None:
    _clob, book = _venue(cfg)
    rest = book.submit(_limit("alice", Side.SELL, 10, 10.00))
    part = book.submit(_limit("bob", Side.BUY, 4, 10.00))
    assert part.status is OrderStatus.FILLED
    assert len(part.fills) == 1
    assert part.fills[0].qty == 4
    assert part.fills[0].maker == "alice"
    assert part.fills[0].notional == pytest.approx(40.00)
    assert rest.order_id in book.resting_ids()
    assert book.level_order_ids(Side.SELL, 10.00) == (rest.order_id,)
    leftover = book.submit(_market("carol", Side.BUY, 6))
    assert leftover.status is OrderStatus.FILLED
    assert leftover.fills[0].qty == 6
    assert rest.order_id not in book.resting_ids()
    assert book.best_ask is None


def test_stop_trigger(cfg: Config) -> None:
    _clob, book = _venue(cfg)
    book.submit(_limit("alice", Side.BUY, 10, 10.00))
    parked = book.submit(_stop("bob", Side.SELL, 3, 10.00))
    assert parked.status is OrderStatus.RESTING
    assert parked.remaining == 3
    assert book.best_ask is None
    trig = book.submit(_limit("carol", Side.SELL, 1, 10.00))
    assert trig.status is OrderStatus.FILLED
    assert sum(f.qty for f in trig.fills) == 4
    assert {f.taker for f in trig.fills} == {"carol", "bob"}
    assert all(f.maker == "alice" for f in trig.fills)
    assert book.last_price == pytest.approx(10.00)
    assert parked.order_id not in book.resting_ids()
    # 1 (carol) + 3 (bob stop) = 4 shares lifted from alice's 10; 6 should rest.
    assert book.best_bid == pytest.approx(10.00)
    assert len(book.level_order_ids(Side.BUY, 10.00)) == 1


def test_ioc_gtc_day_and_expiries(cfg: Config) -> None:
    _clob, book = _venue(cfg)
    book.submit(_limit("alice", Side.SELL, 3, 10.00))
    ioc = book.submit(_limit("bob", Side.BUY, 10, 10.00, tif=TimeInForce.IOC))
    assert ioc.status is OrderStatus.CANCELLED
    assert sum(f.qty for f in ioc.fills) == 3
    assert book.best_bid is None
    assert ioc.order_id not in book.resting_ids()

    book.submit(_limit("alice", Side.SELL, 3, 10.00))
    gtc = book.submit(_limit("carol", Side.BUY, 10, 10.00, tif=TimeInForce.GTC))
    assert gtc.status is OrderStatus.PARTIAL
    assert gtc.remaining == 7
    assert book.best_bid == pytest.approx(10.00)
    book.end_tick()
    assert gtc.order_id in book.resting_ids()
    assert book.best_bid == pytest.approx(10.00)

    book.cancel(gtc.order_id)
    book.submit(_limit("alice", Side.SELL, 3, 10.00))
    day = book.submit(_limit("dave", Side.BUY, 10, 10.00, tif=TimeInForce.DAY))
    assert day.status is OrderStatus.PARTIAL
    assert day.order_id in book.resting_ids()
    _auction, expired = book.end_tick()
    assert day.order_id in expired
    assert day.order_id not in book.resting_ids()
    assert book.best_bid is None


def test_cancel_replace_loses_priority_on_size_up_only(cfg: Config) -> None:
    _clob, book = _venue(cfg)
    a = book.submit(_limit("alice", Side.SELL, 10, 10.00, sequence=1))
    b = book.submit(_limit("bob", Side.SELL, 10, 10.00, sequence=1))
    assert book.level_order_ids(Side.SELL, 10.00) == (a.order_id, b.order_id)
    down = book.replace(a.order_id, qty=5)
    assert down.order_id == a.order_id
    assert down.remaining == 5
    assert book.level_order_ids(Side.SELL, 10.00) == (a.order_id, b.order_id)
    hit = book.submit(_market("carol", Side.BUY, 5))
    assert hit.fills[0].maker == "alice"
    assert book.level_order_ids(Side.SELL, 10.00) == (b.order_id,)

    book.cancel(b.order_id)
    a2 = book.submit(_limit("alice", Side.SELL, 10, 10.00, sequence=2))
    b2 = book.submit(_limit("bob", Side.SELL, 10, 10.00, sequence=2))
    up = book.replace(a2.order_id, qty=15)
    assert up.order_id == a2.order_id
    assert up.remaining == 15
    assert book.level_order_ids(Side.SELL, 10.00) == (b2.order_id, a2.order_id)
    hit2 = book.submit(_market("carol", Side.BUY, 10))
    assert hit2.fills[0].maker == "bob"
    assert book.level_order_ids(Side.SELL, 10.00) == (a2.order_id,)


def test_self_trade_prevention_cancels_newest(cfg: Config) -> None:
    _clob, book = _venue(cfg)
    rest = book.submit(_limit("alice", Side.SELL, 10, 10.00, sequence=1))
    cross = book.submit(_limit("alice", Side.BUY, 10, 10.00, sequence=2))
    assert cross.status is OrderStatus.CANCELLED
    assert cross.fills == ()
    assert rest.order_id in book.resting_ids()
    assert book.best_ask == pytest.approx(10.00)
    assert book.best_bid is None
    assert not book.is_crossed()


def test_call_auction_clears_at_volume_maximising_price(cfg: Config) -> None:
    # Unique volume-max: 10.00 trades 120 vs 80 at 9.95 and 50 at 10.05.
    clob, book = _venue(cfg, ref=None)
    assert book.status is BookStatus.AUCTION
    book.submit(_limit("b1", Side.BUY, 100, 10.00, sequence=1))
    book.submit(_limit("b2", Side.BUY, 50, 10.05, sequence=1))
    book.submit(_limit("s1", Side.SELL, 80, 9.95, sequence=1))
    book.submit(_limit("s2", Side.SELL, 40, 10.00, sequence=1))
    result = book.uncross()
    assert result.price == pytest.approx(10.00)
    assert result.volume == 120
    assert sum(f.qty for f in result.fills) == 120
    assert all(f.price == pytest.approx(10.00) for f in result.fills)
    assert book.status is BookStatus.CONTINUOUS
    assert book.last_price == pytest.approx(10.00)
    assert not book.is_crossed()

    # Range of max-volume prices: pick the tick closest to the reference.
    book2 = clob.add_book("EQ:FIRM:range", reference_price=None)
    book2.set_reference(10.04)
    book2.start_auction()
    book2.submit(
        Order(
            agent_id="buyer",
            side=Side.BUY,
            qty=10,
            symbol="EQ:FIRM:range",
            price=10.10,
            sequence=1,
        )
    )
    book2.submit(
        Order(
            agent_id="seller",
            side=Side.SELL,
            qty=10,
            symbol="EQ:FIRM:range",
            price=10.00,
            sequence=1,
        )
    )
    ranged = book2.uncross()
    assert ranged.volume == 10
    assert ranged.price == pytest.approx(10.04)


def test_halt_band(cfg: Config) -> None:
    assert cfg.markets is not None
    assert cfg.markets.clob.halt_band == pytest.approx(0.20)
    _clob, book = _venue(cfg, ref=10.0)

    book.submit(_limit("alice", Side.SELL, 10, 12.00))
    inside = book.submit(_market("bob", Side.BUY, 10))
    assert inside.status is OrderStatus.FILLED
    assert inside.fills[0].price == pytest.approx(12.00)
    assert not book.is_halted
    assert book.status is BookStatus.CONTINUOUS

    book.set_reference(10.0)
    book.submit(_limit("alice", Side.SELL, 10, 12.01))
    outside = book.submit(_limit("carol", Side.BUY, 10, 12.50))
    assert outside.fills == ()
    assert book.is_halted
    assert book.status is BookStatus.AUCTION
    assert book.is_crossed()
    auction = book.uncross()
    assert auction.volume == 10
    assert auction.price == pytest.approx(12.01)
    assert not book.is_halted
    assert book.status is BookStatus.CONTINUOUS
    assert book.last_price == pytest.approx(12.01)
    assert not book.is_crossed()


def test_no_crossed_book_after_any_sequence(cfg: Config) -> None:
    _clob, book = _venue(cfg, ref=10.0)

    def _assert_continuous_uncrossed() -> None:
        if book.status is BookStatus.CONTINUOUS:
            assert not book.is_crossed()
            bid, ask = book.best_bid, book.best_ask
            if bid is not None and ask is not None:
                assert bid < ask

    actions: list[Order | tuple[str, object, ...]] = []
    for i in range(8):
        px = 9.90 + 0.02 * i
        actions.append(_limit(f"a{i}", Side.BUY, 4 + i, round(px, 2), sequence=i))
        actions.append(_limit(f"z{i}", Side.SELL, 4 + i, round(10.20 - 0.02 * i, 2), sequence=i))
    actions.append(_market("m1", Side.BUY, 5, sequence=1))
    actions.append(_market("m2", Side.SELL, 5, sequence=1))
    actions.append(_limit("ioc", Side.BUY, 20, 10.50, tif=TimeInForce.IOC, sequence=1))
    actions.append(_limit("day", Side.SELL, 7, 10.30, tif=TimeInForce.DAY, sequence=1))
    actions.append(_stop("stp", Side.BUY, 3, 10.20, sequence=1))
    actions.append(_limit("alice", Side.SELL, 8, 10.00, sequence=9))
    actions.append(_limit("alice", Side.BUY, 8, 10.00, sequence=10))
    actions.append(_limit("big", Side.BUY, 50, 12.50, sequence=1))

    live_ids: list[int] = []
    for i, action in enumerate(actions):
        result = book.submit(action)
        if result.order_id is not None and result.order_id in book.resting_ids():
            live_ids.append(result.order_id)
        if i == 4 and live_ids:
            book.replace(live_ids[0], qty=max(book.lot, 2))
        if i == 6 and live_ids:
            book.replace(live_ids[-1], qty=20)
        if i == 10 and live_ids:
            book.cancel(live_ids[len(live_ids) // 2])
        if book.is_halted:
            book.uncross()
        _assert_continuous_uncrossed()

    book.ingest(
        [
            _limit("ing_a", Side.BUY, 3, 9.80, sequence=1),
            _limit("ing_b", Side.SELL, 3, 10.40, sequence=1),
            _market("ing_a", Side.SELL, 1, sequence=2),
        ]
    )
    if book.is_halted:
        book.uncross()
    book.end_tick()
    _assert_continuous_uncrossed()

    restored = OrderBook.from_state(book.to_state())
    assert restored.to_state() == book.to_state()
    if restored.status is BookStatus.CONTINUOUS:
        assert not restored.is_crossed()


def test_throughput_at_least_20000_events_per_s(cfg: Config) -> None:
    _clob, book = _venue(cfg, ref=10.0)
    book.submit(_limit("seed_b", Side.BUY, 100, 9.95))
    book.submit(_limit("seed_s", Side.SELL, 100, 10.05))
    n = 12_000
    t0 = time.perf_counter()
    events = 0
    for i in range(n):
        agent = f"p{i % 17}"
        if i % 5 == 0:
            book.submit(_market(agent, Side.BUY if i % 2 == 0 else Side.SELL, 1, tif=TimeInForce.IOC))
            events += 1
        elif i % 5 == 1:
            res = book.submit(_limit(agent, Side.BUY, 2, 9.90 + (i % 7) * 0.01, tif=TimeInForce.IOC))
            events += 1
            if res.order_id is not None and res.order_id in book.resting_ids():
                book.cancel(res.order_id)
                events += 1
        elif i % 5 == 2:
            res = book.submit(_limit(agent, Side.SELL, 2, 10.10 - (i % 7) * 0.01))
            events += 1
            if i % 11 == 0 and res.order_id is not None and res.order_id in book.resting_ids():
                book.replace(res.order_id, qty=4)
                events += 1
            elif res.order_id is not None and res.order_id in book.resting_ids() and i % 3 == 0:
                book.cancel(res.order_id)
                events += 1
        else:
            book.submit(_limit(agent, Side.BUY if i % 2 else Side.SELL, 1, 10.00))
            events += 1
    elapsed = time.perf_counter() - t0
    rate = events / elapsed if elapsed > 0 else float("inf")
    assert events >= 12_000
    assert rate >= 20_000, f"{rate:.0f} events/s over {events} events in {elapsed:.3f}s"
