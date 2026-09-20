"""T6.20 — scripted wash, circular and pump are flagged; honest MM is not."""

from __future__ import annotations

from marketsim.core.config import Config
from marketsim.market.clob import CLOB, Fill, Order, OrderType, Side, TimeInForce
from marketsim.market.mm import MM_ACCOUNT
from marketsim.market.surveillance import FlagKind, Surveillance, counterparties

SYM = "EQ:FIRM:acme"


def _surv(cfg: Config) -> Surveillance:
    assert cfg.markets is not None
    return Surveillance(cfg.markets.surveillance)


def _fill(
    *,
    maker: str,
    taker: str,
    side: Side,
    qty: int = 10,
    price: float = 10.0,
    tick: int = 0,
    symbol: str = SYM,
    maker_order_id: int = 1,
    taker_order_id: int = 2,
) -> tuple[Fill, Side]:
    return (
        Fill(
            symbol=symbol,
            price=price,
            qty=qty,
            maker=maker,
            taker=taker,
            maker_order_id=maker_order_id,
            taker_order_id=taker_order_id,
            notional=price * qty,
            tick=tick,
        ),
        side,
    )


def _step(
    surv: Surveillance,
    prints: list[tuple[Fill, Side]],
    *,
    prices: dict[str, float] | None = None,
    tick: int | None = None,
):
    fills = [p[0] for p in prints]
    sides = [p[1] for p in prints]
    return surv.step(fills, sides, prices=prices, tick=tick)


def _kinds(flags) -> set[FlagKind]:
    return {f.kind for f in flags}


def test_yaml_keys_are_the_surveillance_windows(cfg: Config) -> None:
    assert cfg.markets is not None
    s = cfg.markets.surveillance
    assert s.wash_window_ticks == 1
    assert s.circular_n == 3
    assert s.pump_volume_share == 0.50
    assert s.pump_runup == 0.20
    assert s.pump_window_d == 5


def test_scripted_wash_trade_is_flagged(cfg: Config) -> None:
    # Two-way A↔B inside wash_window_ticks (same tick).
    surv = _surv(cfg)
    flags = _step(
        surv,
        [
            _fill(maker="bob", taker="alice", side=Side.BUY, tick=0),
            _fill(maker="alice", taker="bob", side=Side.BUY, tick=0, maker_order_id=3, taker_order_id=4),
        ],
        tick=0,
    )
    wash = [f for f in flags if f.kind is FlagKind.WASH]
    assert wash
    assert wash[0].symbol == SYM
    assert wash[0].agents == ("alice", "bob")
    assert FlagKind.CIRCULAR not in _kinds(flags)

    # Self-cross on the tape (CLOB would have cancelled newest).
    surv = _surv(cfg)
    flags = _step(
        surv,
        [_fill(maker="alice", taker="alice", side=Side.BUY, tick=0)],
        tick=0,
    )
    assert any(f.kind is FlagKind.WASH and f.agents == ("alice",) for f in flags)

    # Reverse one tick later is outside wash_window_ticks=1.
    surv = _surv(cfg)
    _step(surv, [_fill(maker="bob", taker="alice", side=Side.BUY, tick=0)], tick=0)
    later = _step(
        surv,
        [_fill(maker="alice", taker="bob", side=Side.BUY, tick=1, maker_order_id=3, taker_order_id=4)],
        tick=1,
    )
    assert FlagKind.WASH not in _kinds(later)


def test_scripted_circular_trade_is_flagged(cfg: Config) -> None:
    # A → B → C → A inside circular_n ticks.
    surv = _surv(cfg)
    _step(surv, [_fill(maker="ann", taker="ben", side=Side.BUY, tick=0)], tick=0)
    _step(
        surv,
        [_fill(maker="ben", taker="cam", side=Side.BUY, tick=1, maker_order_id=3, taker_order_id=4)],
        tick=1,
    )
    flags = _step(
        surv,
        [_fill(maker="cam", taker="ann", side=Side.BUY, tick=2, maker_order_id=5, taker_order_id=6)],
        tick=2,
    )
    circ = [f for f in flags if f.kind is FlagKind.CIRCULAR]
    assert circ
    assert circ[0].agents == ("ann", "ben", "cam")
    assert FlagKind.WASH not in _kinds(flags)

    # Same ring stretched past circular_n=3 is not circular.
    surv = _surv(cfg)
    _step(surv, [_fill(maker="ann", taker="ben", side=Side.BUY, tick=0)], tick=0)
    _step(
        surv,
        [_fill(maker="ben", taker="cam", side=Side.BUY, tick=1, maker_order_id=3, taker_order_id=4)],
        tick=1,
    )
    late = _step(
        surv,
        [_fill(maker="cam", taker="ann", side=Side.BUY, tick=3, maker_order_id=5, taker_order_id=6)],
        tick=3,
    )
    assert FlagKind.CIRCULAR not in _kinds(late)


def test_scripted_pump_is_flagged(cfg: Config) -> None:
    assert cfg.markets is not None
    surv = _surv(cfg)
    # Five-day marks 10 → 12.1 (21 % > pump_runup); alice takes > pump_volume_share.
    flags = None
    for t, px in enumerate((10.0, 10.5, 11.0, 11.5, 12.1)):
        flags = _step(
            surv,
            [
                _fill(
                    maker=MM_ACCOUNT,
                    taker="alice",
                    side=Side.BUY,
                    qty=80,
                    price=px,
                    tick=t,
                ),
                _fill(
                    maker=MM_ACCOUNT,
                    taker="bob",
                    side=Side.BUY,
                    qty=20,
                    price=px,
                    tick=t,
                    maker_order_id=3,
                    taker_order_id=4,
                ),
            ],
            tick=t,
        )
    assert flags is not None
    pump = [f for f in flags if f.kind is FlagKind.PUMP]
    assert pump
    assert pump[0].agents == ("alice",)
    assert "bob" not in {a for f in pump for a in f.agents}
    assert MM_ACCOUNT not in {a for f in pump for a in f.agents}


def test_honest_market_making_is_not_flagged(cfg: Config) -> None:
    assert cfg.markets is not None
    clob = CLOB.from_clob_cfg(cfg.markets.clob)
    book = clob.add_book(SYM, reference_price=10.0)
    surv = _surv(cfg)

    def _limit(agent: str, side: Side, qty: int, price: float) -> Order:
        return Order(
            agent_id=agent,
            side=side,
            qty=qty,
            symbol=SYM,
            order_type=OrderType.LIMIT,
            tif=TimeInForce.GTC,
            price=price,
        )

    for t in range(cfg.markets.surveillance.pump_window_d + 2):
        book.set_reference(10.0)
        ask = book.submit(_limit(MM_ACCOUNT, Side.SELL, 50, 10.01))
        bid = book.submit(_limit(MM_ACCOUNT, Side.BUY, 50, 9.99))
        assert ask.order_id is not None and bid.order_id is not None
        lift = book.submit(_limit("bob", Side.BUY, 10, 10.01))
        hit = book.submit(_limit("carol", Side.SELL, 10, 9.99))
        prints: list[tuple[Fill, Side]] = []
        for fill in lift.fills:
            prints.append((fill, Side.BUY))
            assert counterparties(fill, Side.BUY) == ("bob", MM_ACCOUNT)
        for fill in hit.fills:
            prints.append((fill, Side.SELL))
            assert counterparties(fill, Side.SELL) == (MM_ACCOUNT, "carol")
        flags = _step(surv, prints, prices={SYM: 10.0}, tick=t)
        assert flags == ()
        if ask.order_id in book.resting_ids():
            book.cancel(ask.order_id)
        if bid.order_id in book.resting_ids():
            book.cancel(bid.order_id)
        clob.end_tick()
