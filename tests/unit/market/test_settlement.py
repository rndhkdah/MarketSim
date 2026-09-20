"""T6.12 — settlement, fees, taxes (§6.5 / §6.10)."""

from __future__ import annotations

import pytest

from marketsim.core.config import Config
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent
from marketsim.market.instruments import FeesCfg
from marketsim.market.mm import MM_ACCOUNT
from marketsim.market.settlement import (
    settle_fill,
    settle_npc_dividends,
    settle_variation_margin,
)
from marketsim.market.venue import Fill, Side

SYM = "EQ:NPC:AUTOS"
OIL = "OIL"


def _ledger(*entities: str) -> Ledger:
    led = Ledger.empty(debug_journal=True)
    for name in entities:
        led.register_entity(name)
    return led


def _fill(
    *,
    symbol: str = SYM,
    price: float = 10.0,
    qty: int = 4,
    maker: str = MM_ACCOUNT,
    taker: str = "alice",
    tick: int = 1,
    maker_order_id: int = 0,
    taker_order_id: int = 1,
) -> Fill:
    return Fill(
        symbol=symbol,
        price=price,
        qty=qty,
        maker=maker,
        taker=taker,
        maker_order_id=maker_order_id,
        taker_order_id=taker_order_id,
        notional=price * qty,
        tick=tick,
    )


def _sums(tx: Tx) -> dict[str, float]:
    out: dict[str, float] = {}
    for e in tx.entries:
        out[e.instrument] = out.get(e.instrument, 0.0) + float(e.amount)
    return out


def test_every_fill_is_balanced_cash_for_asset_tx() -> None:
    led = _ledger("alice", "bob", MM_ACCOUNT)
    buy = _fill(price=12.5, qty=8)  # notional 100 cr
    txs = settle_fill(led, buy, Side.BUY)
    assert [tx.tag for tx in txs] == ["equity_trade"]
    trade = txs[0]
    assert trade.tick == buy.tick
    assert _sums(trade)["DEP"] == pytest.approx(0.0)
    assert _sums(trade)[SYM] == pytest.approx(0.0)
    # Taker bought: +shares / −DEP. Maker (MM) is the short. Never clipped.
    assert led.position("alice", SYM) == pytest.approx(8.0)
    assert led.position(MM_ACCOUNT, SYM) == pytest.approx(-8.0)
    assert led.position("alice", "DEP") == pytest.approx(-100.0)
    assert led.position(MM_ACCOUNT, "DEP") == pytest.approx(100.0)

    sell = _fill(taker="bob", qty=3, price=10.0, tick=2, taker_order_id=2)
    sell_txs = settle_fill(led, sell, Side.SELL)
    assert sell_txs[0].tag == "equity_trade"
    assert _sums(sell_txs[0])["DEP"] == pytest.approx(0.0)
    assert _sums(sell_txs[0])[SYM] == pytest.approx(0.0)
    assert led.position("bob", SYM) == pytest.approx(-3.0)
    assert led.position("bob", "DEP") == pytest.approx(30.0)
    assert led.position(MM_ACCOUNT, SYM) == pytest.approx(-5.0)
    assert_consistent(led)


def test_fees_and_transaction_tax_routed(cfg: Config) -> None:
    assert cfg.markets is not None
    assert cfg.markets.fees.commission == pytest.approx(0.0)
    assert cfg.markets.fees.transaction_tax == pytest.approx(0.0)

    led0 = _ledger("alice", "bob", MM_ACCOUNT, "GOVT")
    zero = settle_fill(led0, _fill(maker="bob"), Side.BUY, fees=cfg.markets.fees)
    assert [tx.tag for tx in zero] == ["equity_trade"]
    assert led0.position("GOVT", "DEP") == pytest.approx(0.0)
    assert led0.position(MM_ACCOUNT, "DEP") == pytest.approx(0.0)

    # CLOB-style: two agents. Commission → MM, tax → GOVT (§6.10).
    led = _ledger("alice", "bob", MM_ACCOUNT, "GOVT")
    fill = _fill(maker="bob", taker="alice", price=10.0, qty=10)  # notional 100 cr
    fees = FeesCfg(commission=0.01, transaction_tax=0.02)
    txs = settle_fill(led, fill, Side.BUY, fees=fees)
    assert [tx.tag for tx in txs] == ["equity_trade", "fees", "transaction_tax"]
    # Taker pays commission; buyer pays tax. Destinations from §6.10 / markets.yaml.
    assert led.position("alice", "DEP") == pytest.approx(-103.0)
    assert led.position("bob", "DEP") == pytest.approx(100.0)
    assert led.position(MM_ACCOUNT, "DEP") == pytest.approx(1.0)
    assert led.position("GOVT", "DEP") == pytest.approx(2.0)
    for tx in txs:
        for total in _sums(tx).values():
            assert total == pytest.approx(0.0)
    assert_consistent(led)


def test_npc_dividends_reach_holders_prorata() -> None:
    led = _ledger("alice", "bob", "NPC:0:AUTOS")
    led.register_instrument(SYM, financial=True)
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry("alice", SYM, 60.0),
                Entry("bob", SYM, 40.0),
                Entry("NPC:0:AUTOS", SYM, -100.0),
            ),
        )
    )
    tx = settle_npc_dividends(led, SYM, 10.0, tick=1)
    assert tx is not None
    assert tx.tag == "dividends"
    assert _sums(tx)["DEP"] == pytest.approx(0.0)
    assert led.position("alice", "DEP") == pytest.approx(6.0)
    assert led.position("bob", "DEP") == pytest.approx(4.0)
    assert led.position("NPC:0:AUTOS", "DEP") == pytest.approx(-10.0)
    assert_consistent(led)


def test_commodity_variation_margin_nets_to_zero() -> None:
    led = _ledger("alice", "bob", MM_ACCOUNT)
    led.register_instrument(OIL, financial=True)
    led.post(
        Tx(
            0,
            "opening",
            (
                Entry("alice", OIL, 10.0),
                Entry("bob", OIL, 5.0),
                Entry(MM_ACCOUNT, OIL, -15.0),
            ),
        )
    )
    tx = settle_variation_margin(led, OIL, delta_price=2.0, tick=1)
    assert tx is not None
    assert _sums(tx)["DEP"] == pytest.approx(0.0)
    assert led.position("alice", "DEP") == pytest.approx(20.0)
    assert led.position("bob", "DEP") == pytest.approx(10.0)
    assert led.position(MM_ACCOUNT, "DEP") == pytest.approx(-30.0)
    assert sum(led.position(e, "DEP") for e in led.entities.names) == pytest.approx(0.0)
    assert led.position("alice", OIL) == pytest.approx(10.0)
    assert led.position(MM_ACCOUNT, OIL) == pytest.approx(-15.0)
    assert_consistent(led)
