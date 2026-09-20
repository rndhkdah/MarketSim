"""Fill settlement, fees, NPC dividends and commodity variation margin (T6.12).

Every fill is a balanced cash-for-asset ``Tx`` (taker/maker swap ``DEP`` and the
instrument). Commission and transaction tax are shares of notional from
``markets.yaml`` ``fees:`` (defaults 0; §6.10). Tax goes to ``GOVT``; commission
goes to ``MM`` (the NPC-investor / broker book — no separate broker entity).
NPC dividends (tag ``dividends``) and commodity variation margin are pro-rata
on holdings so each financial instrument sums to 0. Never clipped.
"""

from __future__ import annotations

from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.market.instruments import FeesCfg
from marketsim.market.mm import MM_ACCOUNT
from marketsim.market.venue import Fill, Side

# Ledger cash is deposits. ``markets.yaml`` ``CASH`` is the non-tradable name.
CASH = "DEP"
# §6.5 / §6.10: commission sink. Prefer MM; ledger.yaml has no broker entity.
COMMISSION_TO = MM_ACCOUNT
# §6.10 / markets.yaml ``fees.transaction_tax`` → GOVT.
TAX_TO = "GOVT"

TRADE_TAG = "equity_trade"
FEES_TAG = "fees"
TAX_TAG = "transaction_tax"
DIVIDEND_TAG = "dividends"
# Daily cash MTM on cash-settled commodities (§6.2). Must be a FLOW_TAGS name
# so the TFM records the DEP move (``assert_consistent``).
VARIATION_MARGIN_TAG = "capital_transfer"

# Skip dust; same floor as auction / bond posting (ledger rejects |sum| > 1e-9).
_EPS = 1e-15


def _ensure_financial(ledger: Ledger, instrument: str) -> None:
    if instrument not in ledger.instruments:
        ledger.register_instrument(instrument, financial=True)


def _post(ledger: Ledger, tx: Tx) -> Tx:
    ledger.post(tx)
    return tx


def settle_fill(
    ledger: Ledger,
    fill: Fill,
    taker_side: Side,
    *,
    fees: FeesCfg | None = None,
    cash: str = CASH,
    commission_to: str = COMMISSION_TO,
    tax_to: str = TAX_TO,
) -> tuple[Tx, ...]:
    """Post one fill as cash-for-asset plus optional fees/tax.

    ``fill.price`` is cr/unit; ``fill.qty`` is units; ``fill.notional`` is cr
    (``price * qty``). ``taker_side`` is the aggressor's buy/sell — ``Fill``
    does not carry side. Commission and tax are shares of notional
    (``FeesCfg``, defaults 0). Units: cr.
    """
    qty = float(fill.qty)
    notional = float(fill.notional)
    if abs(qty) < _EPS and abs(notional) < _EPS:
        return ()
    if taker_side is Side.BUY:
        buyer, seller = fill.taker, fill.maker
    else:
        buyer, seller = fill.maker, fill.taker
    _ensure_financial(ledger, fill.symbol)
    posted: list[Tx] = []
    if fill.taker != fill.maker and (abs(qty) >= _EPS or abs(notional) >= _EPS):
        posted.append(
            _post(
                ledger,
                Tx(
                    fill.tick,
                    TRADE_TAG,
                    (
                        Entry(buyer, fill.symbol, qty),
                        Entry(seller, fill.symbol, -qty),
                        Entry(buyer, cash, -notional),
                        Entry(seller, cash, notional),
                    ),
                    memo=fill.symbol,
                ),
            )
        )
    cfg = fees if fees is not None else FeesCfg()
    commission = float(cfg.commission) * notional
    tax = float(cfg.transaction_tax) * notional
    if abs(commission) >= _EPS and fill.taker != commission_to:
        posted.append(
            _post(
                ledger,
                Tx(
                    fill.tick,
                    FEES_TAG,
                    (
                        Entry(fill.taker, cash, -commission),
                        Entry(commission_to, cash, commission),
                    ),
                    memo=fill.symbol,
                ),
            )
        )
    if abs(tax) >= _EPS and buyer != tax_to:
        posted.append(
            _post(
                ledger,
                Tx(
                    fill.tick,
                    TAX_TAG,
                    (
                        Entry(buyer, cash, -tax),
                        Entry(tax_to, cash, tax),
                    ),
                    memo=fill.symbol,
                ),
            )
        )
    return tuple(posted)


def settle_npc_dividends(
    ledger: Ledger,
    symbol: str,
    amount: float,
    tick: int,
    *,
    cash: str = CASH,
) -> Tx | None:
    """Pay ``amount`` (cr) of NPC dividends on ``symbol`` pro-rata to holdings.

    Longs receive ``q / Q+ · amount``; shorts (the issuer book) pay the same
    rate so ``DEP`` sums to 0. Tag ``dividends``. Never clipped.
    """
    if symbol not in ledger.instruments or abs(float(amount)) < _EPS:
        return None
    holdings: list[tuple[str, float]] = []
    long = 0.0
    for name in ledger.entities.names:
        q = ledger.position(name, symbol)
        if abs(q) < _EPS:
            continue
        holdings.append((name, q))
        if q > 0.0:
            long += q
    if long <= _EPS or not holdings:
        return None
    dps = float(amount) / long  # cr / unit
    entries = tuple(Entry(name, cash, q * dps) for name, q in holdings)
    return _post(ledger, Tx(tick, DIVIDEND_TAG, entries, memo=f"{symbol} dividends"))


def settle_variation_margin(
    ledger: Ledger,
    symbol: str,
    delta_price: float,
    tick: int,
    *,
    cash: str = CASH,
) -> Tx | None:
    """Cash-settle ``delta_price`` (cr/unit) on a commodity.

    Entity ``q`` receives ``q · ΔP`` DEP. Because the instrument sums to 0
    across counterparties, DEP nets to 0. Never clipped. Tag
    ``capital_transfer`` (FLOW_TAGS; memo names variation margin).
    """
    if symbol not in ledger.instruments or abs(float(delta_price)) < _EPS:
        return None
    entries: list[Entry] = []
    for name in ledger.entities.names:
        q = ledger.position(name, symbol)
        pnl = q * float(delta_price)
        if abs(pnl) < _EPS:
            continue
        entries.append(Entry(name, cash, pnl))
    if not entries:
        return None
    return _post(
        ledger,
        Tx(tick, VARIATION_MARGIN_TAG, tuple(entries), memo=f"variation margin {symbol}"),
    )
