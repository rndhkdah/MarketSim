"""Dividends, primary issuance and buybacks (T6.14 / §6.7).

Record date = declaration + 2 ticks. Issuance dilutes (new shares, cash to the
firm). Buybacks are cash-for-shares back to the issuer, which cancels them.
Every post is a balanced ``Tx``; the issuer keeps the negative ``EQ`` so the
instrument sums to 0. Units: shares; cash legs in cr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marketsim.core.errors import LedgerError
from marketsim.equity.captable import _EPS, CASH, CapTable, _positive_shares
from marketsim.ledger.journal import Entry, Ledger, Tx

# §6.7: record date = declaration + 2 ticks.
RECORD_LAG_TICKS = 2
DIVIDEND_TAG = "dividends"
ISSUE_TAG = "equity_issue"
TRADE_TAG = "equity_trade"


def record_date(declared_tick: int) -> int:
    """Record tick (days) = ``declared_tick`` + ``RECORD_LAG_TICKS`` (§6.7)."""
    return int(declared_tick) + RECORD_LAG_TICKS


@dataclass
class Dividend:
    """One declared cash dividend. ``amount`` is cr (total), ticks are days."""

    firm_id: str
    amount: float
    declared_tick: int
    record_tick: int
    paid_tick: int | None = None
    record_holdings: dict[str, float] = field(default_factory=dict)

    def to_state(self) -> dict[str, Any]:
        return {
            "firm_id": self.firm_id,
            "amount": float(self.amount),
            "declared_tick": int(self.declared_tick),
            "record_tick": int(self.record_tick),
            "paid_tick": self.paid_tick,
            "record_holdings": {k: self.record_holdings[k] for k in sorted(self.record_holdings)},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Dividend:
        paid = state.get("paid_tick")
        return cls(
            firm_id=str(state["firm_id"]),
            amount=float(state["amount"]),
            declared_tick=int(state["declared_tick"]),
            record_tick=int(state["record_tick"]),
            paid_tick=None if paid is None else int(paid),
            record_holdings={str(k): float(v) for k, v in (state.get("record_holdings") or {}).items()},
        )


def issue_shares(
    table: CapTable,
    ledger: Ledger,
    *,
    subscriber: str,
    shares: float,
    price: float,
    tick: int,
    cash: str = CASH,
) -> Tx:
    """Primary issue: ``shares`` new units at ``price`` cr/share; cash to the firm.

    Dilutes existing longs. Tag ``equity_issue``. Subscriber +q, issuer −q.
    """
    qty = _positive_shares(shares, what="issuance")
    px = float(price)
    if px < 0.0:
        raise LedgerError("issue price must be >= 0 cr/share")
    if subscriber == table.issuer:
        raise LedgerError("issuer cannot subscribe to its own issue")
    if subscriber not in ledger.entities:
        ledger.register_entity(subscriber)
    notional = qty * px
    entries: list[Entry] = [
        Entry(subscriber, table.symbol, qty),
        Entry(table.issuer, table.symbol, -qty),
    ]
    if abs(notional) >= _EPS:
        entries.extend(
            (
                Entry(subscriber, cash, -notional),
                Entry(table.issuer, cash, notional),
            )
        )
    tx = Tx(int(tick), ISSUE_TAG, tuple(entries), memo=f"issue {table.symbol}")
    ledger.post(tx)
    table.remember(ledger, int(tick))
    return tx


def buyback_shares(
    table: CapTable,
    ledger: Ledger,
    *,
    seller: str,
    shares: float,
    price: float,
    tick: int,
    cash: str = CASH,
) -> Tx:
    """Firm buys ``shares`` from ``seller`` at ``price`` cr/share and cancels them.

    Shares return to the issuer (negative EQ shrinks). Tag ``equity_trade``.
    Cash is never clipped.
    """
    qty = _positive_shares(shares, what="buyback")
    px = float(price)
    if px < 0.0:
        raise LedgerError("buyback price must be >= 0 cr/share")
    if seller == table.issuer:
        raise LedgerError("issuer cannot sell to itself")
    held = table.holding(ledger, seller)
    if held + _EPS < qty:
        raise LedgerError(f"{seller} holds {held} < {qty} shares")
    notional = qty * px
    entries: list[Entry] = [
        Entry(seller, table.symbol, -qty),
        Entry(table.issuer, table.symbol, qty),
        Entry(table.issuer, cash, -notional),
        Entry(seller, cash, notional),
    ]
    tx = Tx(int(tick), TRADE_TAG, tuple(entries), memo=f"buyback {table.symbol}")
    ledger.post(tx)
    table.remember(ledger, int(tick))
    return tx


def _pay_dividend(
    table: CapTable,
    ledger: Ledger,
    holdings: dict[str, float],
    amount: float,
    tick: int,
    *,
    cash: str,
) -> Tx | None:
    """Pro-rata ``amount`` (cr) to ``holdings`` (shares). Firm pays −amount."""
    total = float(sum(holdings.values()))
    cash_amt = float(amount)
    if total <= _EPS or abs(cash_amt) < _EPS:
        return None
    items = [(name, qty) for name, qty in holdings.items() if qty > _EPS]
    items.sort(key=lambda row: row[0])
    if not items:
        return None
    entries: list[Entry] = []
    allocated = 0.0
    last = len(items) - 1
    for i, (name, qty) in enumerate(items):
        if i == last:
            pay = cash_amt - allocated
        else:
            pay = cash_amt * (qty / total)
            allocated += pay
        entries.append(Entry(name, cash, pay))
    entries.append(Entry(table.issuer, cash, -cash_amt))
    tx = Tx(int(tick), DIVIDEND_TAG, tuple(entries), memo=f"{table.symbol} dividends")
    ledger.post(tx)
    return tx


@dataclass
class CorporateActions:
    """Pending dividend declarations. Issue / buyback are posted immediately."""

    declarations: list[Dividend] = field(default_factory=list)

    def declare_dividend(self, table: CapTable, *, amount: float, tick: int) -> Dividend:
        """Declare a cash dividend (cr). Record date is ``tick + 2`` (§6.7)."""
        cash_amt = float(amount)
        if abs(cash_amt) < _EPS:
            raise LedgerError("dividend amount must be non-zero")
        declared = Dividend(
            firm_id=table.firm_id,
            amount=cash_amt,
            declared_tick=int(tick),
            record_tick=record_date(tick),
        )
        self.declarations.append(declared)
        return declared

    def settle(self, table: CapTable, ledger: Ledger, tick: int, *, cash: str = CASH) -> tuple[Tx, ...]:
        """Pay unpaid dividends whose record date is ``tick`` or earlier.

        Holders are the cap-table snapshot on the record date (declaration + 2),
        not the live register after later trades.
        """
        posted: list[Tx] = []
        for declared in self.declarations:
            if declared.firm_id != table.firm_id:
                continue
            if declared.paid_tick is not None:
                continue
            if int(tick) < declared.record_tick:
                continue
            holders = table.holdings_on(declared.record_tick, ledger)
            declared.record_holdings = dict(holders)
            tx = _pay_dividend(table, ledger, holders, declared.amount, int(tick), cash=cash)
            declared.paid_tick = int(tick)
            if tx is not None:
                posted.append(tx)
        return tuple(posted)

    def issue(
        self,
        table: CapTable,
        ledger: Ledger,
        *,
        subscriber: str,
        shares: float,
        price: float,
        tick: int,
        cash: str = CASH,
    ) -> Tx:
        """See :func:`issue_shares`."""
        return issue_shares(
            table, ledger, subscriber=subscriber, shares=shares, price=price, tick=tick, cash=cash
        )

    def buyback(
        self,
        table: CapTable,
        ledger: Ledger,
        *,
        seller: str,
        shares: float,
        price: float,
        tick: int,
        cash: str = CASH,
    ) -> Tx:
        """See :func:`buyback_shares`."""
        return buyback_shares(
            table, ledger, seller=seller, shares=shares, price=price, tick=tick, cash=cash
        )

    def to_state(self) -> dict[str, Any]:
        return {"declarations": [d.to_state() for d in self.declarations]}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CorporateActions:
        return cls(declarations=[Dividend.from_state(row) for row in state.get("declarations", ())])
