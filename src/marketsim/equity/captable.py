"""Share register for agent-firm equity (T6.14 / §6.7).

The founder holds 100 % at opening (default 1,000,000 shares). Long holdings are
the positive ``EQ:FIRM:<id>`` positions; the issuer (``FIRM:<id>``) carries the
matching negative so the instrument sums to 0. Units: shares, except cash legs
in cr. Ledger is the position book; the table snapshots holdings by tick so
dividends can pay the record-date register.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marketsim.core.errors import LedgerError
from marketsim.firms.accounts import firm_entity, firm_equity_instrument, register_firm
from marketsim.ledger.journal import Entry, Ledger, Tx

# §6.7: founder holds 100 % (default 1,000,000 shares).
DEFAULT_FOUNDER_SHARES = 1_000_000
CASH = "DEP"
TRADE_TAG = "equity_trade"
ISSUE_TAG = "equity_issue"
# Dust; same floor as settlement / journal (|sum| > 1e-9 is unbalanced).
_EPS = 1e-15
_ATOL = 1e-8


def equity_symbol(firm_id: str) -> str:
    """``EQ:FIRM:<id>`` share instrument. Unit: shares."""
    return firm_equity_instrument(firm_id)


def _positive_shares(shares: float, *, what: str) -> float:
    qty = float(shares)
    if qty <= _EPS:
        raise LedgerError(f"{what} must be > 0 shares")
    return qty


def _holdings_from_ledger(ledger: Ledger, issuer: str, symbol: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ledger.entities.names:
        if name == issuer:
            continue
        qty = ledger.position(name, symbol)
        if qty > _EPS:
            out[name] = qty
    return out


@dataclass
class CapTable:
    """Per-firm share register. Positions live on the ledger (shares)."""

    firm_id: str
    founder: str
    _history: list[tuple[int, dict[str, float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.firm_id = str(self.firm_id)
        self.founder = str(self.founder)

    @property
    def issuer(self) -> str:
        """Ledger entity that carries −shares outstanding. Unitless id."""
        return firm_entity(self.firm_id)

    @property
    def symbol(self) -> str:
        """``EQ:FIRM:<id>``. Unit: shares."""
        return equity_symbol(self.firm_id)

    @classmethod
    def open(
        cls,
        ledger: Ledger,
        *,
        firm_id: str,
        founder: str,
        shares: float = DEFAULT_FOUNDER_SHARES,
        tick: int = 0,
    ) -> CapTable:
        """Issue the founder stake. ``shares`` is shares; no cash leg.

        Posts ``equity_issue``: founder +S, issuer −S. Default ``shares`` is
        ``DEFAULT_FOUNDER_SHARES`` (1,000,000; §6.7).
        """
        qty = _positive_shares(shares, what="founder shares")
        register_firm(ledger, firm_id)
        if founder not in ledger.entities:
            ledger.register_entity(founder)
        table = cls(firm_id=firm_id, founder=founder)
        if abs(ledger.position(table.issuer, table.symbol)) > _EPS:
            raise LedgerError(f"cap table already open for {firm_id}")
        ledger.post(
            Tx(
                int(tick),
                ISSUE_TAG,
                (
                    Entry(founder, table.symbol, qty),
                    Entry(table.issuer, table.symbol, -qty),
                ),
                memo=f"found {table.symbol}",
            )
        )
        table.remember(ledger, int(tick))
        return table

    def shares_outstanding(self, ledger: Ledger) -> float:
        """−issuer position (shares). Cancelled buybacks shrink this."""
        return -ledger.position(self.issuer, self.symbol)

    def holding(self, ledger: Ledger, entity: str) -> float:
        """Long shares of ``entity``, or 0. Issuer is never a holder."""
        if entity == self.issuer or entity not in ledger.entities:
            return 0.0
        qty = ledger.position(entity, self.symbol)
        return qty if qty > _EPS else 0.0

    def holdings(self, ledger: Ledger) -> dict[str, float]:
        """Positive holders → shares. Keys in entity-registry order."""
        return _holdings_from_ledger(ledger, self.issuer, self.symbol)

    def total_holdings(self, ledger: Ledger) -> float:
        """Σ long holdings (shares). Equals outstanding when the issuer is the only short."""
        return float(sum(self.holdings(ledger).values()))

    def ownership(self, ledger: Ledger, entity: str) -> float:
        """``holding / outstanding`` (dimensionless). 0 if nothing is outstanding."""
        outstanding = self.shares_outstanding(ledger)
        if outstanding <= _EPS:
            return 0.0
        return self.holding(ledger, entity) / outstanding

    def remember(self, ledger: Ledger, tick: int) -> None:
        """Snapshot longs after a mutation. ``tick`` is the simulated day."""
        snap = self.holdings(ledger)
        tick_i = int(tick)
        for i, (t, _) in enumerate(self._history):
            if t == tick_i:
                self._history[i] = (tick_i, snap)
                return
            if t > tick_i:
                self._history.insert(i, (tick_i, snap))
                return
        self._history.append((tick_i, snap))

    def holdings_on(self, tick: int, ledger: Ledger) -> dict[str, float]:
        """Register as of ``tick`` (last snapshot with t ≤ tick), else live holdings."""
        chosen: dict[str, float] | None = None
        for t, snap in self._history:
            if t <= int(tick):
                chosen = snap
            else:
                break
        if chosen is None:
            return self.holdings(ledger)
        return dict(chosen)

    def invariant_holds(self, ledger: Ledger, *, atol: float = _ATOL) -> bool:
        """Σ longs = outstanding, issuer = −outstanding, instrument sums to 0."""
        symbol = self.symbol
        issuer = self.issuer
        outstanding = self.shares_outstanding(ledger)
        longs = self.total_holdings(ledger)
        if abs(longs - outstanding) > atol:
            return False
        if abs(ledger.position(issuer, symbol) + outstanding) > atol:
            return False
        inst = 0.0
        for name in ledger.entities.names:
            inst += ledger.position(name, symbol)
        return abs(inst) <= atol

    def transfer(
        self,
        ledger: Ledger,
        *,
        src: str,
        dst: str,
        shares: float,
        tick: int,
        price: float | None = None,
        cash: str = CASH,
    ) -> Tx:
        """Move ``shares`` from ``src`` to ``dst``. Optional ``price`` is cr/share.

        Secondary: cash goes to the seller, not the firm. Outstanding unchanged.
        Tag ``equity_trade``. Issuer is not a party — use issue / buyback.
        """
        qty = _positive_shares(shares, what="transfer")
        if src == dst:
            raise LedgerError("transfer src and dst must differ")
        if src == self.issuer or dst == self.issuer:
            raise LedgerError("issuer is not a transfer party; use issue_shares or buyback_shares")
        if src not in ledger.entities:
            raise LedgerError(f"unknown holder {src}")
        if self.holding(ledger, src) + _EPS < qty:
            raise LedgerError(f"{src} holds {self.holding(ledger, src)} < {qty} shares")
        if dst not in ledger.entities:
            ledger.register_entity(dst)
        entries: list[Entry] = [
            Entry(src, self.symbol, -qty),
            Entry(dst, self.symbol, qty),
        ]
        if price is not None:
            notional = qty * float(price)
            if abs(notional) >= _EPS:
                entries.extend((Entry(dst, cash, -notional), Entry(src, cash, notional)))
        tx = Tx(int(tick), TRADE_TAG, tuple(entries), memo=self.symbol)
        ledger.post(tx)
        self.remember(ledger, int(tick))
        return tx

    def to_state(self) -> dict[str, Any]:
        history = [[t, {k: snap[k] for k in sorted(snap)}] for t, snap in self._history]
        return {"firm_id": self.firm_id, "founder": self.founder, "history": history}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CapTable:
        table = cls(firm_id=state["firm_id"], founder=state["founder"])
        history: list[tuple[int, dict[str, float]]] = []
        for row in state.get("history", ()):
            tick, snap = row[0], row[1]
            history.append((int(tick), {str(k): float(v) for k, v in snap.items()}))
        table._history = history
        return table
