"""IPO / listing via a CLOB call auction with a reserve (T6.15 / §6.7).

Primary issuance creates new shares (cash to the firm). A secondary sale moves
the founder's shares (cash to the founder). The float is the cleared volume.
A primary block that would leave the founder with ≤ 50 % of post-issue SO
warns and does not block. Units: ``shares`` are shares; ``reserve`` / prices
are cr/share; cash legs are cr.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from marketsim.core.config import Config
from marketsim.core.errors import ConfigError, LedgerError
from marketsim.equity.captable import _EPS, CASH, CapTable
from marketsim.equity.corporate_actions import issue_shares
from marketsim.firms.accounts import firm_entity, register_firm
from marketsim.ledger.journal import Ledger
from marketsim.market.clob import (
    CLOB,
    AuctionResult,
    BookStatus,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    price_to_ticks,
)
from marketsim.market.instruments import InstrumentRegistry, firm_symbol

ListingKind = Literal["primary", "secondary"]
CashTo = Literal["firm", "founder"]

# §6.7: warn when issued / (SO + issued) > 0.49 (founder would then hold ≤ 50 %).
# 0.50 is the control line; the engine warns and does not block.
_WARN_FLOAT_SHARE = 0.49  # §6.7


def offering_account(firm_id: str) -> str:
    """Synthetic CLOB seller for a primary issue. Unitless id; not a ledger holder."""
    return f"LISTING:{firm_id}"


def warning_49(offered: int, outstanding: float) -> bool:
    """True when ``offered / (SO + offered) > 0.49`` (shares; §6.7)."""
    post = float(outstanding) + float(offered)
    if post <= _EPS:
        return False
    return (float(offered) / post) > _WARN_FLOAT_SHARE


@dataclass(frozen=True)
class ListingResult:
    """Outcome of a listing auction.

    ``price`` is cr/share (None if cancelled). ``volume`` is shares.
    ``cash_to`` is ``\"firm\"`` (primary) or ``\"founder\"`` (secondary).
    ``fills`` are the settled CLOB prints (empty when the reserve is missed).
    """

    price: float | None
    volume: int
    cash_to: CashTo
    warning_49: bool
    fills: tuple[Fill, ...]


@dataclass
class Listing:
    """One primary or secondary offering. ``shares`` are shares; ``reserve`` is cr/share."""

    kind: ListingKind
    shares: int
    reserve: float
    firm_id: str = ""
    founder: str = ""
    tick: int = 0
    seller: str = ""
    offer_id: int | None = None

    def to_state(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "shares": int(self.shares),
            "reserve": float(self.reserve),
            "firm_id": self.firm_id,
            "founder": self.founder,
            "tick": int(self.tick),
            "seller": self.seller,
            "offer_id": self.offer_id,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Listing:
        offer = state.get("offer_id")
        return cls(
            kind=state["kind"],
            shares=int(state["shares"]),
            reserve=float(state["reserve"]),
            firm_id=str(state.get("firm_id", "")),
            founder=str(state.get("founder", "")),
            tick=int(state.get("tick", 0)),
            seller=str(state.get("seller", "")),
            offer_id=None if offer is None else int(offer),
        )

    def run(
        self,
        table: CapTable,
        ledger: Ledger,
        clob: CLOB,
        cfg: Config,
        *,
        bids: Sequence[Order] = (),
        registry: InstrumentRegistry | None = None,
        cash: str = CASH,
    ) -> ListingResult:
        """Post the offering, take ``bids``, uncross, settle. See :func:`list_firm`."""
        self.prepare(table, ledger, clob, cfg, registry=registry)
        symbol = firm_symbol(table.firm_id)
        for raw in bids:
            order = raw if raw.symbol else replace(raw, symbol=symbol)
            clob.submit(order)
        return self.settle(table, ledger, clob, cash=cash)

    def prepare(
        self,
        table: CapTable,
        ledger: Ledger,
        clob: CLOB,
        cfg: Config,
        *,
        registry: InstrumentRegistry | None = None,
    ) -> Listing:
        """Ensure the CLOB book is in auction mode and post the reserve sell (shares)."""
        _bind(self, table)
        lot, tick_sz = _clob_lot_tick(cfg)
        offered = _check_shares(self.shares, lot)
        reserve = _check_reserve(self.reserve, tick_sz)
        self.shares = offered
        self.reserve = reserve
        register_firm(ledger, table.firm_id)
        if table.issuer != firm_entity(table.firm_id):
            raise LedgerError("cap-table issuer does not match FIRM:<id>")
        reg = registry if registry is not None else InstrumentRegistry()
        reg.list_firm(table.firm_id)
        symbol = firm_symbol(table.firm_id)
        _ensure_book(clob, symbol)
        if self.kind == "secondary":
            held = table.holding(ledger, table.founder)
            if held + _EPS < float(offered):
                raise LedgerError(f"{table.founder} holds {held} < {offered} shares")
            self.seller = self.seller or table.founder
        else:
            self.seller = self.seller or offering_account(table.firm_id)
        posted = clob.submit(
            Order(
                agent_id=self.seller,
                side=Side.SELL,
                qty=offered,
                symbol=symbol,
                order_type=OrderType.LIMIT,
                tif=TimeInForce.IOC,
                price=reserve,
            )
        )
        if posted.order_id is None or posted.status is OrderStatus.REJECTED:
            raise LedgerError(posted.reason or "offering order rejected")
        self.offer_id = posted.order_id
        return self

    def settle(
        self,
        table: CapTable,
        ledger: Ledger,
        clob: CLOB,
        *,
        cash: str = CASH,
    ) -> ListingResult:
        """Uncross and post cash/shares, or cancel if the reserve is missed.

        Clearing ``price`` is cr/share. ``volume`` is shares. Cash notional is cr.
        """
        cash_to: CashTo = "firm" if self.kind == "primary" else "founder"
        outstanding = table.shares_outstanding(ledger)
        warn = self.kind == "primary" and warning_49(int(self.shares), outstanding)
        symbol = firm_symbol(table.firm_id)
        auction: AuctionResult = clob.uncross(symbol)[0]
        if auction.price is None or auction.price < float(self.reserve) or auction.volume <= 0:
            _cancel_offer(clob, symbol, self.offer_id)
            return ListingResult(price=None, volume=0, cash_to=cash_to, warning_49=warn, fills=())
        price = float(auction.price)
        fills = tuple(f for f in auction.fills if self.seller in (f.maker, f.taker))
        if self.kind == "primary":
            _settle_primary(table, ledger, fills, price, self.seller, self.tick, cash)
        else:
            _settle_secondary(table, ledger, fills, price, self.seller, self.tick, cash)
        volume = int(sum(f.qty for f in fills))
        return ListingResult(price=price, volume=volume, cash_to=cash_to, warning_49=warn, fills=fills)


def list_firm(
    table: CapTable,
    ledger: Ledger,
    clob: CLOB,
    cfg: Config,
    *,
    kind: ListingKind,
    shares: int,
    reserve: float,
    bids: Sequence[Order] = (),
    registry: InstrumentRegistry | None = None,
    tick: int = 0,
    cash: str = CASH,
) -> ListingResult:
    """List ``EQ:FIRM:<id>`` through a call auction with a reserve.

    ``shares`` is the offered block (shares, integer lots from ``cfg.markets.clob``).
    ``reserve`` is the minimum clearing price (cr/share). Primary cash (cr) lands
    in ``FIRM:<id>`` via :func:`issue_shares`; secondary cash lands in the founder
    via :meth:`CapTable.transfer`. The float is the cleared volume. Warns — does
    not block — when a primary block satisfies ``issued / (SO + issued) > 0.49``
    (§6.7). ``bids`` are agent buy orders (funds permitting).
    """
    if kind not in ("primary", "secondary"):
        raise LedgerError(f"listing kind must be 'primary' or 'secondary', got {kind!r}")
    listing = Listing(
        kind=kind,
        shares=int(shares),
        reserve=float(reserve),
        firm_id=table.firm_id,
        founder=table.founder,
        tick=int(tick),
    )
    return listing.run(table, ledger, clob, cfg, bids=bids, registry=registry, cash=cash)


def _bind(listing: Listing, table: CapTable) -> None:
    listing.firm_id = listing.firm_id or table.firm_id
    listing.founder = listing.founder or table.founder
    if listing.kind not in ("primary", "secondary"):
        raise LedgerError(f"listing kind must be 'primary' or 'secondary', got {listing.kind!r}")


def _clob_lot_tick(cfg: Config) -> tuple[int, float]:
    if cfg.markets is None:
        raise ConfigError("listing requires config.markets (markets.yaml clob lot/tick)")
    return int(cfg.markets.clob.lot), float(cfg.markets.clob.tick)


def _check_shares(shares: int, lot: int) -> int:
    offered = int(shares)
    if offered != shares or offered < 1:
        raise LedgerError("offered shares must be a positive integer (shares)")
    if offered % lot != 0:
        raise LedgerError(f"offered shares must be a multiple of lot {lot} (shares)")
    return offered


def _check_reserve(reserve: float, tick: float) -> float:
    px = float(reserve)
    if px <= 0.0:
        raise LedgerError("reserve must be > 0 cr/share")
    try:
        price_to_ticks(px, tick)
    except ValueError as exc:
        raise LedgerError(str(exc)) from exc
    return px


def _ensure_book(clob: CLOB, symbol: str) -> None:
    if symbol in clob.symbols():
        book = clob.book(symbol)
        if book.status is not BookStatus.AUCTION:
            book.start_auction()
        return
    clob.add_book(symbol, reference_price=None)


def _cancel_offer(clob: CLOB, symbol: str, offer_id: int | None) -> None:
    if offer_id is None:
        return
    book = clob.book(symbol)
    if offer_id in book.resting_ids():
        book.cancel(offer_id)


def _counterparty(fill: Fill, seller: str) -> str:
    if fill.maker == seller:
        return fill.taker
    if fill.taker == seller:
        return fill.maker
    raise LedgerError(f"fill {fill.maker}/{fill.taker} does not include seller {seller}")


def _settle_primary(
    table: CapTable,
    ledger: Ledger,
    fills: tuple[Fill, ...],
    price: float,
    seller: str,
    tick: int,
    cash: str,
) -> None:
    """Issue filled shares at ``price`` (cr/share); cash (cr) to ``FIRM:<id>``."""
    for fill in fills:
        buyer = _counterparty(fill, seller)
        issue_shares(
            table,
            ledger,
            subscriber=buyer,
            shares=float(fill.qty),
            price=price,
            tick=tick,
            cash=cash,
        )


def _settle_secondary(
    table: CapTable,
    ledger: Ledger,
    fills: tuple[Fill, ...],
    price: float,
    seller: str,
    tick: int,
    cash: str,
) -> None:
    """Transfer filled founder shares at ``price`` (cr/share); cash (cr) to founder."""
    for fill in fills:
        buyer = _counterparty(fill, seller)
        table.transfer(
            ledger,
            src=table.founder,
            dst=buyer,
            shares=float(fill.qty),
            price=price,
            tick=tick,
            cash=cash,
        )
