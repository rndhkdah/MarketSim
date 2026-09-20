"""Tender offers, dual-class stock, defences, mergers (T9.04 / §8.4).

Consideration is ``shares × price`` (cr). Transfers post on the ledger
(``equity_trade``). Control still follows T6.16 (> 50 % of **voting** shares at
the next strictly-later month-end) — this module does not move ``firm.operator``.
Quantity limits (pills, share caps) never change the common borrowing rate (D14).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from marketsim.equity.captable import CapTable
from marketsim.equity.control import CONTROL_THRESHOLD
from marketsim.equity.corporate_actions import buyback_shares, issue_shares
from marketsim.firms.accounts import agent_entity
from marketsim.ledger.journal import Ledger
from marketsim.ledger.sfc import assert_consistent

TENDER_TAG = "equity_trade"
ShareClass = Literal["voting", "nonvoting"]

# §6.7 / T9.04: typical US pill trigger (dimensionless voting share).
PILL_TRIGGER = 0.15
# New voting shares issued to each non-bidder holder per share they already hold.
PILL_RATIO = 1.0
# Extra month-ends a staggered board inserts before a queued operator change.
STAGGERED_DELAY_M = 1


def _entity(name: str) -> str:
    return agent_entity(name) if ":" not in name else name


@dataclass(frozen=True)
class TenderOffer:
    """One cash tender. ``shares`` are units; ``price`` is cr / share."""

    firm_id: str
    bidder: str
    target_holder: str
    shares: float
    price: float


def consideration(shares: float, price: float) -> float:
    """Cash due (cr). ``shares`` are units; ``price`` is cr / share."""
    return float(shares) * float(price)


def apply_tender(ledger: Ledger, table: CapTable, offer: TenderOffer, *, tick: int) -> None:
    """Move ``shares`` from ``target_holder`` to ``bidder`` for cash. SFC-green.

    ``tick`` is days. Holder names are ledger entities (``AGENT:<id>``).
    """
    buyer = _entity(offer.bidder)
    seller = _entity(offer.target_holder)
    if buyer not in ledger.entities:
        ledger.register_entity(buyer)
    table.transfer(
        ledger,
        src=seller,
        dst=buyer,
        shares=float(offer.shares),
        tick=int(tick),
        price=float(offer.price),
    )
    assert_consistent(ledger)


def crossing_control(bidder_share: float) -> bool:
    """True when the bidder's **voting** share exceeds T6.16 ``CONTROL_THRESHOLD``."""
    return float(bidder_share) > float(CONTROL_THRESHOLD)


@dataclass
class DualClassBook:
    """Voting vs non-voting register. Control uses voting shares only.

    ``voting`` / ``nonvoting`` map holder entity → shares. Insertion is not
    used for control; holders are sorted when iterating.
    """

    voting: dict[str, float] = field(default_factory=dict)
    nonvoting: dict[str, float] = field(default_factory=dict)

    def grant(self, holder: str, shares: float, klass: ShareClass = "voting") -> None:
        """Add ``shares`` of ``klass`` to ``holder``."""
        book = self.voting if klass == "voting" else self.nonvoting
        name = _entity(holder)
        book[name] = book.get(name, 0.0) + float(shares)

    def outstanding(self, klass: ShareClass = "voting") -> float:
        """Σ shares of ``klass`` (units)."""
        book = self.voting if klass == "voting" else self.nonvoting
        return float(sum(max(v, 0.0) for v in book.values()))

    def holding(self, holder: str, klass: ShareClass = "voting") -> float:
        """Shares of ``klass`` held by ``holder`` (units)."""
        book = self.voting if klass == "voting" else self.nonvoting
        return float(book.get(_entity(holder), 0.0))

    def voting_ownership(self, holder: str) -> float:
        """``voting held / voting outstanding`` (dimensionless). 0 if none out."""
        tot = self.outstanding("voting")
        if tot <= 0.0:
            return 0.0
        return self.holding(holder, "voting") / tot

    def economic_ownership(self, holder: str) -> float:
        """``(voting + nonvoting) / total`` (dimensionless)."""
        tot = self.outstanding("voting") + self.outstanding("nonvoting")
        if tot <= 0.0:
            return 0.0
        return (self.holding(holder, "voting") + self.holding(holder, "nonvoting")) / tot

    def to_state(self) -> dict[str, Any]:
        return {
            "voting": {k: self.voting[k] for k in sorted(self.voting)},
            "nonvoting": {k: self.nonvoting[k] for k in sorted(self.nonvoting)},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> DualClassBook:
        return cls(
            voting={str(k): float(v) for k, v in (state.get("voting") or {}).items()},
            nonvoting={str(k): float(v) for k, v in (state.get("nonvoting") or {}).items()},
        )


@dataclass(frozen=True)
class Defence:
    """Quantity defences. None of these change a loan or bond price (D14).

    ``pill_trigger`` is a dimensionless voting-share tripwire.
    ``pill_ratio`` is new voting shares per existing share issued to others.
    ``staggered_board`` inserts ``STAGGERED_DELAY_M`` extra month-ends.
    """

    poison_pill: bool = False
    pill_trigger: float = PILL_TRIGGER
    pill_ratio: float = PILL_RATIO
    staggered_board: bool = False


def control_delay_m(defence: Defence) -> int:
    """Extra month-ends before a queued operator change. Months."""
    return STAGGERED_DELAY_M if defence.staggered_board else 0


def would_trip_pill(pre_share: float, incoming_share: float, defence: Defence) -> bool:
    """True when the tender would take voting ownership above ``pill_trigger``."""
    if not defence.poison_pill:
        return False
    return float(pre_share) + float(incoming_share) > float(defence.pill_trigger) + 1e-12


def apply_poison_pill(
    ledger: Ledger,
    table: CapTable,
    *,
    bidder: str,
    defence: Defence,
    tick: int,
) -> float:
    """Issue ``pill_ratio`` new shares to every non-bidder holder. Returns shares issued.

    ``tick`` is days. Price is 0 cr/share (rights, no cash). SFC-checked.
    """
    if not defence.poison_pill:
        return 0.0
    buyer = _entity(bidder)
    issued = 0.0
    holders = [(h, q) for h, q in table.holdings(ledger).items() if h != buyer and q > 0.0]
    holders.sort(key=lambda row: row[0])
    for holder, qty in holders:
        extra = float(qty) * float(defence.pill_ratio)
        if extra <= 0.0:
            continue
        issue_shares(table, ledger, subscriber=holder, shares=extra, price=0.0, tick=int(tick))
        issued += extra
    assert_consistent(ledger)
    return issued


def apply_tender_with_defence(
    ledger: Ledger,
    table: CapTable,
    offer: TenderOffer,
    defence: Defence,
    *,
    tick: int,
) -> bool:
    """Run the tender; fire the pill first if it would trip. Returns True if transferred.

    ``tick`` is days. A staggered board does not block the cash transfer — it
    only lengthens :func:`control_delay_m`.
    """
    buyer = _entity(offer.bidder)
    out = table.shares_outstanding(ledger)
    incoming = float(offer.shares) / out if out > 0.0 else 0.0
    pre = table.ownership(ledger, buyer)
    if would_trip_pill(pre, incoming, defence):
        apply_poison_pill(ledger, table, bidder=offer.bidder, defence=defence, tick=tick)
    apply_tender(ledger, table, offer, tick=tick)
    return True


def merge_firms(
    ledger: Ledger,
    survivor: CapTable,
    target: CapTable,
    *,
    exchange_ratio: float,
    tick: int,
) -> dict[str, float]:
    """Stock-for-stock merger. Target holders receive ``ratio ×`` survivor shares.

    Target shares are bought back at 0 cr and cancelled. New survivor shares are
    issued at 0 cr (no cash, no common-rate change). ``tick`` is days.
    Returns ``{holder: new_survivor_shares}`` in entity order. SFC-checked.
    """
    ratio = float(exchange_ratio)
    if ratio <= 0.0:
        raise ValueError("exchange_ratio must be > 0")
    issued: dict[str, float] = {}
    holders = list(target.holdings(ledger).items())
    holders.sort(key=lambda row: row[0])
    for holder, qty in holders:
        new = float(qty) * ratio
        if new <= 0.0:
            continue
        if holder not in ledger.entities:
            ledger.register_entity(holder)
        issue_shares(survivor, ledger, subscriber=holder, shares=new, price=0.0, tick=int(tick))
        buyback_shares(target, ledger, seller=holder, shares=float(qty), price=0.0, tick=int(tick))
        issued[holder] = new
    assert_consistent(ledger)
    return issued
