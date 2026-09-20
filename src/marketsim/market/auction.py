"""Uniform-price bond auctions (T6.26 / §6.11 primary market)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from marketsim.core.errors import ConfigError
from marketsim.ledger.journal import Entry, Ledger, Tx

# §6.11: Q_npc(y) = Q0 · exp(η_a · (y − y_fair) · 100). η_a is 1 / (100 bp).
ETA_A = 0.50
# Announcement lead (ticks) before the auction (§6.11).
ANNOUNCE_LEAD_TICKS = 5
# 1-based day of month for the auction (§6.11). tick = 1 simulated day.
AUCTION_DAY = 10
DAYS_PER_MONTH = 21


@dataclass(frozen=True)
class Bid:
    """Competitive bid. ``yield_`` is an annual decimal; ``qty`` is face (cr)."""

    agent_id: str
    yield_: float
    qty: float


@dataclass
class AuctionResult:
    """Uniform-price outcome. Yields are annual decimals; qtys are face (cr)."""

    stop_out: float
    size: float
    filled: float
    npc_fill: float
    agent_fill: dict[str, float]
    bid_to_cover: float
    tail: float  # stop-out minus pre-auction secondary yield (annual decimal)
    winning_bids: tuple[Bid, ...]

    def to_state(self) -> dict[str, Any]:
        return {
            "stop_out": self.stop_out,
            "size": self.size,
            "filled": self.filled,
            "npc_fill": self.npc_fill,
            "agent_fill": dict(self.agent_fill),
            "bid_to_cover": self.bid_to_cover,
            "tail": self.tail,
            "winning_bids": [
                {"agent_id": b.agent_id, "yield": b.yield_, "qty": b.qty} for b in self.winning_bids
            ],
        }


def npc_schedule(y: float, y_fair: float, q0: float, eta_a: float = ETA_A) -> float:
    """``Q_npc(y) = Q0 · exp(η_a · (y − y_fair) · 100)``. Face (cr); unbounded in y."""
    if q0 < 0:
        raise ConfigError("Q0 must be >= 0")
    return float(q0) * float(np.exp(float(eta_a) * (float(y) - float(y_fair)) * 100.0))


def auction_tick(month: int, *, days_per_month: int = DAYS_PER_MONTH, day: int = AUCTION_DAY) -> int:
    """Tick of the 1-based auction day in 0-based ``month``."""
    return int(month) * int(days_per_month) + (int(day) - 1)


def announce_tick(month: int, *, lead: int = ANNOUNCE_LEAD_TICKS, **kwargs: Any) -> int:
    """Announcement tick (``auction_tick − lead``)."""
    return auction_tick(month, **kwargs) - int(lead)


def clear_uniform(
    size: float,
    bids: tuple[Bid, ...] | list[Bid],
    *,
    y_fair: float,
    q0: float,
    eta_a: float = ETA_A,
    secondary_yield: float | None = None,
) -> AuctionResult:
    """Find the stop-out yield that clears ``size`` of face.

    Competitive bids are accepted from the lowest yield up. Uniform price: every
    winner (and the NPC residual) gets the stop-out yield. The NPC schedule is
    unbounded, so the auction always fills.
    """
    if size <= 0:
        raise ConfigError("auction size must be > 0")
    if q0 <= 0:
        raise ConfigError("NPC Q0 must be > 0 so the schedule can clear")
    ordered = tuple(sorted(bids, key=lambda b: (b.yield_, b.agent_id)))
    y_sec = float(y_fair if secondary_yield is None else secondary_yield)

    def total_at(y: float) -> float:
        agent = sum(b.qty for b in ordered if b.yield_ <= y)
        return agent + npc_schedule(y, y_fair, q0, eta_a)

    # Candidate yields: fair, each bid, and the closed-form NPC-only solve.
    y_npc = y_fair + np.log(size / q0) / (eta_a * 100.0)
    candidates = sorted({float(y_fair), float(y_npc), *[b.yield_ for b in ordered]})
    stop = None
    for y in candidates:
        if total_at(y) + 1e-12 >= size:
            stop = y
            break
    if stop is None:
        # Unbounded NPC: raise y until Q_npc covers whatever agents miss.
        stop = float(y_npc) if total_at(float(y_npc)) >= size else float(y_npc) + 0.01
        while total_at(stop) < size:
            stop += 0.01
    winners = tuple(b for b in ordered if b.yield_ <= stop)
    # Pro-rata the last yield group if agents oversubscribe.
    agent_fill: dict[str, float] = {}
    remaining = size
    by_y: dict[float, list[Bid]] = {}
    for b in winners:
        by_y.setdefault(b.yield_, []).append(b)
    for y in sorted(by_y):
        group = by_y[y]
        group_qty = sum(b.qty for b in group)
        take = min(remaining, group_qty)
        if group_qty <= remaining:
            for b in group:
                agent_fill[b.agent_id] = agent_fill.get(b.agent_id, 0.0) + b.qty
            remaining -= group_qty
        else:
            for b in group:
                agent_fill[b.agent_id] = agent_fill.get(b.agent_id, 0.0) + take * (b.qty / group_qty)
            remaining = 0.0
            break
    agent_amt = sum(agent_fill.values())
    npc_fill = max(size - agent_amt, 0.0)
    # Bid-to-cover uses the advertised book at y_fair (not the residual at stop).
    cover = (sum(b.qty for b in ordered) + npc_schedule(y_fair, y_fair, q0, eta_a)) / size
    return AuctionResult(
        stop_out=float(stop),
        size=float(size),
        filled=float(size),
        npc_fill=float(npc_fill),
        agent_fill=agent_fill,
        bid_to_cover=float(cover),
        tail=float(stop - y_sec),
        winning_bids=winners,
    )


def settle_auction(
    ledger: Ledger,
    result: AuctionResult,
    *,
    instrument: str,
    tick: int,
    npc_entity: str = "MM",
    price: float = 1.0,
) -> None:
    """Cash-for-bonds at ``price`` per unit face. Tag ``bond_issue``. SFC-safe."""
    entries: list[Entry] = []
    allocations = dict(result.agent_fill)
    if result.npc_fill > 1e-15:
        allocations[npc_entity] = allocations.get(npc_entity, 0.0) + result.npc_fill
    for entity, face in sorted(allocations.items()):
        cash = float(face) * float(price)
        if abs(cash) < 1e-15:
            continue
        entries.extend(
            (
                Entry(entity, instrument, float(face)),
                Entry("GOVT", instrument, -float(face)),
                Entry(entity, "DEP", -cash),
                Entry("GOVT", "DEP", cash),
            )
        )
    if entries:
        ledger.post(Tx(tick, "bond_issue", entries, memo=f"auction {instrument}"))


def buyback(
    ledger: Ledger,
    *,
    instrument: str,
    face: float,
    tick: int,
    holder: str = "MM",
    price: float = 1.0,
) -> None:
    """GOVT retires ``face`` (cr) from ``holder`` at ``price``. Tag ``bond_redeem``.

    Mirror of :func:`settle_auction`: cash leaves ``GOVT`` deposits, the instrument
    sums to zero. Caller must fund ``GOVT`` deposits; we never clip a shortfall.
    """
    qty = float(face)
    if qty <= 0:
        raise ConfigError("buyback face must be > 0")
    cash = qty * float(price)
    ledger.post(
        Tx(
            tick,
            "bond_redeem",
            (
                Entry(holder, instrument, -qty),
                Entry("GOVT", instrument, qty),
                Entry(holder, "DEP", cash),
                Entry("GOVT", "DEP", -cash),
            ),
            memo=f"buyback {instrument}",
        )
    )
