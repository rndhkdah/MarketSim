"""Shortage allocation across firm buyers (T5.09 / §5.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.core.errors import LedgerError
from marketsim.firms.firm import FirmsFile
from marketsim.ledger.journal import Entry, Ledger, Tx


@dataclass
class Bid:
    """One buyer's order. ``order`` is real units; ``premium`` is dimensionless."""

    buyer_id: str
    order: float
    premium: float = 0.0
    relationship: float = 0.0  # 12-month EMA purchase share


@dataclass
class Allocation:
    fill: dict[str, float]
    premium_paid: dict[str, float]


def allocate_shortage(
    bids: list[Bid],
    supply: float,
    cfg: FirmsFile,
    *,
    shortage: bool | None = None,
) -> Allocation:
    """No shortage → pro-rata (premiums unused). Shortage → bid and relationship weights."""
    orders = np.array([max(b.order, 0.0) for b in bids], dtype=float)
    tot = float(orders.sum())
    ids = [b.buyer_id for b in bids]
    if tot <= 0.0 or supply <= 0.0:
        return Allocation({i: 0.0 for i in ids}, {i: 0.0 for i in ids})
    cap = cfg.procurement.order_size_cap_of_cell * supply
    orders = np.minimum(orders, cap)
    tot = float(orders.sum())
    is_short = (tot > supply) if shortage is None else shortage
    if not is_short:
        scale = min(1.0, supply / tot) if tot > 0 else 0.0
        fill = {i: float(o * scale) for i, o in zip(ids, orders, strict=True)}
        return Allocation(fill, {i: 0.0 for i in ids})
    w = []
    for b, o in zip(bids, orders, strict=True):
        prem = min(max(b.premium, 0.0), cfg.procurement.premium_max)
        w.append(o * np.exp(cfg.procurement.beta_premium * prem) * (1.0 + cfg.procurement.rho_relationship * b.relationship))
    w = np.asarray(w, dtype=float)
    if float(w.sum()) <= 0:
        w = orders
    fill_v = supply * w / float(w.sum())
    fill_v = np.minimum(fill_v, orders)
    # water-fill leftover
    leftover = supply - float(fill_v.sum())
    room = orders - fill_v
    if leftover > 1e-15 and float(room.sum()) > 0:
        fill_v = fill_v + leftover * room / float(room.sum())
        fill_v = np.minimum(fill_v, orders)
    fill = {i: float(x) for i, x in zip(ids, fill_v, strict=True)}
    paid = {b.buyer_id: fill[b.buyer_id] * min(max(b.premium, 0.0), cfg.procurement.premium_max) for b in bids}
    return Allocation(fill, paid)


def post_premiums(ledger: Ledger, alloc: Allocation, *, seller: str, tick: int) -> None:
    """Premium is a tagged fee posting (cr)."""
    for buyer, amt in sorted(alloc.premium_paid.items()):
        if amt <= 0:
            continue
        if buyer not in ledger.entities:
            ledger.register_entity(buyer)
        ledger.post(
            Tx(tick, "fees", (Entry(buyer, "DEP", -amt), Entry(seller, "DEP", amt)), memo="shortage premium")
        )


def cornering_capped(order: float, supply: float, cfg: FirmsFile) -> float:
    """Order-size cap as a share of cell supply."""
    cap = cfg.procurement.order_size_cap_of_cell * supply
    if order > cap + 1e-12:
        raise LedgerError(f"order {order} exceeds cell cap {cap}")
    return min(order, cap)
