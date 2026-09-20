"""Heterogeneous-seller goods market (T5.05 / §5.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marketsim.firms.firm import FirmsFile


@dataclass
class SellerQuote:
    """One seller in a cell. Real quantities are units / month; price is an index."""

    id: str
    capacity: float
    posted_price: float
    avail: float
    share: float
    backlog: float = 0.0
    availability: float = 1.0  # dimensionless fill / stock-cover factor (§5.3)


@dataclass
class MarketResult:
    """One month. ``sales`` and ``unmet`` are units; ``share`` is dimensionless."""

    p_ref: float
    share: dict[str, float]
    sales: dict[str, float]
    unmet: dict[str, float]
    backlog: dict[str, float]
    spill: float
    demand: float

    @property
    def delivered(self) -> float:
        return float(sum(self.sales.values()))


def _target_shares(
    quotes: list[SellerQuote],
    p_ref: float,
    epsilon_s: float,
    kappa: float,
) -> np.ndarray:
    raw = []
    for q in quotes:
        rel = q.posted_price / p_ref if p_ref > 0 else 1.0
        raw.append(max(q.capacity, 0.0) * (rel ** (-epsilon_s)) * (max(q.availability, 0.0) ** kappa))
    w = np.asarray(raw, dtype=float)
    tot = float(w.sum())
    if tot <= 0.0:
        cap = np.array([max(q.capacity, 0.0) for q in quotes], dtype=float)
        s = float(cap.sum())
        return cap / s if s > 0 else np.full(len(quotes), 1.0 / max(len(quotes), 1))
    return w / tot


def _pref(quotes: list[SellerQuote]) -> float:
    sales_proxy = np.array([q.share * max(q.capacity, 0.0) for q in quotes], dtype=float)
    prices = np.array([q.posted_price for q in quotes], dtype=float)
    caps = np.array([max(q.capacity, 0.0) for q in quotes], dtype=float)
    w = float(sales_proxy.sum())
    if w > 0:
        return float((prices * sales_proxy).sum() / w)
    wc = float(caps.sum())
    if wc > 0:
        return float((prices * caps).sum() / wc)
    return float(prices.mean()) if prices.size else 1.0


def allocate(
    quotes: list[SellerQuote],
    demand: float,
    cfg: FirmsFile,
    *,
    backlog_loss: float = 0.10,
) -> MarketResult:
    """Stickiness + pro-rata ration + spare-supply spill. Conserves demand up to loss."""
    if not quotes:
        return MarketResult(1.0, {}, {}, {}, {}, 0.0, demand)
    gm = cfg.goods_market
    p_ref = _pref(quotes)
    star = _target_shares(quotes, p_ref, gm.epsilon_s, gm.availability_kappa)
    tau = gm.tau_loyalty_m
    new_share = []
    for q, s_star in zip(quotes, star, strict=True):
        s = q.share + (float(s_star) - q.share) / tau
        new_share.append(max(s, 0.0))
    z = float(sum(new_share))
    if z > 0:
        new_share = [s / z for s in new_share]
    else:
        new_share = [1.0 / len(quotes)] * len(quotes)
    for q, s in zip(quotes, new_share, strict=True):
        q.share = s

    orders = {q.id: s * demand + q.backlog for q, s in zip(quotes, new_share, strict=True)}
    sales = {q.id: min(orders[q.id], max(q.avail, 0.0)) for q in quotes}
    spare = {q.id: max(q.avail - sales[q.id], 0.0) for q in quotes}
    unmet = {q.id: max(orders[q.id] - sales[q.id], 0.0) for q in quotes}
    spill_pool = float(sum(unmet.values()))
    spare_tot = float(sum(spare.values()))
    if spill_pool > 0 and spare_tot > 0:
        take = min(spill_pool, spare_tot)
        for q in quotes:
            extra = take * spare[q.id] / spare_tot
            sales[q.id] += extra
            unmet[q.id] = max(unmet[q.id] - extra * (unmet[q.id] / spill_pool if spill_pool else 0.0), 0.0)
            spare[q.id] -= extra
        # leftover unmet after spill
        leftover = demand + sum(q.backlog for q in quotes) - float(sum(sales.values()))
        leftover = max(leftover, 0.0)
        w = float(sum(new_share))
        unmet = {q.id: leftover * (q.share / w if w else 0.0) for q in quotes}
        spill_used = take
    else:
        leftover = float(sum(unmet.values()))
        spill_used = 0.0
    backlog = {q.id: unmet[q.id] * (1.0 - backlog_loss) for q in quotes}
    for q in quotes:
        q.backlog = backlog[q.id]
        q.avail = max(q.avail - sales[q.id], 0.0)
    return MarketResult(
        p_ref=p_ref,
        share={q.id: q.share for q in quotes},
        sales=sales,
        unmet=unmet,
        backlog=backlog,
        spill=spill_used,
        demand=demand,
    )


def one_seller_r6(
    *,
    avail: float,
    demand: float,
    backlog: float,
    backlog_loss: float,
    is_stock: bool = True,
) -> tuple[float, float]:
    """Single-seller stock-mode R6: ``sales = min(avail, D+B)``, leftover leaks."""
    dem_eff = demand + (backlog if is_stock else 0.0)
    sales = min(avail, dem_eff)
    back = (dem_eff - sales) * (1.0 - backlog_loss)
    return sales, back
