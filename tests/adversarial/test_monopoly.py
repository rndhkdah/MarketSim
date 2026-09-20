"""T5.17 gate 3 — +30 % monopolist loses target share."""

from __future__ import annotations

from marketsim.firms.firm import FirmsFile
from marketsim.firms.goods_market import SellerQuote, allocate


def test_monopoly_price_hike_loses_share_to_npc() -> None:
    cfg = FirmsFile()
    quotes = [
        SellerQuote("npc", 10.0, 1.0, 10.0, 0.0),  # entry
        SellerQuote("mono", 100.0, 1.3, 100.0, 1.0),
    ]
    for _ in range(36):
        for q in quotes:
            q.avail = q.capacity
        allocate(quotes, 80.0, cfg)
    assert quotes[1].share < 0.95
    assert quotes[0].share > 0.05
