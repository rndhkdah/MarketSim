"""Shared in-process venue protocol (T6.08 / ADR-P8).

Daily ticks keep matching in-process; ``Venue`` is the seam so the engine MM
and the CLOB can be swapped or moved out of process later. Order / fill
shapes are the CLOB's — imported, not copied.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from marketsim.market.clob import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    SubmitResult,
    TimeInForce,
)

__all__ = [
    "Fill",
    "Order",
    "OrderStatus",
    "OrderType",
    "Side",
    "SubmitResult",
    "TimeInForce",
    "Venue",
]


@runtime_checkable
class Venue(Protocol):
    """Order-entry surface shared by ``EngineMM`` and ``CLOB``.

    ``submit`` / ``ingest`` accept an ``Order``. ``cancel`` / ``replace`` take
    ``(symbol, order_id)``. ``end_tick`` is the matching-session / day
    boundary (no required arguments). Return types may differ by venue;
    fills always use the shared ``Fill`` record.
    """

    def submit(self, order: Order) -> SubmitResult: ...

    def ingest(self, orders: Sequence[Order]) -> list[SubmitResult]: ...

    def cancel(self, symbol: str, order_id: int) -> SubmitResult: ...

    def replace(
        self,
        symbol: str,
        order_id: int,
        *,
        qty: int | None = None,
        price: float | None = None,
    ) -> SubmitResult: ...

    def end_tick(self) -> object: ...
