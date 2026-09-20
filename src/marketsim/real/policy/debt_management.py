"""Debt-management office (T6.26 / §6.11). GOVT lever; autopilot splits 20/40/40."""

from __future__ import annotations

from typing import Any

from marketsim.core.clock import EventQueue
from marketsim.core.errors import ConfigError
from marketsim.ledger.opening import GOVT_MIX
from marketsim.market.auction import ANNOUNCE_LEAD_TICKS, announce_tick, auction_tick
from marketsim.pricing.bond_buckets import BondsFile, issuance_mix_tuple

DEPOSIT_BUFFER_OF_GDP = 0.0  # cr / annual GDP; top-up is explicit in financing_need


def financing_need(
    *,
    deficit: float,
    redemptions: float,
    deposit_buffer_topup: float = 0.0,
) -> float:
    """``deficit + redemptions + deposit-buffer top-up`` (cr / period)."""
    return float(deficit) + float(redemptions) + float(deposit_buffer_topup)


def split_issuance(
    need: float,
    mix: tuple[tuple[str, float], ...] | None = None,
    bonds: BondsFile | None = None,
) -> dict[str, float]:
    """Split a financing need across government buckets (cr of face)."""
    if need < 0:
        raise ConfigError("financing need must be >= 0")
    used = mix if mix is not None else issuance_mix_tuple(bonds)
    if abs(sum(s for _, s in used) - 1.0) > 1e-12:
        raise ConfigError("issuance mix must sum to 1")
    return {name: float(need) * float(share) for name, share in used}


def autopilot_calendar(month: int) -> dict[str, int]:
    """Announcement and auction ticks for 0-based ``month``."""
    return {
        "announce": announce_tick(month),
        "auction": auction_tick(month),
        "lead": ANNOUNCE_LEAD_TICKS,
    }


def default_mix() -> tuple[tuple[str, float], ...]:
    """Phase-2 / §6.11 20/40/40 split (same object as ``GOVT_MIX``)."""
    return GOVT_MIX


def dmo_state(need: float, bonds: BondsFile | None = None) -> dict[str, Any]:
    """JSON-safe snapshot of an autopilot issue plan."""
    return {
        "need": float(need),
        "sizes": split_issuance(need, bonds=bonds),
        "mix": list(issuance_mix_tuple(bonds)),
    }


def schedule_month(queue: EventQueue, month: int, *, sizes: dict[str, float] | None = None) -> None:
    """Queue the announcement and the auction as two payloads (§6.11, 5-tick lead)."""
    payload = {"month": int(month), "sizes": dict(sizes or {})}
    queue.schedule(announce_tick(month), {"kind": "auction_announce", **payload}, priority=0)
    queue.schedule(auction_tick(month), {"kind": "auction", **payload}, priority=0)
