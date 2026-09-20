"""Corporate bond pool: ``CORPPOOL`` issues ``CORP_POOL`` and on-lends ``CLOAN`` (T6.29 / §6.11).

One common menu (D14): floating bank loan at ``r + s_t`` or fixed pool funding at
``y_match + s_t``. No firm / sector / region / rating price term. Selling pressure
and defaults enter the **borrowing** spread equally for every name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from marketsim.firms.financing import FundingMenu, funding_menu
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.market.auction import (
    ETA_A,
    AuctionResult,
    Bid,
    clear_uniform,
)

ISSUER = "CORPPOOL"
POOL_INSTRUMENT = "CORP_POOL"
CLOAN = "CLOAN"

# §6.11 duration table at 4.2 % (years). Linear interpolation of NOTE / BOND yields.
NOTE_DURATION_Y = 2.7
BOND_DURATION_Y = 7.1
POOL_DURATION_Y = 4.2
# §6.11 CORP_POOL decay δ (1/year); CLOAN uses the same decay.
POOL_DECAY = 0.20
# §6.11 λ_s on z_risk. Numerical value is the §2.9 risk-appetite loading Δspread += 0.5·z.
LAMBDA_S = 0.5
# §6.11 21-day mean of the ξ-implied spread deviation. Tick = 1 day.
XI_MEAN_DAYS = 21
# Weekly tap; 21-day month (master plan §7) → 5 ticks / week.
TAP_PERIOD_TICKS = 5


def y_match(y_note: float, y_bond: float) -> float:
    """Government yield of equal duration, interpolated NOTE / BOND (§6.11).

    ``y_note``, ``y_bond`` and the return value are annual decimals.
    """
    span = BOND_DURATION_Y - NOTE_DURATION_Y
    weight = (POOL_DURATION_Y - NOTE_DURATION_Y) / span
    return float(y_note) + weight * (float(y_bond) - float(y_note))


def fair_pool_spread(s_t: float, z_risk: float, lambda_s: float = LAMBDA_S) -> float:
    """Phase-2 ``s_t`` + ``λ_s · z_risk`` (§6.11). Annual decimal."""
    return float(s_t) + float(lambda_s) * float(z_risk)


def implied_spread_deviation(price: float, y_fair: float, decay: float = POOL_DECAY) -> float:
    """ξ-implied spread: invert ``P = (κ + δ) / (y + δ)`` at ``κ = y_fair`` (§6.11).

    ``price`` is the observed price per unit (NAV × exp(ξ), dimensionless).
    ``y_fair`` and the return (``y_imp − y_fair``) are annual decimals.
    """
    p = float(price)
    if p <= 0.0:
        raise ValueError("observed pool price must be > 0")
    yf = float(y_fair)
    d = float(decay)
    y_imp = (yf + d) / p - d
    return y_imp - yf


def observed_price(nav: float, xi: float) -> float:
    """``P = NAV · exp(ξ)`` (§6.11 price = fair × exp(ξ); NAV is credit value / unit)."""
    return float(nav) * math.exp(float(xi))


def pool_units(ledger: Ledger) -> float:
    """Outstanding ``CORP_POOL`` units (face)."""
    return -ledger.position(ISSUER, POOL_INSTRUMENT)


def cloan_face(ledger: Ledger) -> float:
    """Pool ``CLOAN`` asset face (cr)."""
    return ledger.position(ISSUER, CLOAN)


def nav_per_unit(ledger: Ledger) -> float:
    """Pool NAV per ``CORP_POOL`` unit (cr / unit). Holders share losses pro rata."""
    units = pool_units(ledger)
    if units <= 0.0:
        return 1.0
    return cloan_face(ledger) / units


def holder_value(ledger: Ledger, entity: str) -> float:
    """Mark of ``entity``'s ``CORP_POOL`` at pool NAV (cr)."""
    return ledger.position(entity, POOL_INSTRUMENT) * nav_per_unit(ledger)


def write_down_cloan(ledger: Ledger, firm: str, amount: float, *, tick: int) -> None:
    """Default writes down ``CLOAN`` (pool asset, firm liability). Tag ``loan_writeoff``."""
    loss = float(amount)
    if abs(loss) < 1e-15:
        return
    ledger.post(
        Tx(
            tick,
            "loan_writeoff",
            (Entry(ISSUER, CLOAN, -loss), Entry(firm, CLOAN, loss)),
            memo="pool CLOAN default",
        )
    )


def settle_tap(
    ledger: Ledger,
    result: AuctionResult,
    *,
    tick: int,
    npc_entity: str = "MM",
    price: float = 1.0,
) -> None:
    """Cash-for-``CORP_POOL`` at ``price`` per unit face. Tag ``bond_issue``. SFC-safe.

    Allocation matches :func:`marketsim.market.auction.settle_auction` (same
    ``clear_uniform`` book); issuer is ``CORPPOOL`` (§6.11), not ``GOVT``.
    """
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
                Entry(entity, POOL_INSTRUMENT, float(face)),
                Entry(ISSUER, POOL_INSTRUMENT, -float(face)),
                Entry(entity, "DEP", -cash),
                Entry(ISSUER, "DEP", cash),
            )
        )
    if entries:
        ledger.post(Tx(tick, "bond_issue", entries, memo="tap CORP_POOL"))


def on_lend(
    ledger: Ledger,
    face_by_borrower: Mapping[str, float],
    *,
    tick: int,
    price: float = 1.0,
) -> None:
    """On-lend tap proceeds through ``CLOAN`` at the issue ``price`` (cr / unit)."""
    entries: list[Entry] = []
    for firm, face in sorted(face_by_borrower.items()):
        qty = float(face)
        cash = qty * float(price)
        if abs(qty) < 1e-15 and abs(cash) < 1e-15:
            continue
        entries.extend(
            (
                Entry(ISSUER, CLOAN, qty),
                Entry(firm, CLOAN, -qty),
                Entry(ISSUER, "DEP", -cash),
                Entry(firm, "DEP", cash),
            )
        )
    if entries:
        ledger.post(Tx(tick, "loan_new", entries, memo="pool on-lend"))


def weekly_tap(
    ledger: Ledger,
    size: float,
    bids: tuple[Bid, ...] | list[Bid],
    *,
    y_fair: float,
    q0: float,
    tick: int,
    borrowers: Mapping[str, float],
    price: float = 1.0,
    npc_entity: str = "MM",
    eta_a: float = ETA_A,
    secondary_yield: float | None = None,
) -> AuctionResult:
    """Weekly tap: ``clear_uniform`` then settle and on-lend. Face in cr."""
    result = clear_uniform(
        size,
        bids,
        y_fair=y_fair,
        q0=q0,
        eta_a=eta_a,
        secondary_yield=secondary_yield,
    )
    settle_tap(ledger, result, tick=tick, npc_entity=npc_entity, price=price)
    on_lend(ledger, borrowers, tick=tick, price=price)
    return result


def is_tap_tick(tick: int, *, period: int = TAP_PERIOD_TICKS) -> bool:
    """True on weekly tap ticks (every ``period`` days; §6.11 weekly tap)."""
    return int(tick) % int(period) == 0


@dataclass
class CorpPool:
    """21-day window of ξ-implied spread deviations (§6.11). Units: annual decimal."""

    xi_devs: list[float] = field(default_factory=list)

    def xi_mean(self) -> float:
        """Mean of the last 21 daily deviations; missing days count as 0."""
        if not self.xi_devs:
            return 0.0
        window = self.xi_devs[-XI_MEAN_DAYS:]
        return float(sum(window) / XI_MEAN_DAYS)

    def mark_day(
        self,
        *,
        xi: float = 0.0,
        nav: float = 1.0,
        y_match: float,
        s_t: float,
        z_risk: float = 0.0,
    ) -> float:
        """Record today's ξ-implied spread deviation. Returns the borrowing spread."""
        s_fair = fair_pool_spread(s_t, z_risk)
        y_fair = float(y_match) + s_fair
        dev = implied_spread_deviation(observed_price(nav, xi), y_fair)
        self.xi_devs.append(float(dev))
        if len(self.xi_devs) > XI_MEAN_DAYS:
            del self.xi_devs[0]
        return self.borrowing_spread(s_t, z_risk)

    def borrowing_spread(self, s_t: float, z_risk: float = 0.0) -> float:
        """Fair pool spread plus the 21-day mean of the ξ-implied deviation. Annual decimal."""
        return fair_pool_spread(s_t, z_risk) + self.xi_mean()

    def terms(
        self,
        *,
        policy_rate: float,
        y_match: float,
        s_t: float,
        z_risk: float = 0.0,
    ) -> FundingMenu:
        """Common menu at this tick. ``y_match`` is already interpolated (§6.11)."""
        return funding_menu(
            policy_rate=policy_rate,
            y_match=y_match,
            spread=self.borrowing_spread(s_t, z_risk),
        )

    def to_state(self) -> dict[str, Any]:
        return {"xi_devs": list(self.xi_devs)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CorpPool:
        return cls(xi_devs=[float(x) for x in state["xi_devs"]])
