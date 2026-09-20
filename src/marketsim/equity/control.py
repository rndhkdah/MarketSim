"""Operating control of agent firms (T6.16 / §6.7).

Control is a majority of voting shares: ownership > 0.50 (§6.7). A crossing
transfer does **not** move ``firm.operator`` on the trade tick. It queues
``pending_operator`` for the next :meth:`Calendar.is_month_end` tick strictly
after the transfer — so a transfer *on* a month-end waits for the following
month-end. Until that boundary, :func:`~marketsim.firms.levers.validate_decision`
still sees the old operator. The outgoing operator keeps shares and dividends.

Holdings ≥ 0.05 are disclosed in professional mode and hidden in game mode (§6.7).
Tender offers are out of scope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from marketsim.core.calendar import Calendar
from marketsim.core.errors import ConfigError
from marketsim.equity.captable import CASH, CapTable
from marketsim.events.news import NewsItem
from marketsim.firms.accounts import AGENT_PREFIX, agent_entity
from marketsim.firms.firm import Firm
from marketsim.ledger.journal import Ledger, Tx

# §6.7: > 50 % of voting shares → operating control.
CONTROL_THRESHOLD = 0.50
# §6.7: holdings ≥ 5 % are disclosed in professional mode.
DISCLOSURE_THRESHOLD = 0.05

NEWS_CATEGORY = "corporate"


def holder_entity(name: str) -> str:
    """Ledger entity for an operator or holder id. Unitless."""
    if ":" in name:
        return name
    return agent_entity(name)


def operator_id(holder: str, *, like: str) -> str:
    """Rewrite ``holder`` to match the naming of ``like`` (``firm.operator``).

    Both arguments are unitless ids. Short ids (``\"bob\"``) stay short when
    ``like`` is short; ``AGENT:`` entities stay prefixed when ``like`` is.
    """
    hold = holder_entity(holder)
    if ":" in like:
        return hold
    if hold.startswith(AGENT_PREFIX):
        return hold[len(AGENT_PREFIX) :]
    if ":" in hold:
        return hold.split(":", 1)[-1]
    return hold


def next_month_end(calendar: Calendar, tick: int) -> int:
    """First month-end tick strictly after ``tick`` (days).

    A crossing transfer on a month-end is effective at the *following*
    month-end, not the same day (§6.7).
    """
    t = int(tick) + 1
    dpm = calendar.days_per_month
    rem = t % dpm
    if rem == dpm - 1:
        return t
    return t + (dpm - 1 - rem)


def disclosures(table: CapTable, ledger: Ledger, *, mode: str) -> dict[str, float]:
    """Holders with ownership ≥ 0.05 (dimensionless) when ``mode==\"professional\"``.

    Game mode returns ``{}`` (hidden). Keys are ledger entity ids in register
    order; values are ``holding / outstanding`` (§6.7).
    """
    if mode != "professional":
        return {}
    out: dict[str, float] = {}
    for holder in table.holdings(ledger):
        own = table.ownership(ledger, holder)
        if own >= DISCLOSURE_THRESHOLD:
            out[holder] = own
    return out


def _controlling_holder(table: CapTable, ledger: Ledger) -> str | None:
    """Ledger entity with ownership > 0.50, or None. Unitless id."""
    for holder in table.holdings(ledger):
        if table.ownership(ledger, holder) > CONTROL_THRESHOLD:
            return holder
    return None


@dataclass
class PendingOperator:
    """Queued control change. ``effective_tick`` is a day index (month-end)."""

    firm_id: str
    operator: str
    holder: str
    effective_tick: int

    def to_state(self) -> dict[str, Any]:
        return {
            "firm_id": self.firm_id,
            "operator": self.operator,
            "holder": self.holder,
            "effective_tick": int(self.effective_tick),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> PendingOperator:
        return cls(
            firm_id=str(state["firm_id"]),
            operator=str(state["operator"]),
            holder=str(state["holder"]),
            effective_tick=int(state["effective_tick"]),
        )


def _news_from_state(raw: dict[str, Any]) -> NewsItem:
    return NewsItem(
        id=str(raw["id"]),
        tick=int(raw["tick"]),
        category=str(raw["category"]),
        headline=str(raw["headline"]),
        regions=tuple(str(r) for r in raw.get("regions", ())),
        sectors=tuple(str(s) for s in raw.get("sectors", ())),
        severity_hint=int(raw.get("severity_hint", 0)),
        is_rumour=bool(raw.get("is_rumour", False)),
    )


@dataclass
class ControlDesk:
    """Queues majority takeovers and applies them at the next month boundary."""

    calendar: Calendar = field(default_factory=Calendar)
    pending: dict[str, PendingOperator] = field(default_factory=dict)
    news: list[NewsItem] = field(default_factory=list)

    def pending_operator(self, firm_id: str) -> str | None:
        """Queued new ``firm.operator``, or None. Unitless agent id."""
        row = self.pending.get(firm_id)
        return None if row is None else row.operator

    def disclosures(self, table: CapTable, ledger: Ledger, *, mode: str) -> dict[str, float]:
        """See :func:`disclosures`. ``mode`` is ``professional`` or ``game``."""
        return disclosures(table, ledger, mode=mode)

    def transfer(
        self,
        table: CapTable,
        ledger: Ledger,
        firm: Firm,
        *,
        src: str,
        dst: str,
        shares: float,
        tick: int,
        price: float | None = None,
        cash: str = CASH,
    ) -> Tx:
        """Secondary transfer (shares; optional ``price`` cr/share) then review control.

        ``tick`` is days. Cash legs are cr. Does not change ``firm.operator``.
        """
        tx = table.transfer(ledger, src=src, dst=dst, shares=shares, tick=tick, price=price, cash=cash)
        self.review(table, ledger, firm, tick)
        return tx

    def review(self, table: CapTable, ledger: Ledger, firm: Firm, tick: int) -> PendingOperator | None:
        """Queue control if some holder now has ownership > 0.50 (§6.7).

        ``tick`` is the transfer day. Effective date is :func:`next_month_end`.
        Cancels a stale queue when nobody (or the incumbent) holds a majority.
        """
        if table.firm_id != firm.id:
            raise ConfigError(f"cap table {table.firm_id!r} does not match firm {firm.id!r}")
        holder = _controlling_holder(table, ledger)
        if holder is None:
            self.pending.pop(firm.id, None)
            return None
        new_op = operator_id(holder, like=firm.operator)
        if holder_entity(new_op) == holder_entity(firm.operator):
            self.pending.pop(firm.id, None)
            return None
        row = PendingOperator(
            firm_id=firm.id,
            operator=new_op,
            holder=holder,
            effective_tick=next_month_end(self.calendar, tick),
        )
        self.pending[firm.id] = row
        return row

    def apply_month_boundary(self, firm: Firm, tick: int) -> list[NewsItem]:
        """Set ``firm.operator`` if ``tick`` (days) is a due month-end.

        No-op mid-month or before the queued effective tick. Previous operator
        keeps shares. Emits a ``corporate`` :class:`NewsItem` (no magnitude).
        """
        row = self.pending.get(firm.id)
        if row is None:
            return []
        t = int(tick)
        if t < row.effective_tick or not self.calendar.is_month_end(t):
            return []
        firm.operator = row.operator
        self.pending.pop(firm.id, None)
        item = NewsItem(
            id=f"control:{firm.id}:{t}",
            tick=t,
            category=NEWS_CATEGORY,
            headline=f"Operating control of {firm.id} passed to {row.operator}",
            regions=tuple(sorted({p.region for p in firm.plants})),
            sectors=tuple(sorted({p.sector for p in firm.plants})),
            severity_hint=0,
            is_rumour=False,
        )
        self.news.append(item)
        return [item]

    def to_state(self) -> dict[str, Any]:
        pending = [self.pending[k].to_state() for k in sorted(self.pending)]
        return {
            "calendar": {"days_per_month": int(self.calendar.days_per_month)},
            "pending": pending,
            "news": [asdict(item) for item in self.news],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ControlDesk:
        raw_cal = state.get("calendar") or {}
        dpm = raw_cal.get("days_per_month")
        calendar = Calendar() if dpm is None else Calendar(days_per_month=int(dpm))
        desk = cls(calendar=calendar)
        for row in state.get("pending", ()):
            item = PendingOperator.from_state(row)
            desk.pending[item.firm_id] = item
        desk.news = [_news_from_state(raw) for raw in state.get("news", ())]
        return desk
