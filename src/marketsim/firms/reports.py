"""Lagged firm reports and professional blackout (T5.13 / §5.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from marketsim.core.errors import ConfigError
from marketsim.firms.accounts import FirmBooks
from marketsim.firms.firm import FirmsFile

Mode = Literal["game", "professional"]


@dataclass
class ReportDesk:
    """Operators see live books; others see lagged copies."""

    cfg: FirmsFile
    mode: Mode
    live: dict[str, FirmBooks] = field(default_factory=dict)
    monthly: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    quarterly: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    last_publish_tick: dict[str, int] = field(default_factory=dict)
    quarter_end_tick: dict[str, int] = field(default_factory=dict)

    def publish_month(self, firm_id: str, books: FirmBooks, tick: int) -> None:
        self.live[firm_id] = books
        sheet = books.sheets[-1].__dict__ if books.sheets else {}
        self.monthly.setdefault(firm_id, []).append({"tick": tick, "sheet": dict(sheet)})
        self.last_publish_tick[firm_id] = tick
        if tick % 63 == 62:  # 21 days × 3 months
            self.quarterly.setdefault(firm_id, []).append({"tick": tick, "sheet": dict(sheet)})
            self.quarter_end_tick[firm_id] = tick

    def observe(self, firm_id: str, *, viewer: str, operator: str, tick: int) -> dict[str, Any]:
        """Return books the viewer is allowed to see."""
        if viewer == operator:
            books = self.live.get(firm_id)
            return {"live": True, "books": books.to_state() if books else {}}
        lag = self.cfg.reports.monthly_lag_days
        hist = self.monthly.get(firm_id, [])
        visible = [row for row in hist if tick - int(row["tick"]) >= lag]
        return {"live": False, "books": visible[-1] if visible else None}

    def can_trade_own_shares(self, firm_id: str, *, tick: int) -> bool:
        """Blackout from quarter end to publication; professional mode only."""
        if self.mode != "professional":
            return True
        qend = self.quarter_end_tick.get(firm_id)
        if qend is None:
            return True
        pub = qend + self.cfg.reports.quarterly_lag_days
        return tick >= pub


def reject_unlagged(desk: ReportDesk, firm_id: str, viewer: str, operator: str, tick: int) -> None:
    obs = desk.observe(firm_id, viewer=viewer, operator=operator, tick=tick)
    if viewer != operator and obs["live"]:
        raise ConfigError("non-operator saw live books")
