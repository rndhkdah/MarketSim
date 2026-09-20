"""T6.17 — agent-firm value from lagged public reports; thin quote feed."""

from __future__ import annotations

import pytest

from marketsim.firms.accounts import BalanceSheet, FirmBooks, IncomeStatement
from marketsim.firms.firm import FirmsFile
from marketsim.firms.reports import ReportDesk
from marketsim.pricing.firm_value import (
    THIN_QUOTE_W,
    firm_value,
    lagged_public_report,
    leverage_risk_adj,
    stamp_published,
    thin_quote_feed,
)


class _PoisonLiveBooks:
    """Published snapshot plus live/truth fields valuation must never touch."""

    def __init__(
        self,
        *,
        profit: float,
        dividends: float,
        shares: float,
        debt: float,
        equity: float,
        live_profit: float,
    ) -> None:
        self.published_profit = float(profit)
        self.published_dividends = float(dividends)
        self.published_shares = float(shares)
        self.published_debt = float(debt)
        self.published_equity = float(equity)
        self._live_profit = float(live_profit)
        self.live_reads = 0

    @property
    def live(self) -> dict[str, float]:
        self.live_reads += 1
        return {"profit": self._live_profit}

    @property
    def profit(self) -> float:
        self.live_reads += 1
        return self._live_profit

    @property
    def dividends(self) -> float:
        self.live_reads += 1
        return self._live_profit

    @property
    def shares(self) -> float:
        self.live_reads += 1
        return self._live_profit

    @property
    def debt(self) -> float:
        self.live_reads += 1
        return self._live_profit

    @property
    def equity(self) -> float:
        self.live_reads += 1
        return self._live_profit

    @property
    def statements(self) -> list[dict[str, float]]:
        self.live_reads += 1
        return [{"net_income": self._live_profit, "dividends": -self._live_profit}]

    @property
    def sheets(self) -> list[dict[str, float]]:
        self.live_reads += 1
        return [{"debt": self._live_profit, "equity": 1.0}]

    @property
    def truth(self) -> float:
        self.live_reads += 1
        return self._live_profit

    @property
    def books(self) -> dict[str, float]:
        self.live_reads += 1
        return {"profit": self._live_profit}


def _pub(
    *,
    profit: float = 12.0,
    dividends: float = 6.0,
    shares: float = 100.0,
    debt: float = 20.0,
    equity: float = 80.0,
) -> dict[str, float]:
    return {
        "published_profit": profit,
        "published_dividends": dividends,
        "published_shares": shares,
        "published_debt": debt,
        "published_equity": equity,
    }


def _closed_books(*, revenue: float, dividends: float, debt: float, equity: float) -> FirmBooks:
    books = FirmBooks("acme")
    books.statements.append(IncomeStatement(revenue=revenue, dividends=-abs(dividends)))
    books.sheets.append(BalanceSheet(cash=10.0, real_assets=equity + debt - 10.0, debt=debt, equity=equity))
    return books


def test_value_reacts_to_published_reports_only() -> None:
    desk = ReportDesk(FirmsFile(), "professional")
    books = _closed_books(revenue=12.0, dividends=6.0, debt=20.0, equity=80.0)
    desk.publish_month("acme", books, tick=0)
    stamp_published(desk, "acme", books, shares=100.0)

    assert lagged_public_report(desk, "acme", tick=9) is None
    kwargs = {"pe0": 10.0, "duration": 8.0}
    published = lagged_public_report(desk, "acme", tick=10)
    assert published is not None
    v0 = firm_value(published, **kwargs)

    books.statements[-1].revenue = 1e9
    books.statements[-1].dividends = -1e9
    books.sheets[-1].debt = 1e9
    books.sheets[-1].equity = 1.0
    desk.live["acme"] = books

    v_poisoned_desk = firm_value(lagged_public_report(desk, "acme", tick=10), **kwargs)
    assert v_poisoned_desk == v0

    live_obs = desk.observe("acme", viewer="alice", operator="alice", tick=10)
    assert live_obs["live"] is True
    with pytest.raises(ValueError, match="live"):
        firm_value(live_obs, **kwargs)

    mkt_obs = desk.observe("acme", viewer="bob", operator="alice", tick=10)
    assert mkt_obs["live"] is False
    assert firm_value(mkt_obs, **kwargs) == v0

    poison = _PoisonLiveBooks(
        profit=12.0, dividends=6.0, shares=100.0, debt=20.0, equity=80.0, live_profit=1e9
    )
    v_obj = firm_value(poison, **kwargs)
    assert poison.live_reads == 0
    assert v_obj == pytest.approx(v0)
    poison.published_profit = 24.0
    poison.published_dividends = 12.0
    v_new_pub = firm_value(poison, **kwargs)
    assert poison.live_reads == 0
    assert v_new_pub > v0

    later = _closed_books(revenue=24.0, dividends=12.0, debt=20.0, equity=80.0)
    desk.publish_month("acme", later, tick=21)
    stamp_published(desk, "acme", later, shares=100.0)
    assert firm_value(lagged_public_report(desk, "acme", tick=30), **kwargs) == v0
    assert firm_value(lagged_public_report(desk, "acme", tick=31), **kwargs) > v0


def test_dividend_cut_lowers_value() -> None:
    base = _pub(profit=12.0, dividends=6.0)
    cut = _pub(profit=12.0, dividends=3.0)
    v0 = firm_value(base, pe0=10.0, duration=8.0)
    v1 = firm_value(cut, pe0=10.0, duration=8.0)
    assert v1 < v0
    assert v1 == pytest.approx(0.5 * v0)


def test_dilution_lowers_value() -> None:
    base = _pub(shares=100.0)
    diluted = _pub(shares=200.0)
    v0 = firm_value(base, pe0=10.0, duration=8.0)
    v1 = firm_value(diluted, pe0=10.0, duration=8.0)
    assert v1 < v0
    assert v1 == pytest.approx(0.5 * v0)


def test_leverage_raises_risk_adj() -> None:
    low = leverage_risk_adj(20.0, 80.0)
    high = leverage_risk_adj(60.0, 80.0)
    assert high > low
    assert low == pytest.approx(0.25)
    assert high == pytest.approx(0.75)
    v_low = firm_value(_pub(debt=20.0, equity=80.0), pe0=10.0, duration=8.0)
    v_high = firm_value(_pub(debt=60.0, equity=80.0), pe0=10.0, duration=8.0)
    assert v_high < v_low


def test_thin_quote_feed_around_fundamental() -> None:
    bid, ask = thin_quote_feed(100.0)
    assert THIN_QUOTE_W == 0.03
    assert bid == pytest.approx(100.0 * (1.0 - THIN_QUOTE_W))
    assert ask == pytest.approx(100.0 * (1.0 + THIN_QUOTE_W))
    bid_w, ask_w = thin_quote_feed(100.0, w=0.05)
    assert bid_w == pytest.approx(95.0)
    assert ask_w == pytest.approx(105.0)
