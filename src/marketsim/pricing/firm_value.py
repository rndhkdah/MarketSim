"""Agent-firm fundamental value and thin engine quote (T6.17 / §6.3, §6.6).

The sector model ``ln V = ln E^e + ln PE0 − D·Δρ + D·Δg_lr`` is applied to the
firm's *lagged public* reports (T5.13). Live books are never read. Dividends and
dilution enter the per-share cash-flow term; leverage raises ``adj_j`` on ``ρ``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marketsim.firms.accounts import FirmBooks
from marketsim.firms.reports import ReportDesk
from marketsim.pricing.fundamentals import fundamental_values

# markets.clob.thin_quote_w default (§6.6 / markets.yaml). Pass ``w``; do not load yaml.
THIN_QUOTE_W = 0.03
# Floor on book equity in D/E so distressed reports stay defined (provider uses 1e-12 on ee).
_EQUITY_FLOOR = 1e-12


def read_published_profit(report: Any) -> float:
    """Published after-tax profit (cr / year, same convention as ``fundamental_values``)."""
    return _read_published(report, "published_profit")


def read_published_dividends(report: Any) -> float:
    """Published dividends (cr / year). Magnitude; sign is ignored."""
    return _read_published(report, "published_dividends")


def read_published_shares(report: Any) -> float:
    """Published shares outstanding (shares)."""
    return _read_published(report, "published_shares")


def read_published_debt(report: Any) -> float:
    """Published debt (cr). Lagged ``sheet.debt`` is public."""
    return _read_published(report, "published_debt", sheet_key="debt")


def read_published_equity(report: Any) -> float:
    """Published book equity (cr). Lagged ``sheet.equity`` is public."""
    return _read_published(report, "published_equity", sheet_key="equity")


def lagged_public_report(desk: ReportDesk, firm_id: str, tick: int) -> dict[str, Any] | None:
    """Latest monthly row visible after ``reports.monthly_lag_days``.

    Reads ``desk.monthly`` only — never ``desk.live``. Returns a shallow copy, or
    ``None`` when nothing has been published long enough. Units follow the row.
    """
    lag = int(desk.cfg.reports.monthly_lag_days)
    hist = desk.monthly.get(firm_id, [])
    visible = [row for row in hist if int(tick) - int(row["tick"]) >= lag]
    if not visible:
        return None
    row = visible[-1]
    out = dict(row)
    sheet = out.get("sheet")
    if isinstance(sheet, Mapping):
        out["sheet"] = dict(sheet)
        if "published_debt" not in out and "debt" in sheet:
            out["published_debt"] = float(sheet["debt"])
        if "published_equity" not in out and "equity" in sheet:
            out["published_equity"] = float(sheet["equity"])
    return out


def stamp_published(
    desk: ReportDesk,
    firm_id: str,
    books: FirmBooks,
    *,
    shares: float,
) -> dict[str, float]:
    """Copy valuation fields onto the latest monthly row.

    Call at publication (alongside ``publish_month``). Later valuation must use
    :func:`lagged_public_report`, not ``books`` or ``desk.live``.

    ``shares`` is the published share count. Profit is ``net_income`` (cr / period);
    dividends are ``abs(statement.dividends)`` (cash paid, cr / period).
    """
    hist = desk.monthly.get(firm_id)
    if not hist:
        raise ValueError(f"no published monthly row for {firm_id}")
    stmt = books.statements[-1] if books.statements else None
    sheet = books.sheets[-1] if books.sheets else None
    fields = {
        "published_profit": float(stmt.net_income) if stmt is not None else 0.0,
        "published_dividends": abs(float(stmt.dividends)) if stmt is not None else 0.0,
        "published_shares": float(shares),
        "published_debt": float(sheet.debt) if sheet is not None else 0.0,
        "published_equity": float(sheet.equity) if sheet is not None else 0.0,
    }
    hist[-1].update(fields)
    return fields


def leverage_risk_adj(debt: float, equity: float) -> float:
    """Firm ``adj_j`` from published leverage (annual decimal).

    ``ρ = y10_real + ERP + adj`` (§6.3). ``adj = D / max(E, 1e-12)``. Higher
    leverage raises adj. Equity add-on only — not a loan spread (D14).
    """
    return float(debt) / max(float(equity), _EQUITY_FLOOR)


def firm_value(
    report: Any,
    *,
    pe0: float,
    duration: float,
    delta_rho: float = 0.0,
    delta_g_lr: float = 0.0,
) -> float:
    """Per-share fundamental ``V`` (cr / share) from a lagged public report.

    Same formula as :func:`marketsim.pricing.fundamentals.fundamental_values`:
    ``ln V = ln E^e + ln PE0 − D·(Δρ + adj) + D·Δg_lr``.

    ``E^e`` is published profit per share times the payout ratio (published
    dividends / profit), i.e. dividends per share — so a dividend cut or
    dilution lowers ``V``. ``adj`` is :func:`leverage_risk_adj` on published
    debt and equity.

    ``report`` is a mapping/object with ``published_*`` fields, a lagged desk
    row, or a non-live ``ReportDesk.observe()`` result. Live observe payloads
    are rejected.
    """
    report = _public_report(report)
    profit = read_published_profit(report)
    dividends = abs(read_published_dividends(report))
    shares = read_published_shares(report)
    debt = read_published_debt(report)
    equity = read_published_equity(report)
    if shares <= 0.0:
        raise ValueError("published_shares must be > 0")
    # DPS = (profit / shares) × (dividends / profit); profit==0 → dividends / shares.
    if profit == 0.0:
        ee = dividends / shares
    else:
        ee = (profit / shares) * (dividends / profit)
    adj = leverage_risk_adj(debt, equity)
    return float(fundamental_values(ee, float(pe0), float(duration), float(delta_rho) + adj, float(delta_g_lr)))


def thin_quote_feed(value: float, w: float = THIN_QUOTE_W) -> tuple[float, float]:
    """Engine quote around fundamental: ``(V·(1 − w), V·(1 + w))`` (§6.6).

    ``value`` is cr / share. ``w`` is dimensionless; default ``THIN_QUOTE_W``
    = 0.03 (``markets.clob.thin_quote_w``). Pass ``w`` as an argument.
    Returns ``(bid, ask)`` in cr / share.
    """
    v = float(value)
    width = float(w)
    return v * (1.0 - width), v * (1.0 + width)


def _public_report(report: Any) -> Any:
    """Unwrap a non-live observe() payload; reject live books."""
    if isinstance(report, Mapping) and "live" in report and "books" in report:
        if report["live"]:
            raise ValueError("valuation uses lagged public reports, not live books")
        return report["books"]
    return report


def _read_published(report: Any, key: str, *, sheet_key: str | None = None) -> float:
    """Read ``key`` only. Never probes live / truth / unprefixed aliases."""
    if report is None:
        raise TypeError("published report is required")
    if isinstance(report, Mapping):
        if key in report:
            return float(report[key])
        if sheet_key is not None:
            sheet = report.get("sheet")
            if isinstance(sheet, Mapping) and sheet_key in sheet:
                return float(sheet[sheet_key])
        raise KeyError(f"{key} missing from published report")
    try:
        val = getattr(report, key)
    except AttributeError as exc:
        raise AttributeError(f"published report has no {key}") from exc
    return float(val)
