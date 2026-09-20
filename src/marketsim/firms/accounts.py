"""Firm and agent ledger accounts (T5.02 / §5.2, gate 6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marketsim.core.errors import LedgerError
from marketsim.ledger.instruments import DEFAULT_REAL
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent

FIRM_PREFIX = "FIRM:"
AGENT_PREFIX = "AGENT:"
EQUITY_PREFIX = "EQ:"

REVENUE_TAGS = frozenset({"consumption", "govt_purchases", "exports"})
COGS_TAGS = frozenset({"intermediate"})
WAGE_TAGS = frozenset({"wages"})
INTEREST_TAGS = frozenset({"interest_loans", "interest_bonds", "interest_deposits", "interest_reserves"})
TAX_TAGS = frozenset({"corp_tax", "vat", "income_tax", "excise", "tariff", "transaction_tax"})
DIVIDEND_TAGS = frozenset({"dividends"})
CAPITAL_TAGS = frozenset({"capital_transfer", "equity_issue"})


def firm_entity(firm_id: str) -> str:
    """Ledger name for a firm. Unitless id."""
    return f"{FIRM_PREFIX}{firm_id}"


def agent_entity(agent_id: str) -> str:
    """Ledger name for an operator / owner. Unitless id."""
    return f"{AGENT_PREFIX}{agent_id}"


def firm_equity_instrument(firm_id: str) -> str:
    """Owners' equity claim on ``FIRM:<id>``. Financial; sums to zero."""
    return f"{EQUITY_PREFIX}{FIRM_PREFIX}{firm_id}"


def register_firm(ledger: Ledger, firm_id: str, *, operator: str | None = None) -> str:
    """Register ``FIRM:<id>``, optional ``AGENT:<id>``, and the firm's equity instrument."""
    name = firm_entity(firm_id)
    ledger.register_entity(name)
    ledger.register_instrument(firm_equity_instrument(firm_id), financial=True)
    if operator:
        ledger.register_entity(agent_entity(operator))
    return name


def post_founding_capital(
    ledger: Ledger,
    *,
    firm_id: str,
    amount: float,
    source: str,
    tick: int,
    tag: str = "equity_issue",
) -> None:
    """Owner cash → firm DEP; matching equity issue. ``amount`` in cr.

    ``tag`` must be ``equity_issue`` or ``capital_transfer``.
    """
    if tag not in CAPITAL_TAGS:
        raise LedgerError(f"founding tag must be capital_transfer or equity_issue, got {tag!r}")
    if amount <= 0:
        raise LedgerError("founding capital must be > 0")
    firm = firm_entity(firm_id)
    eq = firm_equity_instrument(firm_id)
    if source not in ledger.entities:
        ledger.register_entity(source)
    if firm not in ledger.entities:
        register_firm(ledger, firm_id)
    ledger.post(
        Tx(
            tick,
            tag,
            (
                Entry(source, "DEP", -amount),
                Entry(firm, "DEP", amount),
                Entry(firm, eq, -amount),
                Entry(source, eq, amount),
            ),
            memo=f"found {firm_id}",
        )
    )


def retain_earnings(ledger: Ledger, *, firm_id: str, owner: str, amount: float, tick: int) -> None:
    """Book net income to equity so A = L + E. ``amount`` in cr."""
    if abs(amount) <= 1e-15:
        return
    firm = firm_entity(firm_id)
    eq = firm_equity_instrument(firm_id)
    ledger.post(
        Tx(
            tick,
            "equity_issue",
            (Entry(firm, eq, -amount), Entry(owner, eq, amount)),
            memo="retain",
        )
    )


@dataclass
class IncomeStatement:
    """Period P&L from DEP-tagged journal lines. Fields are cr (income +, expense −)."""

    revenue: float = 0.0
    cogs: float = 0.0
    wages: float = 0.0
    interest: float = 0.0
    tax: float = 0.0
    dividends: float = 0.0
    other: float = 0.0
    capital: float = 0.0

    @property
    def net_income(self) -> float:
        """cr / period; before dividends and capital appropriations."""
        return self.revenue + self.cogs + self.wages + self.interest + self.tax + self.other

    def ties_to_tags(self, dep_by_tag: dict[str, float], *, atol: float = 1e-8) -> bool:
        """Each P&L line equals the DEP journal total for its tags."""
        rev = sum(dep_by_tag.get(t, 0.0) for t in REVENUE_TAGS)
        cogs = sum(dep_by_tag.get(t, 0.0) for t in COGS_TAGS)
        wages = sum(dep_by_tag.get(t, 0.0) for t in WAGE_TAGS)
        interest = sum(dep_by_tag.get(t, 0.0) for t in INTEREST_TAGS)
        tax = sum(dep_by_tag.get(t, 0.0) for t in TAX_TAGS)
        div = sum(dep_by_tag.get(t, 0.0) for t in DIVIDEND_TAGS)
        cap = sum(dep_by_tag.get(t, 0.0) for t in CAPITAL_TAGS)
        return (
            abs(self.revenue - rev) <= atol
            and abs(self.cogs - cogs) <= atol
            and abs(self.wages - wages) <= atol
            and abs(self.interest - interest) <= atol
            and abs(self.tax - tax) <= atol
            and abs(self.dividends - div) <= atol
            and abs(self.capital - cap) <= atol
        )


@dataclass
class BalanceSheet:
    """Snapshot. Assets and liabilities in cr; real assets at replacement cost."""

    cash: float = 0.0
    real_assets: float = 0.0
    other_financial_assets: float = 0.0
    debt: float = 0.0
    equity: float = 0.0

    @property
    def assets(self) -> float:
        return self.cash + self.real_assets + self.other_financial_assets

    @property
    def liabilities_and_equity(self) -> float:
        return self.debt + self.equity

    def identity_holds(self, *, atol: float = 1e-8) -> bool:
        """A = L + E (gate 6)."""
        return abs(self.assets - self.liabilities_and_equity) <= atol


def _dep_by_tag(ledger: Ledger, entity: str, start: int, end: int) -> dict[str, float]:
    ei = ledger.entities.id(entity)
    dep = ledger.instruments.id("DEP")
    out: dict[str, float] = {}
    for i in range(start, end):
        if ledger._ent[i] != ei or ledger._inst[i] != dep:
            continue
        tag = ledger._tags[i]
        out[tag] = out.get(tag, 0.0) + float(ledger._amt[i])
    return out


def _statement_from_dep(dep: dict[str, float]) -> IncomeStatement:
    stmt = IncomeStatement()
    known = REVENUE_TAGS | COGS_TAGS | WAGE_TAGS | INTEREST_TAGS | TAX_TAGS | DIVIDEND_TAGS | CAPITAL_TAGS
    for tag, amt in dep.items():
        if tag in CAPITAL_TAGS:
            stmt.capital += amt
        elif tag in DIVIDEND_TAGS:
            stmt.dividends += amt
        elif tag in REVENUE_TAGS:
            stmt.revenue += amt
        elif tag in COGS_TAGS:
            stmt.cogs += amt
        elif tag in WAGE_TAGS:
            stmt.wages += amt
        elif tag in INTEREST_TAGS:
            stmt.interest += amt
        elif tag in TAX_TAGS:
            stmt.tax += amt
        elif tag not in {"opening"} and tag not in known:
            stmt.other += amt
    return stmt


@dataclass
class FirmBooks:
    """Per-firm chart of accounts driven by ledger flow tags."""

    firm_id: str
    statements: list[IncomeStatement] = field(default_factory=list)
    sheets: list[BalanceSheet] = field(default_factory=list)
    journal_cursor: int = 0

    @property
    def entity(self) -> str:
        return firm_entity(self.firm_id)

    def balance_sheet(self, ledger: Ledger) -> BalanceSheet:
        """Read positions. Real instruments are replacement-cost stocks."""
        e = self.entity
        eq = firm_equity_instrument(self.firm_id)
        cash = ledger.position(e, "DEP") if "DEP" in ledger.instruments else 0.0
        real = 0.0
        for inst in DEFAULT_REAL:
            if inst in ledger.instruments:
                real += ledger.position(e, inst)
        debt = 0.0
        other_fa = 0.0
        equity = 0.0
        for name in ledger.instruments.financial_names:
            pos = ledger.position(e, name)
            if name == eq:
                equity = -pos
            elif name == "DEP":
                continue
            elif pos < 0:
                debt += -pos
            else:
                other_fa += pos
        return BalanceSheet(
            cash=cash,
            real_assets=real,
            other_financial_assets=other_fa,
            debt=debt,
            equity=equity,
        )

    def close_month(self, ledger: Ledger, *, owner: str | None = None, tick: int = 0) -> IncomeStatement:
        """P&L from new DEP tags; optionally retain NI; assert A=L+E and SFC."""
        end = len(ledger._ticks)
        dep = _dep_by_tag(ledger, self.entity, self.journal_cursor, end)
        stmt = _statement_from_dep(dep)
        if owner is not None:
            retain_earnings(ledger, firm_id=self.firm_id, owner=owner, amount=stmt.net_income, tick=tick)
        self.journal_cursor = len(ledger._ticks)
        if not stmt.ties_to_tags(dep):
            raise LedgerError(f"income statement does not tie to tags for {self.entity}")
        sheet = self.balance_sheet(ledger)
        if not sheet.identity_holds():
            raise LedgerError(
                f"balance-sheet identity failed for {self.entity}: "
                f"A={sheet.assets} L+E={sheet.liabilities_and_equity}"
            )
        self.statements.append(stmt)
        self.sheets.append(sheet)
        assert_consistent(ledger)
        return stmt

    def to_state(self) -> dict[str, Any]:
        return {
            "firm_id": self.firm_id,
            "statements": [s.__dict__ for s in self.statements],
            "sheets": [s.__dict__ for s in self.sheets],
            "journal_cursor": self.journal_cursor,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> FirmBooks:
        books = cls(firm_id=state["firm_id"], journal_cursor=int(state.get("journal_cursor", 0)))
        books.statements = [IncomeStatement(**row) for row in state.get("statements", ())]
        books.sheets = [BalanceSheet(**row) for row in state.get("sheets", ())]
        return books
