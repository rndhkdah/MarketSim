"""Margin loans, EOD checks, forced liquidation and the loss waterfall (T6.13 / §6.8).

Loans are originated by the broker (``MM``) and funded by ``BANKSYS`` at
``policy + broker_spread``. Equities use the ``markets.yaml`` 50 % / 25 %
haircuts; commodities 10 % / 7 %. A maintenance breach at end of tick queues
forced market orders for the next tick, largest mark-to-market loser first,
sized to restore initial margin and filled **with impact**. Residual losses
after equity is gone: broker up to ``broker_cap``, then a ``BANKSYS``
``loan_writeoff``. Cash and debt are never clipped.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from marketsim.core.errors import StateError
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.market.impact import ImpactKernel, post_impact_mid
from marketsim.market.instruments import COMMODITY_SECTORS, MarginCfg
from marketsim.market.mm import MM_ACCOUNT, MM_QUOTE_ID
from marketsim.market.settlement import settle_fill
from marketsim.market.venue import Fill, Side

# Master plan §6 / ``core.calendar``: 252 ticks / year.
DAYS_PER_YEAR = 252
# §6.8: customer rate is policy + broker spread. Not a ``markets.yaml`` key.
BROKER_SPREAD = 0.01  # annual decimal
# §6.8 broker loss cap (cr). Default 0 → residual goes to BANKSYS.
BROKER_CAP = 0.0
CASH = "DEP"
LOAN = "LOAN"
BANK = "BANKSYS"
_EPS = 1e-15


def haircut(symbol: str, cfg: MarginCfg, *, initial: bool) -> float:
    """Initial or maintenance rate (share of notional) for ``symbol`` (§6.8).

    Equities (``EQ:*``) use ``equity_*``; ``OIL`` / ``METALS`` / ``GRAINS`` use
    ``commodity_*``. Other names are not in the margin book (rate 0).
    """
    if symbol in COMMODITY_SECTORS:
        return float(cfg.commodity_initial if initial else cfg.commodity_maintenance)
    if symbol.startswith("EQ:"):
        return float(cfg.equity_initial if initial else cfg.equity_maintenance)
    return 0.0


def required_margin(qty: float, price: float, rate: float) -> float:
    """Haircut in cr: ``rate × |qty| × price``. ``price`` is cr/unit."""
    return float(rate) * abs(float(qty)) * float(price)


def account_equity(ledger: Ledger, account: str, prices: Mapping[str, float]) -> float:
    """Mark-to-market equity (cr): ``DEP + LOAN + Σ q_i p_i``.

    ``LOAN`` is at face (borrower negative). ``prices`` are cr/unit. Names
    missing from the ledger contribute 0. Never clipped.
    """
    eq = 0.0
    if CASH in ledger.instruments:
        eq += ledger.position(account, CASH)
    if LOAN in ledger.instruments:
        eq += ledger.position(account, LOAN)
    for symbol, px in prices.items():
        if symbol in (CASH, LOAN) or symbol not in ledger.instruments:
            continue
        eq += ledger.position(account, symbol) * float(px)
    return float(eq)


def requirement(
    ledger: Ledger,
    account: str,
    prices: Mapping[str, float],
    cfg: MarginCfg,
    *,
    initial: bool,
) -> float:
    """Sum of per-name haircuts (cr) at the initial or maintenance rate."""
    total = 0.0
    for symbol, px in prices.items():
        if symbol not in ledger.instruments:
            continue
        rate = haircut(symbol, cfg, initial=initial)
        if rate <= 0.0:
            continue
        q = ledger.position(account, symbol)
        if abs(q) < _EPS:
            continue
        total += required_margin(q, float(px), rate)
    return float(total)


def _ensure(ledger: Ledger, *entities: str) -> None:
    for name in entities:
        if name not in ledger.entities:
            ledger.register_entity(name)


def _post_loan_pair(
    ledger: Ledger,
    tick: int,
    tag: str,
    lender: str,
    borrower: str,
    amount: float,
    memo: str,
) -> Tx:
    """One bilateral loan + cash pair. ``amount`` cr. TFM 1–1 per instrument."""
    tx = Tx(
        tick,
        tag,
        (
            Entry(lender, LOAN, amount),
            Entry(borrower, LOAN, -amount),
            Entry(lender, CASH, -amount),
            Entry(borrower, CASH, amount),
        ),
        memo=memo,
    )
    ledger.post(tx)
    return tx


@dataclass(frozen=True, slots=True)
class MarginSnapshot:
    """EOD mark for one account. All money fields are cr."""

    account: str
    equity: float
    maintenance: float
    initial: float
    breached: bool


@dataclass(frozen=True, slots=True)
class ForcedLiquidation:
    """Queued at EOD, executed next tick. ``qty`` is units; ``pnl`` is cr."""

    account: str
    symbol: str
    side: Side
    qty: int
    pnl: float


@dataclass(frozen=True, slots=True)
class WaterfallResult:
    """Loss split after equity is gone. Fields are cr."""

    account: str
    broker_loss: float
    bank_writeoff: float


class MarginBook:
    """Margin-loan book: debit face, cost basis, and next-tick forced orders."""

    def __init__(
        self,
        cfg: MarginCfg | None = None,
        *,
        broker_spread: float = BROKER_SPREAD,
        broker_cap: float = BROKER_CAP,
        days_per_year: int = DAYS_PER_YEAR,
    ) -> None:
        self.cfg = cfg if cfg is not None else MarginCfg()
        self.broker_spread = float(broker_spread)
        self.broker_cap = float(broker_cap)
        self.days_per_year = int(days_per_year)
        self.debit: dict[str, float] = {}  # cr face outstanding (customer)
        self._cost: dict[tuple[str, str], float] = {}  # cr/unit average
        self._accounts: list[str] = []
        self.pending: list[ForcedLiquidation] = []
        self._next_oid = 1

    def track(self, account: str) -> None:
        """Include ``account`` in EOD checks (idempotent)."""
        if account not in self.debit:
            self.debit[account] = 0.0
        if account not in self._accounts:
            self._accounts.append(account)

    def set_cost(self, account: str, symbol: str, price: float) -> None:
        """Average entry price (cr/unit) used to rank losers."""
        self._cost[(account, symbol)] = float(price)

    def draw(self, ledger: Ledger, account: str, amount: float, tick: int) -> tuple[Tx, ...]:
        """Margin loan ``amount`` cr: ``BANKSYS → MM → account``. Tag ``loan_new``.

        Net: ``BANKSYS`` LOAN +``amount``, customer LOAN −``amount``; ``MM`` nets
        to 0. Customer cash rises by ``amount``. Never clipped.
        """
        x = float(amount)
        if x <= _EPS:
            return ()
        _ensure(ledger, BANK, MM_ACCOUNT, account)
        wholesale = _post_loan_pair(ledger, tick, "loan_new", BANK, MM_ACCOUNT, x, "margin wholesale")
        retail = _post_loan_pair(ledger, tick, "loan_new", MM_ACCOUNT, account, x, "margin retail")
        self.track(account)
        self.debit[account] = self.debit.get(account, 0.0) + x
        return (wholesale, retail)

    def repay(self, ledger: Ledger, account: str, amount: float, tick: int) -> tuple[Tx, ...]:
        """Repay ``amount`` cr of this book's debit (capped at outstanding face).

        Tag ``loan_repay``. Does not clip leftover cash. No-op if debit is 0.
        """
        x = min(float(amount), float(self.debit.get(account, 0.0)))
        if x <= _EPS:
            return ()
        _ensure(ledger, BANK, MM_ACCOUNT, account)
        retail = _post_loan_pair(ledger, tick, "loan_repay", account, MM_ACCOUNT, x, "margin retail")
        wholesale = _post_loan_pair(ledger, tick, "loan_repay", MM_ACCOUNT, BANK, x, "margin wholesale")
        self.debit[account] = self.debit.get(account, 0.0) - x
        if self.debit[account] < _EPS:
            self.debit[account] = 0.0
        return (retail, wholesale)

    def accrue_interest(
        self,
        ledger: Ledger,
        policy_rate: float,
        tick: int,
    ) -> tuple[Tx, ...]:
        """One day's interest (cr). Customer pays ``r + broker_spread`` to ``MM``;
        ``MM`` pays ``r`` to ``BANKSYS``. Tag ``interest_loans``. Annual rates.
        """
        posted: list[Tx] = []
        dt = 1.0 / float(self.days_per_year)
        retail_r = float(policy_rate) + self.broker_spread
        wholesale_r = float(policy_rate)
        for account in sorted(self.debit):
            face = self.debit[account]
            if face <= _EPS:
                continue
            retail = face * retail_r * dt
            wholesale = face * wholesale_r * dt
            if abs(retail) >= _EPS:
                tx = Tx(
                    tick,
                    "interest_loans",
                    (Entry(account, CASH, -retail), Entry(MM_ACCOUNT, CASH, retail)),
                    memo="margin retail",
                )
                ledger.post(tx)
                posted.append(tx)
            if abs(wholesale) >= _EPS:
                tx = Tx(
                    tick,
                    "interest_loans",
                    (Entry(MM_ACCOUNT, CASH, -wholesale), Entry(BANK, CASH, wholesale)),
                    memo="margin wholesale",
                )
                ledger.post(tx)
                posted.append(tx)
        return tuple(posted)

    def check(
        self,
        ledger: Ledger,
        prices: Mapping[str, float],
    ) -> tuple[MarginSnapshot, ...]:
        """End-of-tick maintenance test. Queues next-tick forced orders.

        Accounts are processed most-underwater first; within an account,
        positions are sold/covered largest mark-to-market loser first.
        Sized so remaining haircuts restore the **initial** rate (§6.8).
        """
        snaps: list[MarginSnapshot] = []
        for account in sorted(self._accounts):
            eq = account_equity(ledger, account, prices)
            maint = requirement(ledger, account, prices, self.cfg, initial=False)
            init = requirement(ledger, account, prices, self.cfg, initial=True)
            snaps.append(
                MarginSnapshot(
                    account=account,
                    equity=eq,
                    maintenance=maint,
                    initial=init,
                    breached=eq + _EPS < maint and maint > _EPS,
                )
            )
        # Most negative excess margin first (§6.8 largest loser).
        snaps.sort(key=lambda s: (s.equity - s.maintenance, s.account))
        pending: list[ForcedLiquidation] = []
        for snap in snaps:
            if snap.breached:
                pending.extend(self._plan(ledger, snap, prices))
        self.pending = pending
        return tuple(snaps)

    def take_forced_orders(self) -> tuple[ForcedLiquidation, ...]:
        """Drain the EOD queue (already largest-loser first)."""
        out = tuple(self.pending)
        self.pending = []
        return out

    def execute_forced(
        self,
        ledger: Ledger,
        prices: MutableMapping[str, float],
        kernels: Mapping[str, ImpactKernel],
        *,
        adv: Mapping[str, float],
        sigma: Mapping[str, float],
        spread: float,
        tick: int,
    ) -> tuple[Fill, ...]:
        """Fill queued liquidations next tick, sequentially, each with impact.

        ``prices`` are cr/unit and are updated to the post-impact mid after
        every fill (cascade-by-construction). ``adv`` is cr/day; ``sigma`` is
        daily vol (decimal); ``spread`` is the dimensionless quoted ``s``.
        Sale proceeds repay this book's debit. Never clipped.
        """
        fills: list[Fill] = []
        for item in self.take_forced_orders():
            mid = float(prices[item.symbol])
            kernel = kernels[item.symbol]
            sign = 1.0 if item.side is Side.BUY else -1.0
            q_cr = sign * float(item.qty) * mid
            xi_pre = kernel.xi
            px = float(
                kernel.fill(mid, float(spread), q_cr, float(adv[item.symbol]), float(sigma[item.symbol]))
            )
            oid = self._next_oid
            self._next_oid += 1
            fill = Fill(
                symbol=item.symbol,
                price=px,
                qty=item.qty,
                maker=MM_ACCOUNT,
                taker=item.account,
                maker_order_id=MM_QUOTE_ID,
                taker_order_id=oid,
                notional=px * float(item.qty),
                tick=tick,
            )
            settle_fill(ledger, fill, item.side)
            if item.side is Side.SELL:
                self.repay(ledger, item.account, fill.notional, tick)
            prices[item.symbol] = float(post_impact_mid(mid, xi_pre, kernel.xi))
            fills.append(fill)
        return tuple(fills)

    def waterfall(self, ledger: Ledger, account: str, tick: int) -> WaterfallResult:
        """Absorb remaining debit after the account is flat (§6.8).

        Any leftover ``DEP`` first repays the loan. Residual: ``MM`` up to
        ``broker_cap`` (cash gift + repay), then ``loan_writeoff`` at
        ``BANKSYS``. Never clipped.
        """
        cash = ledger.position(account, CASH) if CASH in ledger.instruments else 0.0
        if cash > _EPS and self.debit.get(account, 0.0) > _EPS:
            self.repay(ledger, account, cash, tick)
        residual = float(self.debit.get(account, 0.0))
        if residual <= _EPS:
            return WaterfallResult(account, 0.0, 0.0)
        broker = min(residual, self.broker_cap)
        if broker > _EPS:
            _ensure(ledger, MM_ACCOUNT, account)
            gift = Tx(
                tick,
                "capital_transfer",
                (Entry(MM_ACCOUNT, CASH, -broker), Entry(account, CASH, broker)),
                memo="broker margin support",
            )
            ledger.post(gift)
            self.repay(ledger, account, broker, tick)
        bank = float(self.debit.get(account, 0.0))
        if bank > _EPS:
            _ensure(ledger, BANK, account)
            wo = Tx(
                tick,
                "loan_writeoff",
                (Entry(BANK, LOAN, -bank), Entry(account, LOAN, bank)),
                memo="margin write-off",
            )
            ledger.post(wo)
            self.debit[account] = 0.0
        else:
            bank = 0.0
        return WaterfallResult(account, broker, bank)

    def resolve_deficits(
        self,
        ledger: Ledger,
        prices: Mapping[str, float],
        tick: int,
    ) -> tuple[WaterfallResult, ...]:
        """Waterfall accounts that still have debit and no marginable position."""
        out: list[WaterfallResult] = []
        for account in sorted(self.debit):
            if self.debit[account] <= _EPS:
                continue
            if requirement(ledger, account, prices, self.cfg, initial=True) > _EPS:
                continue
            out.append(self.waterfall(ledger, account, tick))
        return tuple(out)

    def _plan(
        self,
        ledger: Ledger,
        snap: MarginSnapshot,
        prices: Mapping[str, float],
    ) -> list[ForcedLiquidation]:
        held: list[tuple[str, float, float, float]] = []
        for symbol in sorted(prices):
            if symbol not in ledger.instruments:
                continue
            if haircut(symbol, self.cfg, initial=True) <= 0.0:
                continue
            q = ledger.position(snap.account, symbol)
            if abs(q) < _EPS:
                continue
            px = float(prices[symbol])
            cost = self._cost.get((snap.account, symbol), px)
            pnl = (px - cost) * q
            held.append((symbol, q, pnl, abs(q) * px))
        held.sort(key=lambda row: (row[2], -row[3], row[0]))
        need = snap.initial - snap.equity  # cr of initial haircut to release
        if need <= _EPS:
            return []
        out: list[ForcedLiquidation] = []
        for symbol, q, pnl, _notional in held:
            if need <= _EPS:
                break
            px = float(prices[symbol])
            lam = haircut(symbol, self.cfg, initial=True)
            per = lam * px
            if per <= _EPS:
                continue
            want = need / per
            n = int(min(abs(q), max(1, math.ceil(want - 1e-12))))
            if n <= 0:
                continue
            side = Side.SELL if q > 0.0 else Side.BUY
            out.append(ForcedLiquidation(snap.account, symbol, side, n, pnl))
            need -= float(n) * per
        return out

    def to_state(self) -> dict[str, Any]:
        return {
            "cfg": {
                "equity_initial": self.cfg.equity_initial,
                "equity_maintenance": self.cfg.equity_maintenance,
                "commodity_initial": self.cfg.commodity_initial,
                "commodity_maintenance": self.cfg.commodity_maintenance,
            },
            "broker_spread": self.broker_spread,
            "broker_cap": self.broker_cap,
            "days_per_year": self.days_per_year,
            "debit": {k: self.debit[k] for k in sorted(self.debit)},
            "cost": [[a, s, c] for (a, s), c in sorted(self._cost.items())],
            "accounts": list(self._accounts),
            "pending": [
                {
                    "account": p.account,
                    "symbol": p.symbol,
                    "side": str(p.side),
                    "qty": p.qty,
                    "pnl": p.pnl,
                }
                for p in self.pending
            ],
            "next_oid": self._next_oid,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> MarginBook:
        needed = ("cfg", "debit", "accounts")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"margin state missing {missing}")
        book = cls(
            MarginCfg(**state["cfg"]),
            broker_spread=float(state.get("broker_spread", BROKER_SPREAD)),
            broker_cap=float(state.get("broker_cap", BROKER_CAP)),
            days_per_year=int(state.get("days_per_year", DAYS_PER_YEAR)),
        )
        book.debit = {str(k): float(v) for k, v in state["debit"].items()}
        book._accounts = [str(a) for a in state["accounts"]]
        book._cost = {(str(a), str(s)): float(c) for a, s, c in state.get("cost", ())}
        book._next_oid = int(state.get("next_oid", 1))
        book.pending = [
            ForcedLiquidation(
                account=str(row["account"]),
                symbol=str(row["symbol"]),
                side=Side(row["side"]),
                qty=int(row["qty"]),
                pnl=float(row["pnl"]),
            )
            for row in state.get("pending", ())
        ]
        return book
