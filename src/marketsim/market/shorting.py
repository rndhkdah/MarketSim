"""Stock-loan locates and utilisation-based borrow fees (T6.13 / §6.8).

Shorts are allowed on NPC sector equity and commodities. ``EQ:FIRM:*`` is not
lendable by default. The borrow fee is an annual decimal that rises with
utilisation ``short_interest / lendable_longs`` (issuer books excluded) and is
posted daily as ``fees`` from each short to ``MM``. Never clipped.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marketsim.core.errors import StateError
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.market.instruments import COMMODITY_SECTORS, FIRM_PREFIX, NPC_PREFIX
from marketsim.market.mm import MM_ACCOUNT

# Master plan §6 / ``core.calendar``: 252 ticks / year.
DAYS_PER_YEAR = 252
# §6.8: fee rises with utilisation. Not a ``markets.yaml`` key (T6.01 haircuts only).
FEE0 = 0.01  # annual decimal at utilisation 0
KAPPA = 0.05  # annual decimal; extra loading as utilisation → 1
# Keeps ``u / (1 − u)`` finite at full utilisation (dimensionless).
U_FLOOR = 1e-6
_EPS = 1e-15
FEE_TAG = "fees"
CASH = "DEP"


def is_lendable(symbol: str, *, firm_shares_lendable: bool = False) -> bool:
    """Whether ``symbol`` can be borrowed for a short (§6.8).

    NPC sector equity and the three cash-settled commodities are lendable.
    Agent-firm shares (``EQ:FIRM:*``) are not, unless ``firm_shares_lendable``.
    """
    if symbol.startswith(FIRM_PREFIX):
        return bool(firm_shares_lendable)
    if symbol.startswith(NPC_PREFIX):
        return True
    return symbol in COMMODITY_SECTORS


def is_issuer(entity: str, symbol: str) -> bool:
    """Structural short (issuer book) — not stock-loan inventory or short interest."""
    if symbol.startswith(NPC_PREFIX):
        code = symbol[len(NPC_PREFIX) :]
        return entity.startswith("NPC:") and entity.endswith(f":{code}")
    if symbol.startswith(FIRM_PREFIX):
        return entity == f"FIRM:{symbol[len(FIRM_PREFIX) :]}"
    return False


def investor_books(ledger: Ledger, symbol: str) -> tuple[float, float]:
    """``(lendable_long, short_interest)`` in units, excluding the issuer.

    Both legs are ≥ 0. Empty if ``symbol`` is not on the ledger.
    """
    if symbol not in ledger.instruments:
        return 0.0, 0.0
    long = 0.0
    short = 0.0
    for name in ledger.entities.names:
        if is_issuer(name, symbol):
            continue
        q = ledger.position(name, symbol)
        if q > _EPS:
            long += q
        elif q < -_EPS:
            short += -q
    return long, short


def utilisation(ledger: Ledger, symbol: str) -> float:
    """Short interest / lendable longs, in ``[0, 1]``. Dimensionless.

    Full utilisation (no longs, positive shorts) is ``1``.
    """
    long, short = investor_books(ledger, symbol)
    if short <= _EPS:
        return 0.0
    if long <= _EPS:
        return 1.0
    u = short / long
    if u < 0.0:
        return 0.0
    if u > 1.0:
        return 1.0
    return float(u)


def borrow_fee_annual(
    u: float,
    *,
    fee0: float = FEE0,
    kappa: float = KAPPA,
) -> float:
    """Annual borrow fee (decimal) rising with utilisation ``u`` (§6.8).

    ``fee0 + kappa · u / (1 − u)`` with a floor on the denominator so the
    fee stays finite at ``u = 1``. Monotone in ``u``.
    """
    util = float(u)
    if util < 0.0:
        util = 0.0
    if util > 1.0:
        util = 1.0
    return float(fee0) + float(kappa) * util / (1.0 - util + U_FLOOR)


def locate(
    symbol: str,
    qty: float,
    ledger: Ledger,
    *,
    firm_shares_lendable: bool = False,
) -> bool:
    """True if ``qty`` units of ``symbol`` can be borrowed from investor longs."""
    if not is_lendable(symbol, firm_shares_lendable=firm_shares_lendable):
        return False
    need = float(qty)
    if need <= _EPS:
        return True
    long, short = investor_books(ledger, symbol)
    return (long - short) + _EPS >= need


def accrue_borrow_fee(
    ledger: Ledger,
    symbol: str,
    price: float,
    tick: int,
    *,
    fee0: float = FEE0,
    kappa: float = KAPPA,
    firm_shares_lendable: bool = False,
    cash: str = CASH,
    to: str = MM_ACCOUNT,
    days_per_year: int = DAYS_PER_YEAR,
) -> tuple[Tx, ...]:
    """Post one day's borrow fee on ``symbol`` (cr). Tag ``fees``.

    Each short pays ``qty · price · fee_annual(u) / days_per_year`` to ``to``
    (the broker). One bilateral ``Tx`` per short so the TFM stays 1–1.
    ``price`` is cr/unit. Returns the posted txs (possibly empty).
    """
    if not is_lendable(symbol, firm_shares_lendable=firm_shares_lendable):
        return ()
    if symbol not in ledger.instruments or float(price) < 0.0:
        return ()
    u = utilisation(ledger, symbol)
    daily = borrow_fee_annual(u, fee0=fee0, kappa=kappa) / float(days_per_year)
    if abs(daily) < _EPS:
        return ()
    if to not in ledger.entities:
        ledger.register_entity(to)
    posted: list[Tx] = []
    for name in ledger.entities.names:
        if name == to or is_issuer(name, symbol):
            continue
        q = ledger.position(name, symbol)
        if q >= -_EPS:
            continue
        fee = (-q) * float(price) * daily  # cr; short qty is negative
        if abs(fee) < _EPS or name == to:
            continue
        tx = Tx(
            tick,
            FEE_TAG,
            (Entry(name, cash, -fee), Entry(to, cash, fee)),
            memo=f"borrow {symbol}",
        )
        ledger.post(tx)
        posted.append(tx)
    return tuple(posted)


class StockLoanBook:
    """Per-book borrow-fee parameters. Locates and utilisation read the ledger."""

    def __init__(
        self,
        *,
        fee0: float = FEE0,
        kappa: float = KAPPA,
        firm_shares_lendable: bool = False,
        days_per_year: int = DAYS_PER_YEAR,
    ) -> None:
        self.fee0 = float(fee0)
        self.kappa = float(kappa)
        self.firm_shares_lendable = bool(firm_shares_lendable)
        self.days_per_year = int(days_per_year)

    def locate(self, symbol: str, qty: float, ledger: Ledger) -> bool:
        """See :func:`locate`."""
        return locate(
            symbol,
            qty,
            ledger,
            firm_shares_lendable=self.firm_shares_lendable,
        )

    def utilisation(self, ledger: Ledger, symbol: str) -> float:
        """See :func:`utilisation`."""
        return utilisation(ledger, symbol)

    def accrue(
        self,
        ledger: Ledger,
        prices: Mapping[str, float],
        tick: int,
    ) -> tuple[Tx, ...]:
        """Daily fee on every priced lendable name. ``prices`` are cr/unit."""
        posted: list[Tx] = []
        for symbol in sorted(prices):
            posted.extend(
                accrue_borrow_fee(
                    ledger,
                    symbol,
                    float(prices[symbol]),
                    tick,
                    fee0=self.fee0,
                    kappa=self.kappa,
                    firm_shares_lendable=self.firm_shares_lendable,
                    days_per_year=self.days_per_year,
                )
            )
        return tuple(posted)

    def to_state(self) -> dict[str, Any]:
        return {
            "fee0": self.fee0,
            "kappa": self.kappa,
            "firm_shares_lendable": self.firm_shares_lendable,
            "days_per_year": self.days_per_year,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> StockLoanBook:
        needed = ("fee0", "kappa", "firm_shares_lendable", "days_per_year")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"stock-loan state missing {missing}")
        return cls(
            fee0=float(state["fee0"]),
            kappa=float(state["kappa"]),
            firm_shares_lendable=bool(state["firm_shares_lendable"]),
            days_per_year=int(state["days_per_year"]),
        )
