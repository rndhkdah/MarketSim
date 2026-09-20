"""Secondary bond market: engine MM venue and NPC holders (T6.27 / §6.11).

``GB_BILL``, ``GB_NOTE``, ``GB_BOND`` and ``CORP_POOL`` trade on the T6.08
engine MM. Marks are ``P = P_fair · exp(ξ)`` with ``ξ`` from the §6.4 impact
kernel. ADV is proportional to outstanding face.

T6.12 settlement is not landed. Agent fills and NPC transfers update the
in-memory holder book only (mark-only); they are not posted on the ledger.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from marketsim.core.config import Config
from marketsim.core.errors import ConfigError, StateError
from marketsim.market.impact import ImpactKernel
from marketsim.market.instruments import ImpactCfg, average_daily_volume
from marketsim.market.mm import MM_ACCOUNT, EngineMM, Quote
from marketsim.market.venue import Fill, Order, OrderStatus, Side, SubmitResult
from marketsim.pricing.bond_buckets import (
    BUCKET_ORDER,
    DURATION_REF_YIELD,
    GOVT_BUCKETS,
    bucket_price,
)

# master plan §6: tick = 1 day; 252 days = 1 year; 21 days = 1 month.
DAYS_PER_YEAR = 252
DAYS_PER_MONTH = 21
# §6.11 "rebalance slowly" — leak of the face-share gap per tick (1/day).
NPC_REBALANCE_GAIN = 1.0 / DAYS_PER_MONTH
# §6.4 σ_j is daily vol (decimal). Bonds have no markets.yaml σ.
DEFAULT_BOND_SIGMA = 0.01
# §6.11 INSURANCE float (share of the insurer's bond book).
INSURANCE_TARGET_SHARES: dict[str, float] = {
    "GB_BOND": 0.60,
    "CORP_POOL": 0.25,
    "GB_NOTE": 0.15,
}
# §6.11 ROW holds 10 % of government debt (face).
ROW_GOVT_SHARE = 0.10


def yield_from_price(price: float, kappa: float, decay: float) -> float:
    """Invert ``P = (κ + δ) / (y + δ)`` (§6.11). Annual decimal.

    ``price`` is the market price per unit of remaining face (dimensionless).
    ``kappa`` and ``decay`` are annual decimals (coupon and δ).
    """
    p = float(price)
    if p <= 0.0:
        raise ValueError("price must be > 0 (cr/unit)")
    return (float(kappa) + float(decay)) / p - float(decay)


def daily_holding_return(
    p_prev: float,
    p_t: float,
    kappa_annual: float,
    delta_annual: float,
    *,
    days_per_year: int = DAYS_PER_YEAR,
) -> float:
    """Daily analog of §6.11 ``(κ + δ + (1 − δ)·P_t) / P_{t−1} − 1``.

    ``p_prev`` / ``p_t`` are prices per unit face. ``kappa_annual`` and
    ``delta_annual`` are annual decimals. Returns a daily decimal.
    """
    if p_prev <= 0.0:
        raise ValueError("p_prev must be > 0 (cr/unit)")
    scale = float(days_per_year)
    k_d = float(kappa_annual) / scale
    d_d = float(delta_annual) / scale
    return float((k_d + d_d + (1.0 - d_d) * float(p_t)) / float(p_prev) - 1.0)


def gb_bond_total_return(book: BondSecondaryMarket) -> float:
    """T6.21 / gate 11 hook: last-tick ``GB_BOND`` total return (daily decimal)."""
    return book.gb_bond_total_return()


def _as_bucket_map(
    raw: float | Mapping[str, float],
    *,
    name: str,
) -> dict[str, float]:
    if isinstance(raw, Mapping):
        missing = [b for b in BUCKET_ORDER if b not in raw]
        if missing:
            raise ConfigError(f"{name} missing buckets {missing}")
        return {b: float(raw[b]) for b in BUCKET_ORDER}
    value = float(raw)
    return {b: value for b in BUCKET_ORDER}


def _holder_books(raw: Mapping[str, Mapping[str, float]] | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for holder, book in (raw or {}).items():
        out[str(holder)] = {b: float(book.get(b, 0.0)) for b in BUCKET_ORDER}
    return out


class BondSecondaryMarket:
    """Engine-MM secondary book for the four §6.11 buckets plus NPC holders."""

    def __init__(
        self,
        *,
        mm: EngineMM,
        face: Mapping[str, float],
        kappa: Mapping[str, float],
        decay: Mapping[str, float],
        fair_yield: Mapping[str, float],
        sigma: Mapping[str, float],
        turnover: float,
        holdings: Mapping[str, Mapping[str, float]] | None = None,
        targets: Mapping[str, Mapping[str, float]] | None = None,
        kernels: Mapping[str, ImpactKernel] | None = None,
        rebalance_gain: float = NPC_REBALANCE_GAIN,
        impact_cfg: ImpactCfg | None = None,
    ) -> None:
        missing = [b for b in BUCKET_ORDER if b not in face]
        if missing:
            raise ConfigError(f"face missing buckets {missing}")
        if turnover <= 0.0:
            raise ConfigError("turnover must be > 0 (1/day)")
        if rebalance_gain < 0.0:
            raise ConfigError("rebalance_gain must be >= 0 (1/tick)")
        self.mm = mm
        self.face = {b: float(face[b]) for b in BUCKET_ORDER}
        if any(self.face[b] <= 0.0 for b in BUCKET_ORDER):
            raise ConfigError("outstanding face must be > 0 (cr) for each bucket")
        self.kappa = {b: float(kappa[b]) for b in BUCKET_ORDER}
        self.decay = {b: float(decay[b]) for b in BUCKET_ORDER}
        self.fair_yield = {b: float(fair_yield[b]) for b in BUCKET_ORDER}
        self.sigma = {b: float(sigma[b]) for b in BUCKET_ORDER}
        self.turnover = float(turnover)
        self.rebalance_gain = float(rebalance_gain)
        self.holdings = _holder_books(holdings)
        self.targets = _holder_books(targets)
        cfg_i = impact_cfg if impact_cfg is not None else ImpactCfg()
        self.kernels = {b: kernels[b] if kernels is not None else ImpactKernel(cfg_i) for b in BUCKET_ORDER}
        self.prices = {b: 1.0 for b in BUCKET_ORDER}
        self._last_return = {b: 0.0 for b in BUCKET_ORDER}
        self._sides: dict[int, Side] = {}
        self._residual_mm()
        self._sync_marks()

    @classmethod
    def from_config(
        cls,
        cfg: Config,
        *,
        face: Mapping[str, float],
        holdings: Mapping[str, Mapping[str, float]] | None = None,
        targets: Mapping[str, Mapping[str, float]] | None = None,
        fair_yield: float | Mapping[str, float] = DURATION_REF_YIELD,
        sigma: float | Mapping[str, float] = DEFAULT_BOND_SIGMA,
        turnover: float | None = None,
        rebalance_gain: float | None = None,
    ) -> BondSecondaryMarket:
        """Build the four MM books from ``config/markets.yaml`` and ``config/bonds.yaml``."""
        if cfg.markets is None:
            raise ConfigError("from_config requires config/markets.yaml")
        if cfg.bonds is None:
            raise ConfigError("from_config requires config/bonds.yaml")
        y = _as_bucket_map(fair_yield, name="fair_yield")
        sig = _as_bucket_map(sigma, name="sigma")
        to = float(cfg.markets.turnover if turnover is None else turnover)
        kappa = {b: y[b] for b in BUCKET_ORDER}  # §6.11 κ fixed at SS yield so P=1 at baseline
        decay = {b: float(cfg.bonds.buckets[b].decay) for b in BUCKET_ORDER}
        mm = EngineMM(cfg.markets.mm, impact_cfg=cfg.markets.impact)
        units = {b: float(face[b]) for b in BUCKET_ORDER}
        for symbol in BUCKET_ORDER:
            p0 = bucket_price(y[symbol], kappa[symbol], decay[symbol])
            mm.add_book(
                symbol,
                mid=p0,
                sigma=sig[symbol],
                cap=units[symbol],
                adv=average_daily_volume(units[symbol], to),
            )
        return cls(
            mm=mm,
            face=units,
            kappa=kappa,
            decay=decay,
            fair_yield=y,
            sigma=sig,
            turnover=to,
            holdings=holdings,
            targets=targets,
            rebalance_gain=NPC_REBALANCE_GAIN if rebalance_gain is None else float(rebalance_gain),
            impact_cfg=cfg.markets.impact,
        )

    def symbols(self) -> tuple[str, ...]:
        """The four §6.11 buckets, in `BUCKET_ORDER`."""
        return BUCKET_ORDER

    def adv(self, symbol: str) -> float:
        """ADV (cr/day) = turnover × outstanding face (§6.11 / §6.2)."""
        self._check(symbol)
        return average_daily_volume(self.face[symbol], self.turnover)

    def price(self, symbol: str) -> float:
        """Market price per unit of remaining face (cr/unit)."""
        self._check(symbol)
        return self.prices[symbol]

    def bucket_yield(self, symbol: str) -> float:
        """Secondary yield (annual decimal) inverted from the mark (§6.11)."""
        self._check(symbol)
        return yield_from_price(self.prices[symbol], self.kappa[symbol], self.decay[symbol])

    def quote(self, symbol: str) -> Quote:
        """Engine-MM quote (cr/unit)."""
        return self.mm.quote(symbol)

    def holder_shares(self, holder: str) -> dict[str, float]:
        """Face shares of ``holder``'s bond book (dimensionless)."""
        book = self.holdings[holder]
        total = sum(book[b] for b in BUCKET_ORDER)
        if total <= 0.0:
            return {b: 0.0 for b in BUCKET_ORDER}
        return {b: book[b] / total for b in BUCKET_ORDER}

    def total_return(self, symbol: str) -> float:
        """Last-tick holding return (daily decimal)."""
        self._check(symbol)
        return self._last_return[symbol]

    def gb_bond_total_return(self) -> float:
        """Last-tick ``GB_BOND`` total return (daily decimal). T6.21 / gate 11."""
        return self.total_return("GB_BOND")

    def set_face(self, symbol: str, face: float) -> None:
        """Update outstanding face (cr). ADV scales one-for-one."""
        self._check(symbol)
        if face <= 0.0:
            raise ConfigError("outstanding face must be > 0 (cr)")
        self.face[symbol] = float(face)
        self._residual_mm()
        self.mm.set_mark(symbol, cap=self.face[symbol], adv=self.adv(symbol))

    def set_fair_yield(self, symbol: str, y: float) -> None:
        """Set the bucket fair yield (annual decimal). Call ``publish_marks`` to refresh ``P``."""
        self._check(symbol)
        if y < 0.0:
            raise ValueError("fair yield must be >= 0 (annual decimal)")
        self.fair_yield[symbol] = float(y)

    def publish_marks(self) -> None:
        """``P = P_fair · exp(ξ)`` and push the mid to the engine MM (§6.11)."""
        self._sync_marks()

    def official_flow(
        self,
        src: str,
        dst: str,
        symbol: str,
        face: float,
        *,
        signed_notional: float,
    ) -> None:
        """Move ``face`` units ``src`` → ``dst`` and hit ξ with ``signed_notional`` (cr).

        T6.28 QE / QT / OMO use this instead of the participation-capped MM match
        so a purchase of ``x`` is the four §6.11 postings plus one kernel step.
        Holder books stay mark-only (T6.12 settlement is a separate card).
        """
        self._check(symbol)
        qty = float(face)
        if qty <= 0.0:
            raise ValueError("official flow face must be > 0 (units)")
        if src == dst:
            raise ValueError("official flow src and dst must differ")
        self._ensure_holder(src)
        self._ensure_holder(dst)
        self.holdings[src][symbol] -= qty
        self.holdings[dst][symbol] += qty
        self.kernels[symbol].step(float(signed_notional), self.adv(symbol), self.sigma[symbol])
        self._sync_marks()

    def apply_row_selloff(self, fraction: float) -> None:
        """§6.11 foreign sell-off: ROW sells ``fraction`` of each government bucket.

        Face moves ROW → MM (mark-only). Signed volume hits the impact kernel
        so secondary yields rise. ``fraction`` is dimensionless in ``[0, 1]``.
        """
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("sell-off fraction must be in [0, 1]")
        if "ROW" not in self.holdings:
            self.holdings["ROW"] = {b: 0.0 for b in BUCKET_ORDER}
        p_prev = dict(self.prices)
        row = self.holdings["ROW"]
        for symbol in GOVT_BUCKETS:
            sold = float(fraction) * row[symbol]
            if sold <= 0.0:
                continue
            row[symbol] -= sold
            self.holdings[MM_ACCOUNT][symbol] += sold
            q = -sold * self.prices[symbol]
            self.kernels[symbol].step(q, self.adv(symbol), self.sigma[symbol])
        self._sync_marks()
        self._record_returns(p_prev)

    def submit(self, order: Order) -> SubmitResult:
        """Park an agent order on the engine MM (next-tick fill)."""
        res = self.mm.submit(order)
        if res.order_id is not None and res.status is not OrderStatus.REJECTED:
            self._sides[int(res.order_id)] = order.side
        return res

    def ingest(self, orders: Sequence[Order]) -> list[SubmitResult]:
        out: list[SubmitResult] = []
        for order in orders:
            out.append(self.submit(order))
        return out

    def cancel(self, symbol: str, order_id: int) -> SubmitResult:
        return self.mm.cancel(symbol, order_id)

    def replace(
        self,
        symbol: str,
        order_id: int,
        *,
        qty: int | None = None,
        price: float | None = None,
    ) -> SubmitResult:
        return self.mm.replace(symbol, order_id, qty=qty, price=price)

    def step(self, orders: Sequence[Order] = ()) -> tuple[Fill, ...]:
        """Rebalance NPC holders, match parked / new orders, mark from ``ξ``."""
        if orders:
            self.ingest(orders)
        extra = self._rebalance()
        return self.end_tick(extra_q=extra)

    def end_tick(self, extra_q: Mapping[str, float] | None = None) -> tuple[Fill, ...]:
        """Match the MM book, fold extra signed volume (cr) into ``ξ``, publish marks."""
        mid_pre = dict(self.prices)
        fills = self.mm.end_tick()
        q = self._signed_q(fills, mid_pre)
        extra = extra_q or {}
        for symbol in BUCKET_ORDER:
            qn = q[symbol] + float(extra.get(symbol, 0.0))
            self.kernels[symbol].step(qn, self.adv(symbol), self.sigma[symbol])
        self._sync_marks()
        self._record_returns(mid_pre)
        return fills

    def to_state(self) -> dict[str, Any]:
        return {
            "mm": self.mm.to_state(),
            "face": dict(self.face),
            "kappa": dict(self.kappa),
            "decay": dict(self.decay),
            "fair_yield": dict(self.fair_yield),
            "sigma": dict(self.sigma),
            "turnover": self.turnover,
            "rebalance_gain": self.rebalance_gain,
            "holdings": {h: dict(self.holdings[h]) for h in sorted(self.holdings)},
            "targets": {h: dict(self.targets[h]) for h in sorted(self.targets)},
            "kernels": {b: self.kernels[b].to_state() for b in BUCKET_ORDER},
            "prices": dict(self.prices),
            "last_return": dict(self._last_return),
            "sides": {str(i): str(s) for i, s in sorted(self._sides.items())},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> BondSecondaryMarket:
        needed = ("mm", "face", "kappa", "decay", "fair_yield", "sigma", "turnover", "kernels")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"bond secondary state missing {missing}")
        obj = cls(
            mm=EngineMM.from_state(state["mm"]),
            face=state["face"],
            kappa=state["kappa"],
            decay=state["decay"],
            fair_yield=state["fair_yield"],
            sigma=state["sigma"],
            turnover=float(state["turnover"]),
            holdings=state.get("holdings"),
            targets=state.get("targets"),
            kernels={b: ImpactKernel.from_state(state["kernels"][b]) for b in BUCKET_ORDER},
            rebalance_gain=float(state.get("rebalance_gain", NPC_REBALANCE_GAIN)),
        )
        obj.prices = {b: float(state["prices"][b]) for b in BUCKET_ORDER} if "prices" in state else obj.prices
        obj._last_return = (
            {b: float(state["last_return"][b]) for b in BUCKET_ORDER}
            if "last_return" in state
            else obj._last_return
        )
        obj._sides = {int(i): Side(s) for i, s in state.get("sides", {}).items()}
        return obj

    def _check(self, symbol: str) -> None:
        if symbol not in self.face:
            raise ConfigError(f"unknown bucket {symbol!r}")

    def _fair_price(self, symbol: str) -> float:
        return bucket_price(self.fair_yield[symbol], self.kappa[symbol], self.decay[symbol])

    def _residual_mm(self) -> None:
        if MM_ACCOUNT not in self.holdings:
            self.holdings[MM_ACCOUNT] = {b: 0.0 for b in BUCKET_ORDER}
        for symbol in BUCKET_ORDER:
            others = sum(self.holdings[h][symbol] for h in self.holdings if h != MM_ACCOUNT)
            self.holdings[MM_ACCOUNT][symbol] = self.face[symbol] - others

    def _sync_marks(self) -> None:
        for symbol in BUCKET_ORDER:
            xi = float(self.kernels[symbol].xi)
            price = self._fair_price(symbol) * math.exp(xi)
            self.prices[symbol] = price
            self.mm.set_mark(symbol, mid=price, adv=self.adv(symbol), cap=self.face[symbol])

    def _record_returns(self, p_prev: Mapping[str, float]) -> None:
        for symbol in BUCKET_ORDER:
            self._last_return[symbol] = daily_holding_return(
                p_prev[symbol],
                self.prices[symbol],
                self.kappa[symbol],
                self.decay[symbol],
            )

    def _ensure_holder(self, name: str) -> None:
        if name not in self.holdings:
            self.holdings[name] = {b: 0.0 for b in BUCKET_ORDER}

    def _signed_q(self, fills: Sequence[Fill], mid_pre: Mapping[str, float]) -> dict[str, float]:
        q = {b: 0.0 for b in BUCKET_ORDER}
        for fill in fills:
            side = self._sides.get(fill.taker_order_id)
            if side is None:
                side = Side.BUY if fill.price >= mid_pre[fill.symbol] else Side.SELL
            sign = 1.0 if side is Side.BUY else -1.0
            q[fill.symbol] += sign * float(fill.qty) * float(mid_pre[fill.symbol])
            self._apply_fill(fill, side)
        return q

    def _apply_fill(self, fill: Fill, side: Side) -> None:
        # Mark-only holder book — T6.12 would post the cash-for-bond Tx.
        self._ensure_holder(fill.taker)
        qty = float(fill.qty)
        if side is Side.BUY:
            self.holdings[fill.taker][fill.symbol] += qty
            self.holdings[MM_ACCOUNT][fill.symbol] -= qty
        else:
            self.holdings[fill.taker][fill.symbol] -= qty
            self.holdings[MM_ACCOUNT][fill.symbol] += qty

    def _rebalance(self) -> dict[str, float]:
        """Slow face-share leak toward NPC targets (§6.11). Signed volume in cr."""
        q = {b: 0.0 for b in BUCKET_ORDER}
        cap_frac = float(self.mm.cfg.participation_cap)  # §6.5 budget / ADV
        for holder in sorted(self.targets):
            if holder == MM_ACCOUNT:
                continue
            book = self.holdings.get(holder)
            if book is None:
                continue
            total = sum(book[b] for b in BUCKET_ORDER)
            if total <= 0.0:
                continue
            shares = self.targets[holder]
            for symbol in BUCKET_ORDER:
                desired = float(shares.get(symbol, 0.0)) * total
                raw = self.rebalance_gain * (desired - book[symbol])
                px = self.prices[symbol]
                cap_face = cap_frac * self.adv(symbol) / px if px > 0.0 else 0.0
                trade = max(-cap_face, min(cap_face, raw))
                if abs(trade) <= 0.0:
                    continue
                book[symbol] += trade
                self.holdings[MM_ACCOUNT][symbol] -= trade
                q[symbol] += trade * px
        return q
