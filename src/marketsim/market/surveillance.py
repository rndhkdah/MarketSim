"""Wash, circular-trade and pump surveillance (T6.20 / §6.10).

Self-cross is blocked on the CLOB (cancel newest). This module flags the tape:
a same-tick (or ``wash_window_ticks``) two-way A↔B, a directed cycle of length
≥ 3 inside ``circular_n`` ticks, and an agent whose **buy** volume share exceeds
``pump_volume_share`` during a ``pump_window_d``-day run-up above ``pump_runup``.
Honest two-sided market making does not match those patterns.

``Fill`` has no side; ``taker_side`` is the aggressor's buy/sell (same split as
T6.12 settlement). Units: ``qty`` shares; ``price`` cr/share; windows in ticks
(1 tick = 1 trading day).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from marketsim.core.errors import StateError
from marketsim.market.clob import Fill, Side
from marketsim.market.instruments import SurveillanceCfg


class FlagKind(StrEnum):
    WASH = "wash"
    CIRCULAR = "circular"
    PUMP = "pump"


@dataclass(frozen=True, slots=True)
class Flag:
    """One surveillance hit. ``agents`` are sorted (wash / pump) or cycle-canon (circular)."""

    kind: FlagKind
    symbol: str
    agents: tuple[str, ...]
    tick: int
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _Print:
    tick: int
    symbol: str
    buyer: str
    seller: str
    qty: int
    price: float
    maker: str
    taker: str


def counterparties(fill: Fill, taker_side: Side) -> tuple[str, str]:
    """Buyer and seller agent ids. ``taker_side`` is the aggressor's buy/sell."""
    if taker_side is Side.BUY:
        return fill.taker, fill.maker
    return fill.maker, fill.taker


def _canon_cycle(path: tuple[str, ...]) -> tuple[str, ...]:
    rots = tuple(path[i:] + path[:i] for i in range(len(path)))
    return min(rots)


def _directed_cycles(edges: set[tuple[str, str]], *, min_len: int = 3) -> list[tuple[str, ...]]:
    adj: dict[str, set[str]] = {}
    for src, dst in edges:
        if src == dst:
            continue
        adj.setdefault(src, set()).add(dst)
    found: set[tuple[str, ...]] = set()

    def dfs(start: str, node: str, path: tuple[str, ...], seen: set[str]) -> None:
        for nxt in sorted(adj.get(node, ())):
            if nxt == start and len(path) >= min_len:
                found.add(_canon_cycle(path))
            elif nxt not in seen:
                dfs(start, nxt, path + (nxt,), seen | {nxt})

    for start in sorted(adj):
        dfs(start, start, (start,), {start})
    return sorted(found)


def _two_way_pairs(edges: set[tuple[str, str]]) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for src, dst in edges:
        if src == dst:
            continue
        if (dst, src) in edges:
            pairs.add((src, dst) if src < dst else (dst, src))
    return sorted(pairs)


class Surveillance:
    """Streaming tape scanner. Windows are inclusive ``[now − n + 1, now]``."""

    def __init__(self, cfg: SurveillanceCfg | None = None) -> None:
        self.cfg = cfg if cfg is not None else SurveillanceCfg()
        self.clock_tick = 0
        self._prints: list[_Print] = []
        self._px: dict[str, dict[int, float]] = {}

    def step(
        self,
        fills: Sequence[Fill] = (),
        taker_sides: Sequence[Side] = (),
        *,
        prices: Mapping[str, float] | None = None,
        tick: int | None = None,
    ) -> tuple[Flag, ...]:
        """Record this batch and return flags active at ``tick`` (or max fill tick).

        ``taker_sides`` aligns 1-1 with ``fills``. ``prices`` are cr/share marks
        written at ``now`` (override last fill). Returns flags, never mutates fills.
        """
        if len(fills) != len(taker_sides):
            raise ValueError("taker_sides must match fills")
        now = int(tick) if tick is not None else self.clock_tick
        if fills:
            now = max(now, max(int(f.tick) for f in fills))
        for fill, side in zip(fills, taker_sides, strict=True):
            self._record(fill, Side(side))
        if prices is not None:
            for symbol in sorted(prices):
                self._px.setdefault(str(symbol), {})[now] = float(prices[symbol])
        self.clock_tick = now
        self._trim(now)
        return self._detect(now)

    def to_state(self) -> dict[str, Any]:
        return {
            "cfg": {
                "wash_window_ticks": self.cfg.wash_window_ticks,
                "circular_n": self.cfg.circular_n,
                "pump_volume_share": self.cfg.pump_volume_share,
                "pump_runup": self.cfg.pump_runup,
                "pump_window_d": self.cfg.pump_window_d,
            },
            "clock_tick": self.clock_tick,
            "prints": [
                {
                    "tick": p.tick,
                    "symbol": p.symbol,
                    "buyer": p.buyer,
                    "seller": p.seller,
                    "qty": p.qty,
                    "price": p.price,
                    "maker": p.maker,
                    "taker": p.taker,
                }
                for p in self._prints
            ],
            "prices": {
                symbol: {str(t): px for t, px in sorted(marks.items())}
                for symbol, marks in ((s, self._px[s]) for s in sorted(self._px))
            },
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Surveillance:
        needed = ("cfg", "clock_tick", "prints", "prices")
        missing = [k for k in needed if k not in state]
        if missing:
            raise StateError(f"surveillance state missing {missing}")
        obj = cls(SurveillanceCfg(**state["cfg"]))
        obj.clock_tick = int(state["clock_tick"])
        obj._prints = [
            _Print(
                tick=int(row["tick"]),
                symbol=str(row["symbol"]),
                buyer=str(row["buyer"]),
                seller=str(row["seller"]),
                qty=int(row["qty"]),
                price=float(row["price"]),
                maker=str(row["maker"]),
                taker=str(row["taker"]),
            )
            for row in state["prints"]
        ]
        px: dict[str, dict[int, float]] = {}
        raw_px = state["prices"]
        for symbol in sorted(raw_px):
            marks = raw_px[symbol]
            px[str(symbol)] = {int(t): float(marks[t]) for t in marks}
        obj._px = px
        return obj

    def _record(self, fill: Fill, taker_side: Side) -> None:
        qty = int(fill.qty)
        if qty <= 0:
            return
        buyer, seller = counterparties(fill, taker_side)
        t = int(fill.tick)
        self._prints.append(
            _Print(
                tick=t,
                symbol=str(fill.symbol),
                buyer=buyer,
                seller=seller,
                qty=qty,
                price=float(fill.price),
                maker=str(fill.maker),
                taker=str(fill.taker),
            )
        )
        self._px.setdefault(str(fill.symbol), {})[t] = float(fill.price)

    def _horizon(self) -> int:
        return max(self.cfg.wash_window_ticks, self.cfg.circular_n, self.cfg.pump_window_d)

    def _trim(self, now: int) -> None:
        lo = now - self._horizon() + 1
        self._prints = [p for p in self._prints if p.tick >= lo]
        for symbol in list(self._px):
            kept = {t: px for t, px in self._px[symbol].items() if t >= lo}
            if kept:
                self._px[symbol] = kept
            else:
                del self._px[symbol]

    def _in_window(self, now: int, width: int) -> list[_Print]:
        lo = now - int(width) + 1
        return [p for p in self._prints if lo <= p.tick <= now]

    def _detect(self, now: int) -> tuple[Flag, ...]:
        flags: list[Flag] = []
        flags.extend(self._wash(now))
        flags.extend(self._circular(now))
        flags.extend(self._pump(now))
        flags.sort(key=lambda f: (str(f.kind), f.symbol, f.agents, f.tick, f.detail))
        return tuple(flags)

    def _wash(self, now: int) -> list[Flag]:
        rows = self._in_window(now, self.cfg.wash_window_ticks)
        out: list[Flag] = []
        by_sym: dict[str, list[_Print]] = {}
        for p in rows:
            by_sym.setdefault(p.symbol, []).append(p)
        for symbol in sorted(by_sym):
            group = by_sym[symbol]
            self_agents = sorted({p.buyer for p in group if p.buyer == p.seller or p.maker == p.taker})
            for agent in self_agents:
                out.append(
                    Flag(FlagKind.WASH, symbol, (agent,), now, detail="self-cross")
                )
            edges = {(p.seller, p.buyer) for p in group}
            for a, b in _two_way_pairs(edges):
                out.append(
                    Flag(FlagKind.WASH, symbol, (a, b), now, detail="two-way")
                )
        return out

    def _circular(self, now: int) -> list[Flag]:
        rows = self._in_window(now, self.cfg.circular_n)
        out: list[Flag] = []
        by_sym: dict[str, list[_Print]] = {}
        for p in rows:
            by_sym.setdefault(p.symbol, []).append(p)
        for symbol in sorted(by_sym):
            edges = {(p.seller, p.buyer) for p in by_sym[symbol]}
            for cycle in _directed_cycles(edges, min_len=3):
                out.append(
                    Flag(
                        FlagKind.CIRCULAR,
                        symbol,
                        cycle,
                        now,
                        detail=f"n={len(cycle)}",
                    )
                )
        return out

    def _runup(self, symbol: str, now: int) -> float | None:
        lo = now - self.cfg.pump_window_d + 1
        series = sorted((t, px) for t, px in self._px.get(symbol, {}).items() if lo <= t <= now)
        if len(series) < 2:
            return None
        p0 = float(series[0][1])
        p1 = float(series[-1][1])
        if p0 <= 0.0:
            return None
        return p1 / p0 - 1.0

    def _pump(self, now: int) -> list[Flag]:
        rows = self._in_window(now, self.cfg.pump_window_d)
        out: list[Flag] = []
        symbols = sorted({p.symbol for p in rows} | set(self._px))
        for symbol in symbols:
            runup = self._runup(symbol, now)
            if runup is None or runup <= float(self.cfg.pump_runup):
                continue
            group = [p for p in rows if p.symbol == symbol]
            total = sum(p.qty for p in group)
            if total <= 0:
                continue
            bought: dict[str, int] = {}
            for p in group:
                bought[p.buyer] = bought.get(p.buyer, 0) + p.qty
            thresh = float(self.cfg.pump_volume_share)
            for agent in sorted(bought):
                share = bought[agent] / float(total)
                if share > thresh:
                    out.append(
                        Flag(
                            FlagKind.PUMP,
                            symbol,
                            (agent,),
                            now,
                            detail=f"share={share:.4f} runup={runup:.4f}",
                        )
                    )
        return out
