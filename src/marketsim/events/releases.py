"""T4.07 — data-release calendar and published vintages (§4.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marketsim.core.calendar import Calendar
from marketsim.core.module import Phase
from marketsim.core.rng import RngHub
from marketsim.layer1.build_io import CODES

# §4.3 calendar (1-based day of month). tick = 1 simulated day.
CPI_DAY = 10
UNEMPLOYMENT_DAY = 5
OUTPUT_DAY = 15
GDP_RELEASE_DAY = 21  # month after quarter end
GDP_NOISE_SD = 0.003  # 0.3 % of level; halves at each revision
N_GDP_REVISIONS = 2  # +1 and +2 months after the first release
DAYS_PER_MONTH = 21
MONTHS_PER_YEAR = 12
PUBLIC_OBS_KEY = "*"


def month_end_tick(month: int, days_per_month: int = DAYS_PER_MONTH) -> int:
    """Tick of day 21 of 0-based month ``month``."""
    return int(month) * days_per_month + (days_per_month - 1)


def following_month_day_tick(month: int, day: int, days_per_month: int = DAYS_PER_MONTH) -> int:
    """Tick of 1-based ``day`` in the month after 0-based ``month``."""
    return (int(month) + 1) * days_per_month + (int(day) - 1)


def quarter_index(month: int) -> int:
    return int(month) // 3


def is_quarter_end_month(month: int) -> bool:
    return (int(month) % 3) == 2


def gdp_release_tick(quarter: int, revision: int, days_per_month: int = DAYS_PER_MONTH) -> int:
    """First GDP print: day 21 of the month after quarter end; then +1 / +2 months."""
    # Quarter q ends at month 3q+2; month after that is 3q+3.
    month = 3 * int(quarter) + 3 + int(revision)
    return month * days_per_month + (GDP_RELEASE_DAY - 1)


def policy_decision_tick(quarter: int, days_per_month: int = DAYS_PER_MONTH) -> int:
    """Quarter-end day 21 of month 3q+2."""
    month = 3 * int(quarter) + 2
    return month_end_tick(month, days_per_month)


@dataclass
class Vintage:
    """One published print. ``period`` is ``m:{month}`` or ``q:{quarter}`` (0-based)."""

    series: str
    period: str
    value: Any
    released_tick: int
    revision: int = 0


@dataclass
class DataReleases:
    """Scheduled publications. ``observe()`` sees only ``visible`` vintages, never truth."""

    name: str = "data_releases"
    calendar: Calendar = field(default_factory=Calendar)
    codes: tuple[str, ...] = CODES
    vintages: list[Vintage] = field(default_factory=list)
    _month_truth: dict[int, dict[str, Any]] = field(default_factory=dict)
    _quarter_gdp_truth: dict[int, float] = field(default_factory=dict)
    _quarter_gdp_eps: dict[int, float] = field(default_factory=dict)
    _quarter_trade: dict[int, float] = field(default_factory=dict)
    _quarter_govt: dict[int, float] = field(default_factory=dict)
    _gdp_acc: list[float] = field(default_factory=list)
    _trade_acc: list[float] = field(default_factory=list)
    _govt_acc: list[float] = field(default_factory=list)
    _pending: list[tuple[int, Vintage]] = field(default_factory=list)
    _visible: dict[str, Any] = field(default_factory=dict)

    def record_month(
        self,
        month: int,
        *,
        cpi: float,
        unemployment: float,
        output: np.ndarray | list[float],
        gdp: float,
        trade_balance: float,
        govt_balance: float,
        policy_rate: float,
        month_end_tick: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Store month-``month`` truth and schedule the §4.3 publications."""
        out = np.asarray(output, dtype=float)
        self._month_truth[int(month)] = {
            "cpi": float(cpi),
            "unemployment": float(unemployment),
            "output": out.tolist(),
            "gdp": float(gdp),
            "trade_balance": float(trade_balance),
            "govt_balance": float(govt_balance),
            "policy_rate": float(policy_rate),
            "month_end_tick": int(month_end_tick),
        }
        dpm = self.calendar.days_per_month
        self._schedule(
            following_month_day_tick(month, CPI_DAY, dpm),
            Vintage("cpi", f"m:{month}", float(cpi), 0, 0),
        )
        self._schedule(
            following_month_day_tick(month, UNEMPLOYMENT_DAY, dpm),
            Vintage("unemployment", f"m:{month}", float(unemployment), 0, 0),
        )
        output_map = {self.codes[i]: float(out[i]) for i in range(min(len(self.codes), out.size))}
        self._schedule(
            following_month_day_tick(month, OUTPUT_DAY, dpm),
            Vintage("output", f"m:{month}", output_map, 0, 0),
        )

        self._gdp_acc.append(float(gdp))
        self._trade_acc.append(float(trade_balance))
        self._govt_acc.append(float(govt_balance))
        if is_quarter_end_month(month):
            q = quarter_index(month)
            gdp_truth = float(sum(self._gdp_acc))
            trade_q = float(sum(self._trade_acc))
            govt_q = float(sum(self._govt_acc))
            self._gdp_acc.clear()
            self._trade_acc.clear()
            self._govt_acc.clear()
            self._quarter_gdp_truth[q] = gdp_truth
            self._quarter_trade[q] = trade_q
            self._quarter_govt[q] = govt_q
            eps = float(rng.normal(0.0, GDP_NOISE_SD)) if rng is not None else 0.0
            self._quarter_gdp_eps[q] = eps
            for rev in range(N_GDP_REVISIONS + 1):
                print_v = gdp_truth * (1.0 + eps / (2.0**rev))
                self._schedule(
                    gdp_release_tick(q, rev, dpm),
                    Vintage("gdp", f"q:{q}", print_v, 0, rev),
                )
            self._schedule(
                gdp_release_tick(q, 0, dpm),
                Vintage("trade_balance", f"q:{q}", trade_q, 0, 0),
            )
            self._schedule(
                gdp_release_tick(q, 0, dpm),
                Vintage("govt_balance", f"q:{q}", govt_q, 0, 0),
            )
            self._schedule(
                policy_decision_tick(q, dpm),
                Vintage("policy_decision", f"q:{q}", float(policy_rate), 0, 0),
            )

    def _schedule(self, tick: int, vintage: Vintage) -> None:
        vintage.released_tick = int(tick)
        self._pending.append((int(tick), vintage))

    def publish_due(self, tick: int) -> list[Vintage]:
        """Move due pending prints into the published vintage table."""
        due: list[Vintage] = []
        keep: list[tuple[int, Vintage]] = []
        for ts, vin in self._pending:
            if ts <= int(tick):
                vin.released_tick = ts
                self.vintages.append(vin)
                due.append(vin)
            else:
                keep.append((ts, vin))
        self._pending = keep
        self._rebuild_visible(int(tick))
        return due

    def _rebuild_visible(self, tick: int) -> None:
        latest: dict[str, Vintage] = {}
        for vin in self.vintages:
            if vin.released_tick > tick:
                continue
            key = f"{vin.series}:{vin.period}"
            prev = latest.get(key)
            if prev is None or vin.revision >= prev.revision:
                latest[key] = vin
        out: dict[str, Any] = {}
        for vin in latest.values():
            out.setdefault(vin.series, {})[vin.period] = vin.value
            out[vin.series + "_latest"] = vin.value
        self._visible = out

    def visible(self, tick: int) -> dict[str, Any]:
        """Published vintages with ``released_tick <= tick``. Never includes unpublished truth."""
        self.publish_due(tick)
        return dict(self._visible)

    def latest(self, series: str, tick: int) -> Any:
        vis = self.visible(tick)
        return vis.get(series + "_latest")

    def gdp_vintages(self, quarter: int) -> list[Vintage]:
        return [v for v in self.vintages if v.series == "gdp" and v.period == f"q:{quarter}"]

    def reset(self, ctx: Any) -> None:
        del ctx
        self.vintages.clear()
        self._month_truth.clear()
        self._quarter_gdp_truth.clear()
        self._quarter_gdp_eps.clear()
        self._quarter_trade.clear()
        self._quarter_govt.clear()
        self._gdp_acc.clear()
        self._trade_acc.clear()
        self._govt_acc.clear()
        self._pending.clear()
        self._visible = {}

    def on_phase(self, ctx: Any, phase: Phase) -> None:
        eco = _real_economy(ctx)
        if phase is Phase.REAL and ctx.world.clock.calendar.is_month_end(ctx.tick) and eco is not None:
            agg = eco.last_agg
            if agg is None:
                return
            rng = ctx.world.rng.stream("releases")
            self.codes = tuple(eco.codes)
            self.record_month(
                max(eco.month - 1, 0),
                cpi=float(agg.cpi),
                unemployment=float(agg.u),
                output=np.asarray(agg.x, dtype=float),
                gdp=float(agg.gdp_prod_real),
                trade_balance=float(agg.trade_balance),
                govt_balance=float(agg.govt_balance),
                policy_rate=float(eco.cb.r),
                month_end_tick=int(ctx.tick),
                rng=rng,
            )
        if phase is Phase.PUBLISH:
            vis = self.visible(ctx.tick)
            ctx.world._observations[PUBLIC_OBS_KEY] = {"releases": vis}

    def to_state(self) -> dict[str, Any]:
        return {
            "vintages": [v.__dict__ for v in self.vintages],
            "month_truth": {str(k): dict(val) for k, val in self._month_truth.items()},
            "quarter_gdp_truth": {str(k): v for k, v in self._quarter_gdp_truth.items()},
            "quarter_gdp_eps": {str(k): v for k, v in self._quarter_gdp_eps.items()},
            "pending": [{"tick": t, **vin.__dict__} for t, vin in self._pending],
            "gdp_acc": list(self._gdp_acc),
            "trade_acc": list(self._trade_acc),
            "govt_acc": list(self._govt_acc),
        }

    def from_state(self, state: dict[str, Any]) -> None:
        self.vintages = [Vintage(**raw) for raw in state.get("vintages", [])]
        self._month_truth = {int(k): dict(v) for k, v in state.get("month_truth", {}).items()}
        self._quarter_gdp_truth = {int(k): float(v) for k, v in state.get("quarter_gdp_truth", {}).items()}
        self._quarter_gdp_eps = {int(k): float(v) for k, v in state.get("quarter_gdp_eps", {}).items()}
        self._pending = []
        for raw in state.get("pending", []):
            tick = int(raw.pop("tick"))
            self._pending.append((tick, Vintage(**raw)))
        self._gdp_acc = [float(x) for x in state.get("gdp_acc", [])]
        self._trade_acc = [float(x) for x in state.get("trade_acc", [])]
        self._govt_acc = [float(x) for x in state.get("govt_acc", [])]
        self._rebuild_visible(10**12)


def _real_economy(ctx: Any) -> Any | None:
    for mod in ctx.world.modules:
        if getattr(mod, "name", None) == "real_economy":
            return mod
    return None


def attach_releases(world: Any, rng_hub: RngHub | None = None) -> DataReleases:
    """Append a ``DataReleases`` module to ``world`` if missing."""
    del rng_hub
    for mod in world.modules:
        if getattr(mod, "name", None) == "data_releases":
            return mod
    rel = DataReleases(calendar=world.clock.calendar)
    world.modules.append(rel)
    return rel
