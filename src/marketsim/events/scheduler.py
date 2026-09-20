"""T4.05 — Events World module: hazard draws, follow-up chains, cascade log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marketsim.core.config import Config
from marketsim.core.module import Phase
from marketsim.events.chains import check_subcritical
from marketsim.events.compose import apply_composition, sample_dist
from marketsim.events.effects import ExtraEffects
from marketsim.events.hazard import should_fire
from marketsim.events.news import NewsFeed
from marketsim.events.schema import EventSpec, load_catalog
from marketsim.real.shocks import ShockBus

DAYS_PER_MONTH = 21


@dataclass(frozen=True)
class CascadeNode:
    """One firing. Magnitudes stay off the news feed (see T4.06)."""

    tick: int
    event_id: str
    parent: str | None
    depth: int
    category: str


@dataclass
class Events:
    """World module. Phase EVENTS. RNG stream ``events`` only."""

    name: str = "events"
    catalog: dict[str, EventSpec] = field(default_factory=dict)
    damping: float = 0.7
    max_depth: int = 3
    max_concurrent: int = 4
    history: list[CascadeNode] = field(default_factory=list)
    last_fire_id: dict[str, int] = field(default_factory=dict)
    last_fire_cat: dict[str, int] = field(default_factory=dict)
    extras: ExtraEffects = field(default_factory=ExtraEffects)
    news: NewsFeed = field(default_factory=NewsFeed)
    _bus: ShockBus | None = None

    @classmethod
    def from_config(cls, cfg: Config, catalog: dict[str, EventSpec] | None = None) -> Events:
        ev_cfg = cfg.world.events
        cat = catalog if catalog is not None else load_catalog(cfg.config_dir / "events")
        check_subcritical(cat, max_depth=ev_cfg.max_depth, p_sum_cap=ev_cfg.p_sum_cap)
        return cls(
            catalog=cat,
            damping=ev_cfg.damping,
            max_depth=ev_cfg.max_depth,
            max_concurrent=ev_cfg.max_concurrent,
            news=NewsFeed(rumour_rate=ev_cfg.rumour_rate),
        )

    def concurrent(self, tick: int) -> int:
        """Same-tick occupancy (one slot per firing on ``tick``)."""
        return sum(1 for node in self.history if node.tick == int(tick))

    def try_fire(
        self,
        event_id: str,
        rng: np.random.Generator,
        tick: int,
        *,
        depth: int = 0,
        parent: str | None = None,
        bus: ShockBus | None = None,
        economy: Any | None = None,
        day: int = 0,
        ignore_cooldown: bool = False,
    ) -> bool:
        """Fire one event if depth, concurrency and cooldown allow. Returns True if fired."""
        if int(depth) > self.max_depth:
            return False
        if self.concurrent(tick) >= self.max_concurrent:
            return False
        spec = self.catalog[event_id]
        if not ignore_cooldown:
            last = self.last_fire_id.get(spec.id)
            if last is not None and spec.cooldown_days > 0 and int(tick) - last < spec.cooldown_days:
                return False
            last_c = self.last_fire_cat.get(spec.category)
            if last_c is not None and spec.cooldown_days > 0 and int(tick) - last_c < spec.cooldown_days:
                return False
        bus_use = bus or self._bus
        if bus_use is None and economy is not None:
            bus_use = economy.bus
        if bus_use is not None and spec.composition:
            apply_composition(spec, bus_use, rng, day=day, economy=economy, tick=tick)
        if spec.effects_extra:
            self.extras.start(spec.effects_extra, rng, tick, economy=economy)
        self.history.append(CascadeNode(int(tick), spec.id, parent, int(depth), spec.category))
        self.last_fire_id[spec.id] = int(tick)
        self.last_fire_cat[spec.category] = int(tick)
        self.news.enqueue(spec, tick, rng)
        return True

    def schedule_followups(
        self,
        spec: EventSpec,
        rng: np.random.Generator,
        tick: int,
        depth: int,
        queue: Any,
    ) -> None:
        """Sample delayed follow-ups with probability ``p × damping^depth``."""
        for fu in spec.followups:
            p = float(fu.probability) * (float(self.damping) ** int(depth))
            if p <= 0.0:
                continue
            if rng.random() >= p:
                continue
            delay = 1
            if fu.delay is not None:
                delay = max(1, int(round(sample_dist(fu.delay, rng))))
            queue.schedule(
                int(tick) + delay,
                {
                    "kind": "followup",
                    "event_id": fu.event_id,
                    "depth": int(depth) + 1,
                    "parent": spec.id,
                },
                priority=1,
            )

    def reset(self, ctx: Any) -> None:
        cfg = ctx.world.cfg
        self.history.clear()
        self.last_fire_id.clear()
        self.last_fire_cat.clear()
        self.extras = ExtraEffects()
        self.news = NewsFeed(rumour_rate=self.news.rumour_rate)
        self._bus = ShockBus.from_config(cfg, cfg.codes)
        ctx.world.rng.stream("events")

    def on_phase(self, ctx: Any, phase: Phase) -> None:
        if phase is Phase.REAL:
            eco = _economy(ctx)
            self.extras.apply(ctx.tick, eco)
            return
        if phase is Phase.PUBLISH:
            debug = ctx.world.cfg.world.mode == "game"
            self.news.publish_due(ctx.tick, debug=debug)
            public = ctx.world._observations.setdefault("*", {})
            public["news"] = self.news.public_dicts()
            return
        if phase is not Phase.EVENTS:
            return
        rng = ctx.world.rng.stream("events")
        eco = _economy(ctx)
        bus = eco.bus if eco is not None else self._bus
        cal = ctx.world.clock.calendar
        _y, _m, day1 = cal.ymd(ctx.tick)
        day = day1 - 1
        features = _features(eco, ctx.tick, self.last_fire_cat)
        self.news.maybe_rumour(self.catalog, ctx.tick, rng)
        for payload in ctx.due:
            if not isinstance(payload, dict):
                continue
            kind = payload.get("kind")
            if kind not in ("followup", "scripted"):
                continue
            eid = str(payload["event_id"])
            depth = 0 if kind == "scripted" else int(payload.get("depth") or 0)
            parent = None if kind == "scripted" else payload.get("parent")
            if eid not in self.catalog:
                continue
            if self.try_fire(
                eid,
                rng,
                ctx.tick,
                depth=depth,
                parent=parent,
                bus=bus,
                economy=eco,
                day=day,
                ignore_cooldown=kind == "scripted",
            ):
                self.schedule_followups(self.catalog[eid], rng, ctx.tick, depth, ctx.world.clock.queue)
        for eid in sorted(self.catalog):
            spec = self.catalog[eid]
            if not should_fire(
                spec.hazard,
                rng,
                features=features,
                tick=ctx.tick,
                last_fire_tick=self.last_fire_id.get(eid),
                cooldown_days=spec.cooldown_days,
            ):
                continue
            if self.try_fire(
                eid, rng, ctx.tick, depth=0, parent=None, bus=bus, economy=eco, day=day
            ):
                self.schedule_followups(spec, rng, ctx.tick, 0, ctx.world.clock.queue)

    def to_state(self) -> dict[str, Any]:
        return {
            "history": [n.__dict__ for n in self.history],
            "last_fire_id": dict(self.last_fire_id),
            "last_fire_cat": dict(self.last_fire_cat),
            "extras": self.extras.to_state(),
            "news": self.news.to_state(),
        }

    def from_state(self, state: dict[str, Any]) -> None:
        self.history = [CascadeNode(**raw) for raw in state.get("history", [])]
        self.last_fire_id = {k: int(v) for k, v in state.get("last_fire_id", {}).items()}
        self.last_fire_cat = {k: int(v) for k, v in state.get("last_fire_cat", {}).items()}
        if "extras" in state:
            self.extras.from_state(state["extras"])
        if "news" in state:
            self.news.from_state(state["news"])


def _economy(ctx: Any) -> Any | None:
    for mod in ctx.world.modules:
        if getattr(mod, "name", None) == "real_economy":
            return mod
    return None


def _features(eco: Any | None, tick: int, last_cat: dict[str, int]) -> dict[str, Any]:
    feats: dict[str, Any] = {
        "days_since_category": {c: float(tick - t) for c, t in last_cat.items()},
    }
    if eco is None or eco.last_agg is None:
        return feats
    agg = eco.last_agg
    gdp0 = float(eco.fin.gdp0)
    feats["output_gap"] = float(agg.gdp_prod_real / max(gdp0, 1e-12) - 1.0)
    feats["inflation"] = float(agg.pi12)
    feats["policy_rate"] = float(eco.cb.r)
    feats["unemployment"] = float(agg.u)
    feats["leverage"] = float(agg.leverage)
    x = np.asarray(eco.x, dtype=float)
    k = np.asarray(eco.k, dtype=float)
    ustar = np.asarray(eco.ustar, dtype=float)
    gap = x / np.maximum(k, 1e-12) - ustar
    feats["utilisation_gap"] = {code: float(gap[i]) for i, code in enumerate(eco.codes)}
    cover = np.asarray(eco.inv, dtype=float) / np.maximum(x, 1e-12)
    feats["inventory_cover"] = {code: float(cover[i]) for i, code in enumerate(eco.codes)}
    feats["bank_capital_ratio"] = float(getattr(eco, "bank_equity", 0.0))
    feats["price_fundamental_gap"] = 0.0
    return feats
