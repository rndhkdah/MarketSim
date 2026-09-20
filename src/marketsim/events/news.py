"""T4.06 — news feed (§4.3). Magnitudes are never published."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from marketsim.events.schema import EventSpec

SEVERITY_LEVELS = (0, 1, 2, 3)


@dataclass(frozen=True)
class NewsItem:
    """Public headline. Units: tick = day; severity_hint is a noisy ordinal 0–3."""

    id: str
    tick: int
    category: str
    headline: str
    regions: tuple[str, ...]
    sectors: tuple[str, ...]
    severity_hint: int
    is_rumour: bool = False


def true_severity(spec: EventSpec) -> int:
    """Internal ordinal from the first primitive's typical size. Not a published field."""
    if not spec.composition:
        return 1
    dist = spec.composition[0].magnitude
    mag = abs(float(dist.median or dist.value or dist.mean or dist.mode or 0.0))
    if mag < 0.05:
        return 0
    if mag < 0.15:
        return 1
    if mag < 0.40:
        return 2
    return 3


def misclassify(true: int, noise: float, rng: np.random.Generator) -> int:
    """With probability ``noise``, draw a different ordinal in ``SEVERITY_LEVELS``."""
    if float(noise) <= 0.0 or rng.random() >= float(noise):
        return int(true)
    others = [s for s in SEVERITY_LEVELS if s != int(true)]
    return int(others[int(rng.integers(0, len(others)))])


def item_from_event(
    spec: EventSpec,
    fire_tick: int,
    rng: np.random.Generator,
    *,
    is_rumour: bool = False,
) -> tuple[int, NewsItem]:
    """Return ``(publish_tick, item)``. Headline never contains a magnitude."""
    news = spec.news
    lag = int(news.publication_lag_days) if news is not None else 0
    noise = float(news.noise) if news is not None else 0.0
    headline = news.headline if news is not None else spec.id.replace("_", " ")
    regions_raw = spec.targets.get("regions") if spec.targets else None
    if regions_raw in (None, "all"):
        regions: tuple[str, ...] = ()
    elif isinstance(regions_raw, list):
        regions = tuple(str(r) for r in regions_raw)
    else:
        regions = ()
    sectors: tuple[str, ...] = ()
    if spec.composition and spec.composition[0].targets:
        raw = spec.composition[0].targets.get("sectors") or []
        if raw != "all":
            sectors = tuple(str(s) for s in raw)
    hint = misclassify(true_severity(spec), noise, rng)
    publish_tick = int(fire_tick) + max(lag, 0)
    item = NewsItem(
        id=spec.id,
        tick=publish_tick,
        category=spec.category,
        headline=headline,
        regions=tuple(str(r) for r in regions if r != "all"),
        sectors=sectors,
        severity_hint=hint,
        is_rumour=bool(is_rumour),
    )
    return publish_tick, item


class NewsFeed:
    """Lagged headlines. Rumours keep ``is_rumour`` only when ``debug`` is true."""

    def __init__(self, *, rumour_rate: float = 0.0) -> None:
        self.rumour_rate = float(rumour_rate)
        self._pending: list[tuple[int, NewsItem]] = []
        self.published: list[NewsItem] = []

    def enqueue(self, spec: EventSpec, fire_tick: int, rng: np.random.Generator, *, is_rumour: bool = False) -> NewsItem:
        ts, item = item_from_event(spec, fire_tick, rng, is_rumour=is_rumour)
        self._pending.append((ts, item))
        return item

    def maybe_rumour(
        self,
        catalog: dict[str, EventSpec],
        tick: int,
        rng: np.random.Generator,
    ) -> None:
        """False headlines at ``rumour_rate`` per category per year, converted to a daily hazard."""
        if self.rumour_rate <= 0.0 or not catalog:
            return
        p = 1.0 - float(np.exp(-self.rumour_rate / 252.0))
        by_cat: dict[str, list[EventSpec]] = {}
        for spec in catalog.values():
            by_cat.setdefault(spec.category, []).append(spec)
        for _cat, specs in sorted(by_cat.items()):
            if rng.random() >= p:
                continue
            spec = specs[int(rng.integers(0, len(specs)))]
            self.enqueue(spec, tick, rng, is_rumour=True)

    def publish_due(self, tick: int, *, debug: bool = False) -> list[NewsItem]:
        due: list[NewsItem] = []
        keep: list[tuple[int, NewsItem]] = []
        for ts, item in self._pending:
            if ts <= int(tick):
                out = item if debug else replace(item, is_rumour=False)
                self.published.append(out)
                due.append(out)
            else:
                keep.append((ts, item))
        self._pending = keep
        return due

    def public_dicts(self) -> list[dict[str, Any]]:
        """JSON-safe items. No magnitude keys."""
        return [asdict(it) for it in self.published]

    def to_state(self) -> dict[str, Any]:
        return {
            "rumour_rate": self.rumour_rate,
            "pending": [{"ts": t, **asdict(it)} for t, it in self._pending],
            "published": [asdict(it) for it in self.published],
        }

    def from_state(self, state: dict[str, Any]) -> None:
        self.rumour_rate = float(state.get("rumour_rate", 0.0))
        self._pending = []
        for raw in state.get("pending", []):
            payload = dict(raw)
            ts = int(payload.pop("ts"))
            self._pending.append((ts, NewsItem(**payload)))
        self.published = [NewsItem(**dict(raw)) for raw in state.get("published", [])]
