"""T4.01 — event schema and catalog loader (§4.2)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketsim.core.errors import ConfigError
from marketsim.layer1.build_io import CODES

PRIMITIVES = frozenset(
    {"demand", "supply", "cost_push", "monetary", "risk_appetite", "fiscal", "catastrophe"}
)
CATEGORIES = frozenset(
    {
        "energy",
        "supply_chain",
        "pandemic",
        "technology",
        "disaster",
        "policy",
        "financial",
        "trade",
        "firm",
    }
)
EXTRA_STEMS = frozenset(
    {
        "want_shift",
        "labour_supply",
        "link_capacity",
        "link_cost",
        "world_demand",
        "import_price",
        "bank_equity",
        "collateral_value",
        "vat",
        "sentiment",
        "capex_preference",
    }
)
EXTRA_INDEXED = frozenset(
    {
        "want_shift",
        "labour_supply",
        "link_capacity",
        "link_cost",
        "import_price",
        "capex_preference",
    }
)
EXTRA_BARE = EXTRA_STEMS - EXTRA_INDEXED
_INDEXED_RE = re.compile(r"^([a-z_]+)\[([^\]]+)\]$")
DISTS = ("fixed", "normal", "lognormal", "uniform", "triangular")


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DistSpec(FrozenModel):
    """Sampled scalar. Truncation bounds are mandatory except ``fixed`` and day-uniform delays.

    Units: dimensionless magnitude unless ``low_days``/``high_days`` (days).
    ``verify`` marks a T4.13 calibration seed (not a sampling parameter).
    """

    dist: Literal["fixed", "normal", "lognormal", "uniform", "triangular"]
    value: float | None = None
    median: float | None = None
    mean: float | None = None
    sigma: float | None = None
    mode: float | None = None
    sign: float = 1.0
    min: float | None = None
    max: float | None = None
    low_days: float | None = None
    high_days: float | None = None
    verify: bool = False

    @model_validator(mode="after")
    def _complete(self) -> DistSpec:
        if self.dist == "fixed":
            if self.value is None:
                raise ValueError("fixed dist requires value")
            return self
        if self.low_days is not None or self.high_days is not None:
            if self.low_days is None or self.high_days is None:
                raise ValueError("delay dist requires low_days and high_days")
            if float(self.low_days) >= float(self.high_days):
                raise ValueError("low_days must be < high_days")
            return self
        if self.min is None or self.max is None:
            raise ValueError("truncation bounds min and max are mandatory")
        if float(self.min) > float(self.max):
            raise ValueError("min must be <= max")
        if self.dist == "lognormal" and (self.median is None or self.sigma is None):
            raise ValueError("lognormal requires median and sigma")
        if self.dist == "normal" and self.sigma is None:
            raise ValueError("normal requires sigma")
        if self.dist == "triangular" and self.mode is None:
            raise ValueError("triangular requires mode")
        return self


class ShockSpec(FrozenModel):
    shock: str
    magnitude: DistSpec
    persistence_q: float = 0.0
    targets: dict[str, Any] | None = None

    @field_validator("shock")
    @classmethod
    def _primitive(cls, v: str) -> str:
        if v not in PRIMITIVES:
            raise ValueError(f"composition may only use the seven primitives, got {v!r}")
        return v


class ExtraEffect(FrozenModel):
    variable: str
    shape: Literal["step", "ramp", "pulse", "decay"]
    magnitude: DistSpec | float
    duration: int = 0

    @field_validator("variable")
    @classmethod
    def _whitelist(cls, v: str) -> str:
        parse_extra_variable(v)
        return v

    @field_validator("duration")
    @classmethod
    def _dur(cls, v: int) -> int:
        if v < 0:
            raise ValueError("duration must be >= 0 (days)")
        return v


class FollowupSpec(FrozenModel):
    event_id: str
    probability: float
    delay: DistSpec | None = None

    @field_validator("probability")
    @classmethod
    def _p(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("follow-up probability must be in [0, 1]")
        return v


class HazardMultiplier(FrozenModel):
    feature: str
    coef: float
    sector: str | None = None


class HazardSpec(FrozenModel):
    base_rate_per_year: float
    multipliers: list[HazardMultiplier] = Field(default_factory=list)

    @field_validator("base_rate_per_year")
    @classmethod
    def _rate(cls, v: float) -> float:
        if v < 0:
            raise ValueError("base_rate_per_year must be >= 0")
        return v


class NewsSpec(FrozenModel):
    headline: str
    publication_lag_days: int = 0
    noise: float = 0.0

    @field_validator("noise")
    @classmethod
    def _noise(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("news.noise must be in [0, 1]")
        return v


class HistoricRef(FrozenModel):
    episode: str = ""
    source: str = ""
    calibrated_params: dict[str, Any] = Field(default_factory=dict)


class EventSpec(FrozenModel):
    """One event definition from ``config/events/*.yaml``."""

    id: str
    category: str
    hazard: HazardSpec
    cooldown_days: int = 0
    composition: list[ShockSpec] = Field(default_factory=list)
    targets: dict[str, Any] = Field(default_factory=dict)
    effects_extra: list[ExtraEffect] = Field(default_factory=list)
    followups: list[FollowupSpec] = Field(default_factory=list)
    news: NewsSpec | None = None
    historic_reference: HistoricRef | None = None

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"unknown category {v!r}")
        return v

    @field_validator("cooldown_days")
    @classmethod
    def _cd(cls, v: int) -> int:
        if v < 0:
            raise ValueError("cooldown_days must be >= 0")
        return v


def parse_extra_variable(raw: str) -> tuple[str, str | None]:
    """Split ``stem`` or ``stem[index]``. Reject unknown stems."""
    text = str(raw).strip()
    m = _INDEXED_RE.match(text)
    if m:
        stem, idx = m.group(1), m.group(2).strip()
        if stem not in EXTRA_INDEXED:
            raise ValueError(f"effects_extra variable {raw!r} is not on the whitelist")
        if not idx:
            raise ValueError(f"empty index in {raw!r}")
        return stem, idx
    if text in EXTRA_BARE:
        return text, None
    if text in EXTRA_INDEXED:
        raise ValueError(f"{text} requires an index")
    raise ValueError(f"effects_extra variable {raw!r} is not on the whitelist")


def _check_sector_mask(mask: dict[str, Any], codes: tuple[str, ...]) -> None:
    sectors = mask.get("sectors")
    if sectors is None or sectors == "all":
        return
    if not isinstance(sectors, list):
        raise ConfigError("targets.sectors must be a list or 'all'")
    unknown = [c for c in sectors if c not in codes]
    if unknown:
        raise ConfigError(f"unknown sector codes in mask: {unknown}")


def validate_event(
    spec: EventSpec,
    *,
    codes: tuple[str, ...] = CODES,
    known_ids: set[str] | None = None,
) -> None:
    """Mask codes and optional dangling follow-up check."""
    _check_sector_mask(spec.targets, codes)
    for shock in spec.composition:
        if shock.targets:
            _check_sector_mask(shock.targets, codes)
    if known_ids is not None:
        for fu in spec.followups:
            if fu.event_id not in known_ids:
                raise ConfigError(f"dangling follow-up {fu.event_id!r} on event {spec.id!r}")


def load_event(path: str | Path, *, codes: tuple[str, ...] = CODES) -> EventSpec:
    """Load one YAML event. Does not check dangling follow-ups (see ``load_catalog``)."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text())
        if not isinstance(raw, dict):
            raise ConfigError(f"{p} must contain a mapping")
        spec = EventSpec.model_validate(raw)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    validate_event(spec, codes=codes)
    return spec


def load_catalog(
    directory: str | Path,
    *,
    codes: tuple[str, ...] = CODES,
) -> dict[str, EventSpec]:
    """Load every ``*.yaml`` except names starting with ``_``. Reject dangling follow-ups."""
    root = Path(directory)
    if not root.is_dir():
        raise ConfigError(f"events dir not found: {root}")
    specs: dict[str, EventSpec] = {}
    for path in sorted(root.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        spec = load_event(path, codes=codes)
        if spec.id in specs:
            raise ConfigError(f"duplicate event id {spec.id!r}")
        specs[spec.id] = spec
    ids = set(specs)
    for spec in specs.values():
        validate_event(spec, codes=codes, known_ids=ids)
    return specs
