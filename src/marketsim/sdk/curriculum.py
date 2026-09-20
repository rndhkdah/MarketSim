"""T7.11 — scenario packs, lever/severity curricula, and evaluation metrics.

Curriculum metadata lives in ``CurriculumSpec`` (this module). Scripted fires
stay on T4.09 ``ScenarioSpec`` (``extra=forbid``). Randomness is only the
``World`` seed / ``events`` stream. No wall-clock, no global RNG.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from marketsim.core.config import load_config
from marketsim.core.errors import ConfigError
from marketsim.events.scheduler import Events
from marketsim.events.schema import load_catalog
from marketsim.scenarios.loader import ScenarioSpec, load_scenario
from marketsim.scenarios.runner import _zero_hazards
from marketsim.sdk.gym_env import is_bankrupt, net_worth_cr
from marketsim.world import FirmAgentModule, World

# Master plan §6: tick = 1 day; 21 days = 1 month; 252 days = 1 year.
DAYS_PER_MONTH = 21
DAYS_PER_YEAR = 12 * DAYS_PER_MONTH
# gym_env trader reward: ΔNW / (σ + 1 cr) so σ = 0 is defined.
SHARPE_VOL_FLOOR_CR = 1.0

# §7.3 lever curriculum (hiring is the ``labour`` lever).
LEVER_CURRICULUM: tuple[str, ...] = (
    "pricing",
    "production",
    "labour",
    "capex",
    "financing",
)
SEVERITY_TIERS: tuple[str, ...] = ("calm", "mild", "moderate", "severe")
CURRICULUM_FILENAMES: frozenset[str] = frozenset({"curriculum.yaml"})

SeverityName = Literal["calm", "mild", "moderate", "severe"]
LeverName = Literal["pricing", "production", "labour", "capex", "financing"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CurriculumStage(FrozenModel):
    """One lever-unlock stage. ``unlocked_levers`` is a prefix of ``LEVER_CURRICULUM``."""

    id: str = Field(description="Unitless stage id.")
    unlocked_levers: tuple[LeverName, ...] = Field(description="Unlocked lever names; unitless.")
    scenario_id: str = Field(description="T4.09 scenario id; unitless.")
    severity: SeverityName = Field(default="calm", description="Event-severity tier; unitless.")

    @field_validator("unlocked_levers")
    @classmethod
    def _levers(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [name for name in v if name not in LEVER_CURRICULUM]
        if unknown:
            raise ValueError(f"unknown curriculum levers: {unknown}")
        return v


class SeverityTier(FrozenModel):
    """Event-severity band. ``scenario_ids`` reference T4.09 scenario ids."""

    id: SeverityName = Field(description="calm|mild|moderate|severe; unitless.")
    scenario_ids: tuple[str, ...] = Field(default_factory=tuple, description="T4.09 scenario ids; unitless.")


class EvaluationSet(FrozenModel):
    """Fixed evaluation pack. ``scenario_ids`` are T4.09 ids (unitless)."""

    id: str = Field(description="Unitless evaluation-set id.")
    scenario_ids: tuple[str, ...] = Field(description="T4.09 scenario ids; unitless.")


class CurriculumSpec(FrozenModel):
    """Companion pack: lever stages + severity tiers + a fixed eval set.

    Not a ``ScenarioSpec`` — extra keys would be rejected there.
    """

    id: str = Field(description="Unitless curriculum id.")
    kind: Literal["curriculum"] = Field(default="curriculum", description="Discriminator; unitless.")
    seed: int | None = Field(default=None, description="World root seed (unitless); None uses scenario seeds.")
    lever_order: tuple[str, ...] = Field(default=LEVER_CURRICULUM, description="Lever names in unlock order; unitless.")
    stages: tuple[CurriculumStage, ...] = Field(description="One stage per lever; unitless ids.")
    severity_tiers: tuple[SeverityTier, ...] = Field(description="Four severity bands; unitless.")
    evaluation: EvaluationSet = Field(description="Fixed evaluation scenario ids; unitless.")

    @field_validator("lever_order")
    @classmethod
    def _order(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(v) != LEVER_CURRICULUM:
            raise ValueError(f"lever_order must be {LEVER_CURRICULUM}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> CurriculumSpec:
        if len(self.stages) != len(self.lever_order):
            raise ValueError("curriculum needs one stage per lever")
        acc: list[str] = []
        for stage, lever in zip(self.stages, self.lever_order, strict=True):
            acc.append(lever)
            if tuple(stage.unlocked_levers) != tuple(acc):
                raise ValueError(
                    f"stage {stage.id!r} must unlock {tuple(acc)}, got {stage.unlocked_levers}"
                )
        seen = tuple(tier.id for tier in self.severity_tiers)
        if seen != SEVERITY_TIERS:
            raise ValueError(f"severity_tiers must list {SEVERITY_TIERS} in order")
        if not self.evaluation.scenario_ids:
            raise ValueError("evaluation.scenario_ids must be non-empty")
        return self


@dataclass(frozen=True)
class EpisodeMetrics:
    """One evaluation year. Money in cr; Sharpe dimensionless; drawdown a peak fraction."""

    scenario_id: str
    ticks: int  # days
    net_worth: float  # cr (terminal)
    sharpe: float  # dimensionless
    max_drawdown: float  # fraction of peak net worth
    market_share: float  # dimensionless; 0 if unpublished
    bankruptcies: int  # count of onsets


def scenario_yaml_paths(directory: str | Path) -> list[Path]:
    """Sorted ``*.yaml`` paths under ``directory`` (unitless paths)."""
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(root.glob("*.yaml"))


def _raw_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return raw


def is_curriculum_mapping(raw: Mapping[str, Any]) -> bool:
    """True when the YAML is a ``CurriculumSpec``, not a T4.09 ``ScenarioSpec``."""
    if raw.get("kind") == "curriculum":
        return True
    return "stages" in raw or "lever_order" in raw or "severity_tiers" in raw


def load_pack_yaml(path: str | Path) -> ScenarioSpec | CurriculumSpec:
    """Load one pack file as ``ScenarioSpec`` or ``CurriculumSpec``."""
    p = Path(path)
    raw = _raw_mapping(p)
    if p.name in CURRICULUM_FILENAMES or is_curriculum_mapping(raw):
        try:
            return CurriculumSpec.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"{p}: {exc}") from exc
    return load_scenario(p)


def load_evaluation_scenarios(directory: str | Path) -> dict[str, ScenarioSpec]:
    """Every T4.09 scenario in ``directory``. Skips curriculum companion files."""
    out: dict[str, ScenarioSpec] = {}
    for path in scenario_yaml_paths(directory):
        loaded = load_pack_yaml(path)
        if isinstance(loaded, CurriculumSpec):
            continue
        if loaded.id in out:
            raise ConfigError(f"duplicate scenario id {loaded.id!r}")
        out[loaded.id] = loaded
    return out


def load_curriculum(path: str | Path) -> CurriculumSpec:
    """Load ``curriculum.yaml`` from a file or a scenarios directory."""
    p = Path(path)
    if p.is_dir():
        candidates = [p / name for name in sorted(CURRICULUM_FILENAMES)]
        for cand in candidates:
            if cand.is_file():
                loaded = load_pack_yaml(cand)
                if isinstance(loaded, CurriculumSpec):
                    return loaded
        for cand in scenario_yaml_paths(p):
            loaded = load_pack_yaml(cand)
            if isinstance(loaded, CurriculumSpec):
                return loaded
        raise ConfigError(f"no CurriculumSpec YAML in {p}")
    loaded = load_pack_yaml(p)
    if not isinstance(loaded, CurriculumSpec):
        raise ConfigError(f"{p} is a ScenarioSpec, not a curriculum")
    return loaded


def net_worth(series: Sequence[float]) -> float:
    """Terminal net worth (cr). 0 if the series is empty."""
    if not series:
        return 0.0
    return float(series[-1])


def sharpe(series: Sequence[float]) -> float:
    """Mean ΔNW / (expanding σ + 1 cr). Dimensionless. Matches gym_env Welford σ."""
    if len(series) < 2:
        return 0.0
    n = 0
    mean = 0.0
    m2 = 0.0
    prev = float(series[0])
    for value in series[1:]:
        pnl = float(value) - prev
        prev = float(value)
        n += 1
        delta = pnl - mean
        mean += delta / float(n)
        m2 += delta * (pnl - mean)
    var = m2 / float(n) if n else 0.0
    vol = math.sqrt(max(var, 0.0))
    return mean / (vol + SHARPE_VOL_FLOOR_CR)


def max_drawdown(series: Sequence[float]) -> float:
    """Largest peak-to-trough drop as a fraction of peak net worth (dimensionless)."""
    peak = 0.0
    worst = 0.0
    for value in series:
        nw = float(value)
        if nw > peak:
            peak = nw
        if peak > 0.0:
            worst = max(worst, (peak - nw) / peak)
    return worst


def bankruptcies(flags: Sequence[bool]) -> int:
    """Count of bankruptcy onsets (rising edges). Unitless."""
    count = 0
    prev = False
    for flag in flags:
        cur = bool(flag)
        if cur and not prev:
            count += 1
        prev = cur
    return count


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def published_market_share(raw: Mapping[str, Any], *, firm_id: str = "") -> float:
    """Published firm/cell share (dimensionless). 0 if unpublished."""
    if "market_share" in raw:
        value = raw.get("market_share")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    port = raw.get("portfolio")
    if isinstance(port, Mapping) and "market_share" in port:
        value = port.get("market_share")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    goods = raw.get("goods")
    if isinstance(goods, Mapping) and "market_share" in goods:
        value = goods.get("market_share")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    for row in raw.get("reports") or ():
        if firm_id and str(_attr(row, "firm_id", "")) != firm_id:
            continue
        books = _attr(row, "books")
        if not isinstance(books, Mapping):
            continue
        for key in ("market_share", "share"):
            if key in books:
                value = books[key]
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    return float(value)
    return 0.0


def episode_metrics(
    series: Sequence[float],
    *,
    scenario_id: str = "",
    market_shares: Sequence[float] | None = None,
    bankrupt_flags: Sequence[bool] | None = None,
) -> EpisodeMetrics:
    """Pack metric functions for one net-worth path (cr)."""
    share = 0.0
    if market_shares:
        share = float(market_shares[-1])
    return EpisodeMetrics(
        scenario_id=str(scenario_id),
        ticks=len(series),
        net_worth=net_worth(series),
        sharpe=sharpe(series),
        max_drawdown=max_drawdown(series),
        market_share=share,
        bankruptcies=bankruptcies(bankrupt_flags or ()),
    )


def open_scenario_world(
    config_dir: str | Path,
    scenario: str | Path | ScenarioSpec,
    *,
    extra_modules: Sequence[Any] | None = None,
) -> tuple[World, ScenarioSpec]:
    """Build a ``World.create`` world, schedule T4.09 scripted fires, do not step.

    Modules are Events (required to honour the script) plus ``FirmAgentModule``
    (the cheap ``World.create`` default used by ``MarketSimEnv``).
    """
    spec = scenario if isinstance(scenario, ScenarioSpec) else load_scenario(scenario)
    cfg = load_config(config_dir)
    catalog = load_catalog(Path(config_dir) / "events")
    ev = Events.from_config(cfg, catalog=catalog)
    if not spec.hazards:
        _zero_hazards(ev)
    mods: list[Any] = [ev, FirmAgentModule(), *(extra_modules or ())]
    world = World.create(config_dir, seed=spec.seed, modules=mods)
    live = next(m for m in world.modules if getattr(m, "name", None) == "events")
    if not spec.hazards:
        _zero_hazards(live)
    for fire in spec.events:
        mag = fire.magnitude
        payload: dict[str, Any] = {
            "kind": "scripted",
            "event_id": fire.event_id,
            "magnitude": mag if not hasattr(mag, "dist") else mag.model_dump(),
        }
        world.clock.queue.schedule(fire.tick, payload, priority=0)
    return world, spec


def run_scenario_year(
    config_dir: str | Path,
    scenario: str | Path | ScenarioSpec,
    *,
    agent_id: str = "eval",
    firm_id: str = "acme",
    n_ticks: int = DAYS_PER_YEAR,
) -> EpisodeMetrics:
    """Step ``n_ticks`` days (default one year = 252) and record evaluation metrics."""
    if n_ticks < 1:
        raise ValueError("n_ticks must be >= 1 (days)")
    world, spec = open_scenario_world(config_dir, scenario)
    nws: list[float] = []
    shares: list[float] = []
    flags: list[bool] = []
    for _ in range(int(n_ticks)):
        world.step(1)
        raw = world.observe(agent_id)
        nw = net_worth_cr(raw)
        nws.append(nw)
        shares.append(published_market_share(raw, firm_id=firm_id))
        flags.append(is_bankrupt(raw, net_worth=nw, firm_id=firm_id))
    return episode_metrics(
        nws,
        scenario_id=spec.id,
        market_shares=shares,
        bankrupt_flags=flags,
    )
