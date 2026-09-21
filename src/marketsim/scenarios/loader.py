"""T4.09 — scripted scenario loader (§4.4 modes)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marketsim.core.errors import ConfigError
from marketsim.events.schema import DistSpec


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScriptedFire(FrozenModel):
    """One scheduled firing. ``tick`` is a day index; magnitude overrides the first primitive if set."""

    event_id: str
    tick: int
    magnitude: DistSpec | float | None = None

    @field_validator("tick")
    @classmethod
    def _tick(cls, v: int) -> int:
        if v < 0:
            raise ValueError("scripted tick must be >= 0")
        return v


class ScenarioSpec(FrozenModel):
    id: str
    seed: int | None = None
    hazards: bool = False
    events: list[ScriptedFire] = Field(default_factory=list)


def load_scenario(path: str | Path) -> ScenarioSpec:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{p} must contain a mapping")
    try:
        return ScenarioSpec.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{p}: {exc}") from exc


def load_scenarios(directory: str | Path) -> dict[str, ScenarioSpec]:
    root = Path(directory)
    out: dict[str, ScenarioSpec] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.yaml")):
        spec = load_scenario(path)
        out[spec.id] = spec
    return out
