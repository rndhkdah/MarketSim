"""World facade: create / reset / submit / step / observe / state_hash / save / load."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from marketsim.core.config import Config, load_config
from marketsim.core.clock import Clock
from marketsim.core.hashing import state_hash as _hash
from marketsim.core.module import PIPELINE, Module, Phase
from marketsim.core.rng import RngHub


@dataclass
class TickContext:
    world: World
    tick: int
    phase: Phase | None = None
    due: list[Any] = field(default_factory=list)


class World:
    def __init__(self, cfg: Config, seed: int, modules: list[Module] | None = None) -> None:
        self.cfg = cfg
        self.seed = int(seed)
        self.clock = Clock()
        self.rng = RngHub(self.seed)
        self.modules: list[Module] = list(modules or [])
        self._inbox: dict[str, list[Any]] = {}
        self._observations: dict[str, dict[str, Any]] = {}

    @classmethod
    def create(
        cls,
        config_dir: str | Path,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
        scenario: str | None = None,
        modules: list[Module] | None = None,
    ) -> World:
        del scenario  # reserved for Phase 4
        cfg = load_config(config_dir, overrides)
        world = cls(cfg, seed if seed is not None else cfg.world.seed, modules=modules)
        world.reset()
        return world

    def reset(self) -> None:
        self.clock = Clock()
        self.rng = RngHub(self.seed)
        self._inbox.clear()
        self._observations.clear()
        ctx = TickContext(self, tick=0)
        for mod in self.modules:
            mod.reset(ctx)

    def submit(self, agent_id: str, actions: Any) -> None:
        self._inbox.setdefault(agent_id, []).append(actions)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            tick, due = self.clock.advance()
            ctx = TickContext(self, tick=tick, due=due)
            for phase in PIPELINE:
                ctx.phase = phase
                if phase is Phase.INGEST:
                    self._observations.clear()
                if phase is Phase.PUBLISH:
                    self._inbox.clear()
                for mod in self.modules:
                    mod.on_phase(ctx, phase)

    def observe(self, agent_id: str) -> dict[str, Any]:
        return {
            "tick": self.clock.tick,
            "agent_id": agent_id,
            **self._observations.get(agent_id, {}),
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "clock": self.clock.to_state(),
            "rng": self.rng.to_state(),
            "inbox": {k: list(v) for k, v in sorted(self._inbox.items())},
            "modules": {mod.name: mod.to_state() for mod in self.modules},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.seed = int(state["seed"])
        self.clock = Clock.from_state(state["clock"])
        self.rng = RngHub.from_state(state["rng"])
        self._inbox = {k: list(v) for k, v in state.get("inbox", {}).items()}
        by_name = {mod.name: mod for mod in self.modules}
        for name, payload in state.get("modules", {}).items():
            by_name[name].from_state(payload)

    def state_hash(self) -> str:
        return _hash(self.to_state())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_state(), default=_json_default))

    @classmethod
    def load(
        cls,
        path: str | Path,
        config_dir: str | Path,
        modules: list[Module] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> World:
        state = json.loads(Path(path).read_text())
        world = cls.create(config_dir, seed=int(state["seed"]), overrides=overrides, modules=modules)
        world.load_state(state)
        return world


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


class RandomWalkModule:
    """Dummy module used by the determinism test — draws from a named stream."""

    name = "random_walk"

    def __init__(self) -> None:
        self.x: float = 0.0

    def reset(self, ctx: TickContext) -> None:
        self.x = 0.0
        ctx.world.rng.stream(self.name)

    def on_phase(self, ctx: TickContext, phase: Phase) -> None:
        if phase is Phase.EVENTS:
            self.x += float(ctx.world.rng.stream(self.name).standard_normal())

    def to_state(self) -> dict[str, Any]:
        return {"x": self.x}

    def from_state(self, state: dict[str, Any]) -> None:
        self.x = float(state["x"])
