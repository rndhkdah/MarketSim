"""In-process world registry over ``FileStore`` (T9.02).

``WorldRecord.tick`` is days; ``seed`` is the RNG root; ``state_hash`` is hex
SHA-256. Listing is sorted by ``world_id`` (no unordered dict iteration).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marketsim.scenarios.replay import ReplayLog
from marketsim.storage.fs import FileStore
from marketsim.world import World


@dataclass(frozen=True)
class WorldRecord:
    """Registry row. ``tick`` is days; ``state_hash`` is a hex digest."""

    world_id: str
    seed: int
    tick: int
    state_hash: str
    config_hash: str


class WorldRegistry:
    """Many worlds on one filesystem root. ``root`` is a directory path."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def store(self, world_id: str) -> FileStore:
        """Return the ``FileStore`` for ``world_id`` (unitless)."""
        return FileStore(self.root, world_id)

    def register(
        self,
        world_id: str,
        world: World,
        *,
        log: ReplayLog | None = None,
    ) -> WorldRecord:
        """Write meta + snapshot (+ optional event log). ``world.clock.tick`` is days."""
        store = self.store(world_id)
        rec = WorldRecord(
            world_id=str(world_id),
            seed=int(world.seed),
            tick=int(world.clock.tick),
            state_hash=world.state_hash(),
            config_hash="" if log is None else log.config_hash,
        )
        store.write_meta(
            {
                "world_id": rec.world_id,
                "seed": rec.seed,
                "tick": rec.tick,
                "state_hash": rec.state_hash,
                "config_hash": rec.config_hash,
            }
        )
        store.write_world(world)
        if log is not None:
            store.write_log(log)
            store.write_export(log)
        return rec

    def get(self, world_id: str) -> WorldRecord:
        """Load the meta row. ``world_id`` is unitless."""
        meta = self.store(world_id).read_meta()
        return WorldRecord(
            world_id=str(meta["world_id"]),
            seed=int(meta["seed"]),
            tick=int(meta["tick"]),
            state_hash=str(meta["state_hash"]),
            config_hash=str(meta.get("config_hash") or ""),
        )

    def list_worlds(self) -> tuple[WorldRecord, ...]:
        """Registered worlds, sorted by ``world_id`` (unitless)."""
        ids = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        return tuple(self.get(wid) for wid in ids if (self.root / wid / "meta.json").is_file())

    def load_world(
        self,
        world_id: str,
        config_dir: str | Path,
        *,
        tick: int | None = None,
        modules: list[Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> World:
        """Restore a snapshot. Default ``tick`` is the meta tick (days)."""
        store = self.store(world_id)
        at = int(self.get(world_id).tick if tick is None else tick)
        return store.read_world(at, config_dir, modules=modules, overrides=overrides)

    def save_world(self, world_id: str, world: World, *, log: ReplayLog | None = None) -> WorldRecord:
        """Overwrite snapshot and meta (and log if given)."""
        return self.register(world_id, world, log=log)
