"""Directory layout for event logs, snapshots, and columnar export (T9.02).

Parquet / a SQL backend can implement the same methods later. This store uses
JSON (World / ReplayLog) plus ``.npz`` (T7.10 ``EXPORT_KEYS``) so the core
stays on MIT/BSD/Apache deps. Paths are filesystem paths; ticks are days.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from marketsim.api.replay import (
    load_snapshot,
    read_replay_log,
    write_export_npz,
    write_replay_log,
    write_snapshot,
)
from marketsim.scenarios.replay import ReplayLog
from marketsim.world import World

# Catalog filename (unitless). T9.02.
_META_NAME = "meta.json"
_LOG_NAME = "log.json"
_SNAP_DIR = "snap"
_EXPORT_NAME = "export.npz"


class FileStore:
    """One world's files under ``root / world_id``. ``root`` is a filesystem path."""

    def __init__(self, root: str | Path, world_id: str) -> None:
        self.root = Path(root)
        self.world_id = str(world_id)
        self.home = self.root / self.world_id

    def ensure(self) -> Path:
        """Create the world directory. Returns the home path."""
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / _SNAP_DIR).mkdir(exist_ok=True)
        return self.home

    def meta_path(self) -> Path:
        return self.home / _META_NAME

    def log_path(self) -> Path:
        return self.home / _LOG_NAME

    def export_path(self) -> Path:
        return self.home / _EXPORT_NAME

    def snapshot_path(self, tick: int) -> Path:
        """``snap/{tick}.json``. ``tick`` is days."""
        return self.home / _SNAP_DIR / f"{int(tick)}.json"

    def write_meta(self, meta: dict[str, Any]) -> Path:
        self.ensure()
        dest = self.meta_path()
        dest.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return dest

    def read_meta(self) -> dict[str, Any]:
        raw = json.loads(self.meta_path().read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("world meta must be a JSON object")
        return raw

    def write_log(self, log: ReplayLog) -> Path:
        self.ensure()
        return write_replay_log(self.log_path(), log)

    def read_log(self) -> ReplayLog:
        return read_replay_log(self.log_path())

    def write_world(self, world: World) -> Path:
        """Snapshot at ``world.clock.tick`` (days)."""
        self.ensure()
        return write_snapshot(self.snapshot_path(int(world.clock.tick)), world)

    def read_world(
        self,
        tick: int,
        config_dir: str | Path,
        *,
        modules: list[Any] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> World:
        """Restore the snapshot at ``tick`` (days). ``config_dir`` is a filesystem path."""
        return load_snapshot(
            self.snapshot_path(int(tick)),
            config_dir,
            modules=modules,
            overrides=overrides,
        )

    def list_snapshot_ticks(self) -> tuple[int, ...]:
        """Sorted ticks (days) that have a JSON snapshot."""
        folder = self.home / _SNAP_DIR
        if not folder.is_dir():
            return ()
        ticks: list[int] = []
        for path in folder.iterdir():
            if path.suffix == ".json" and path.stem.isdigit():
                ticks.append(int(path.stem))
        return tuple(sorted(ticks))

    def write_export(self, log: ReplayLog) -> Path:
        """Columnar ``EXPORT_KEYS`` as ``.npz`` (the parquet stand-in)."""
        self.ensure()
        return write_export_npz(self.export_path(), log)
