"""Thin REST/export helpers for replay logs, snapshots, and ``.npz`` series (T7.10 / §7.2).

``GET /v1/worlds/{wid}/replay``, ``POST …/export``, ``POST …/save`` and
``POST /v1/worlds/load`` call these. They do not depend on FastAPI; ``rest.py``
is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from marketsim.scenarios.replay import (
    EXPORT_DTYPES,
    EXPORT_KEYS,
    EXPORT_UNITS,
    ReplayLog,
    hash_overrides,
    replay,
    series_from_log,
    world_overrides,
)
from marketsim.world import World

__all__ = [
    "EXPORT_DTYPES",
    "EXPORT_KEYS",
    "EXPORT_UNITS",
    "create_world",
    "hash_overrides",
    "load_snapshot",
    "read_replay_log",
    "replay",
    "write_export_npz",
    "write_replay_log",
    "write_snapshot",
]


def create_world(
    log: ReplayLog,
    config_dir: str | Path,
    *,
    modules: list[Any] | None = None,
) -> World:
    """Build a ``World`` from the log header. ``config_dir`` is a filesystem path."""
    overrides = world_overrides(log)
    return World.create(
        config_dir,
        seed=log.seed,
        overrides=overrides or None,
        modules=modules,
    )


def write_replay_log(path: str | Path, log: ReplayLog) -> Path:
    """Write the event-sourced log as JSON. ``path`` is a filesystem path."""
    dest = Path(path)
    dest.write_text(
        json.dumps(log.to_state(), sort_keys=True, default=str),
        encoding="utf-8",
    )
    return dest


def read_replay_log(path: str | Path) -> ReplayLog:
    """Load a JSON log. ``path`` is a filesystem path."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("replay log must be a JSON object")
    return ReplayLog.from_state(raw)


def write_export_npz(path: str | Path, log: ReplayLog) -> Path:
    """Write ``EXPORT_KEYS`` series to ``.npz``. ``path`` is a filesystem path."""
    dest = Path(path)
    arrays = series_from_log(log)
    np.savez(dest, **{key: arrays[key] for key in EXPORT_KEYS})
    return dest


def write_snapshot(path: str | Path, world: World) -> Path:
    """Write ``World.save`` JSON. ``path`` is a filesystem path."""
    dest = Path(path)
    world.save(dest)
    return dest


def load_snapshot(
    path: str | Path,
    config_dir: str | Path,
    *,
    modules: list[Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> World:
    """Restore a ``World`` from ``write_snapshot``. ``config_dir`` is a filesystem path."""
    return World.load(path, config_dir, modules=modules, overrides=overrides)
