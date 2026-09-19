from __future__ import annotations

from pathlib import Path

import pytest

from marketsim.core.config import load_config
from marketsim.layer1.build_io import write_io_table
from marketsim.layer1.io import load_io, resolve_io_path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "config"


@pytest.fixture(scope="session")
def config_dir() -> Path:
    io_path = CONFIG_DIR / "io_table.json"
    if not io_path.exists():
        write_io_table(CONFIG_DIR, io_path)
    return CONFIG_DIR


@pytest.fixture(scope="session")
def cfg(config_dir: Path):
    return load_config(config_dir)


@pytest.fixture(scope="session")
def io(cfg):
    return load_io(resolve_io_path(cfg))
