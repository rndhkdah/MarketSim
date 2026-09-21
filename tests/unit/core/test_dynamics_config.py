from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marketsim.core.config import load_config
from marketsim.core.errors import ConfigError

STOCK = (
    "ENERGY",
    "MATERIALS",
    "AGRIFOOD",
    "SEMIS",
    "AUTOS",
    "STAPLES",
    "DISCRET",
    "HEALTH",
)
ORDER = ("CAPGOODS", "CONSTRUCT")
FLOW = (
    "UTILITIES",
    "TRANSPORT",
    "SOFTWARE",
    "TELECOM",
    "BIZSVC",
    "BANKS",
    "INSURANCE",
    "REALESTATE",
)


def _copy_cfg(tmp_path: Path, config_dir: Path) -> Path:
    dest = tmp_path / "cfg"
    dest.mkdir()
    for name in ("sectors.yaml", "edges.yaml", "world.yaml", "dynamics.yaml"):
        (dest / name).write_text(
            (config_dir / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    return dest


def _load_mutated(tmp_path: Path, config_dir: Path, mutator) -> None:
    dest = _copy_cfg(tmp_path, config_dir)
    data = yaml.safe_load((dest / "dynamics.yaml").read_text(encoding="utf-8"))
    mutator(data)
    (dest / "dynamics.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    load_config(dest)


def test_dynamics_loads(cfg) -> None:
    d = cfg.dynamics
    assert d is not None
    modes = d.production.mode
    assert set(modes) == set(cfg.codes)
    for c in STOCK:
        assert modes[c] == "stock"
    for c in ORDER:
        assert modes[c] == "order"
    for c in FLOW:
        assert modes[c] == "flow"
    assert set(d.production.leak_per_month) <= set(STOCK)
    assert d.credit.pricing == "uniform"
    assert d.banks.mode == "passthrough"
    assert d.households.alpha1 == pytest.approx(0.70)
    assert d.residential.share_of_investment == pytest.approx(0.20)


def test_bad_mode_rejected(tmp_path: Path, config_dir: Path) -> None:
    def mut(data):
        data["production"]["mode"]["AUTOS"] = "just-in-time"

    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, config_dir, mut)


def test_negative_tau_rejected(tmp_path: Path, config_dir: Path) -> None:
    def mut(data):
        data["expectations"]["tau_sales_m"] = -1

    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, config_dir, mut)


def test_leak_on_flow_sector_rejected(tmp_path: Path, config_dir: Path) -> None:
    def mut(data):
        data["production"]["leak_per_month"]["SOFTWARE"] = 0.01

    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, config_dir, mut)


def test_share_out_of_range_rejected(tmp_path: Path, config_dir: Path) -> None:
    def mut(data):
        data["households"]["alpha1"] = 1.2

    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, config_dir, mut)
