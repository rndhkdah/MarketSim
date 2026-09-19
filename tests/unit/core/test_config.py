from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from marketsim.core.config import Config, load_config
from marketsim.core.errors import ConfigError


def test_shipped_config_loads(cfg: Config) -> None:
    assert len(cfg.codes) == 18
    assert cfg.sectors.params("AUTOS").eps_own < 0
    assert abs(sum(cfg.sectors.market_cap_weights.values()) - 1.0) < 1e-6
    assert abs(sum(cfg.edges.capex.routing.values()) - 1.0) < 1e-6
    assert cfg.edges.credit.bank_capital_gate.ratio == "equity_over_rwa"


def test_override_dotted(config_dir: Path) -> None:
    cfg = load_config(config_dir, overrides={"world.seed": 7, "world.scale": 2.5})
    assert cfg.world.seed == 7
    assert cfg.world.scale == 2.5


def test_frozen_mutation_raises(cfg: Config) -> None:
    with pytest.raises((ValidationError, TypeError)):
        cfg.world.seed = 99  # type: ignore[misc]


def _write_bad(tmp_path: Path, config_dir: Path, mutator) -> Path:
    dest = tmp_path / "cfg"
    dest.mkdir()
    for name in ("sectors.yaml", "edges.yaml", "world.yaml", "dynamics.yaml"):
        text = (config_dir / name).read_text()
        data = yaml.safe_load(text)
        if name == "sectors.yaml":
            data = mutator(data) if mutator.__name__.startswith("sec") else data
        if name == "edges.yaml":
            data = mutator(data) if mutator.__name__.startswith("edg") else data
        (dest / name).write_text(yaml.safe_dump(data))
    return dest


def test_eps_own_must_be_negative(tmp_path: Path, config_dir: Path) -> None:
    def sec_bad(data):
        data["sectors"]["AUTOS"]["eps_own"] = 0.2
        return data

    with pytest.raises(ConfigError):
        load_config(_write_bad(tmp_path, config_dir, sec_bad))


def test_pass_through_bounds(tmp_path: Path, config_dir: Path) -> None:
    def sec_bad(data):
        data["sectors"]["UTILITIES"]["pass_through"] = 0.0
        return data

    with pytest.raises(ConfigError):
        load_config(_write_bad(tmp_path, config_dir, sec_bad))


def test_negative_lag_rejected(tmp_path: Path, config_dir: Path) -> None:
    def sec_bad(data):
        data["sectors"]["ENERGY"]["build_lag_q"] = -1
        return data

    with pytest.raises(ConfigError):
        load_config(_write_bad(tmp_path, config_dir, sec_bad))


def test_market_cap_must_sum(tmp_path: Path, config_dir: Path) -> None:
    def sec_bad(data):
        data["market_cap_weights"]["SOFTWARE"] = 0.99
        return data

    with pytest.raises(ConfigError):
        load_config(_write_bad(tmp_path, config_dir, sec_bad))


def test_unknown_edge_endpoint(tmp_path: Path, config_dir: Path) -> None:
    def edg_bad(data):
        data["substitution"].append(
            {
                "src": "MOON",
                "dst": "AUTOS",
                "channel": "substitution",
                "elasticity": 0.1,
                "lag": "erlang(2)",
                "sign": "+",
            }
        )
        return data

    with pytest.raises(ConfigError):
        load_config(_write_bad(tmp_path, config_dir, edg_bad))
