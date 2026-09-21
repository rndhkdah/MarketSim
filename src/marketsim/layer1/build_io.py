"""Build the seed A matrix from `config/io_seed.yaml`. No import-time matrix work."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from marketsim.core.errors import ConfigError, IOTableError

FD_KEYS = ("HOUSEHOLD", "INVESTMENT", "GOVT", "EXPORTS")


def _config_dir(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[3] / "config"


@lru_cache(maxsize=4)
def load_seed(config_dir: str | None = None) -> dict[str, Any]:
    """Read `io_seed.yaml`. Cached per path; contains no Leontief work."""
    path = _config_dir(config_dir) / "io_seed.yaml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"invalid seed file: {path}")
    return data


def _codes(seed: dict[str, Any]) -> list[str]:
    return [row["code"] for row in seed["sectors"]]


# Public alias — the order in io_seed.yaml / io_table.json. Populated lazily.
def sector_codes(config_dir: str | Path | None = None) -> tuple[str, ...]:
    return tuple(_codes(load_seed(str(_config_dir(config_dir)))))


# Imported by callers that need a stable list without touching the IO table.
# Reading YAML is data, not a Leontief solve (B2).
CODES: tuple[str, ...] = sector_codes()


def _index(codes: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(codes)}


def mix_matrix(seed: dict[str, Any] | None = None) -> np.ndarray:
    seed = seed or load_seed()
    codes = _codes(seed)
    idx = _index(codes)
    n = len(codes)
    mix = np.zeros((n, n))
    raw = seed["mix"]
    for j, buyer in enumerate(codes):
        col = raw[buyer]
        for supplier, weight in col.items():
            mix[idx[supplier], j] = float(weight)
        s = mix[:, j].sum()
        if abs(s - 1.0) > 1e-6:
            if s <= 0:
                raise IOTableError(f"MIX column {buyer} is empty")
            mix[:, j] /= s
    return mix


def mu_vector(seed: dict[str, Any] | None = None) -> np.ndarray:
    seed = seed or load_seed()
    return np.array([float(row["mu"]) for row in seed["sectors"]], dtype=float)


def build_A(seed: dict[str, Any] | None = None) -> np.ndarray:
    """A[:, j] = mu[j] * MIX[:, j]. Units: cr input from i per 1 cr output of j."""
    seed = seed or load_seed()
    return mix_matrix(seed) * mu_vector(seed)[None, :]


def build_final_demand(seed: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    seed = seed or load_seed()
    codes = _codes(seed)
    idx = _index(codes)
    n = len(codes)
    out: dict[str, np.ndarray] = {}
    for key in FD_KEYS:
        vec = np.zeros(n)
        for code, w in seed["final_demand"][key].items():
            vec[idx[code]] = float(w)
        s = vec.sum()
        if abs(s - 1.0) > 1e-6:
            if s <= 0:
                raise IOTableError(f"final demand {key} is empty")
            vec = vec / s
        out[key] = vec
    return out


def fd_weights(seed: dict[str, Any] | None = None) -> dict[str, float]:
    seed = seed or load_seed()
    return {k: float(v) for k, v in seed["fd_weights"].items()}


def table_dict(seed: dict[str, Any] | None = None, *, decimals: int = 6) -> dict[str, Any]:
    seed = seed or load_seed()
    codes = _codes(seed)
    A = np.round(build_A(seed), decimals)
    fd = {k: np.round(v, decimals).tolist() for k, v in build_final_demand(seed).items()}
    return {
        "codes": codes,
        "names": [row["name"] for row in seed["sectors"]],
        "tiers": [row["tier"] for row in seed["sectors"]],
        "A": A.tolist(),
        "mu": np.round(mu_vector(seed), decimals).tolist(),
        "final_demand": fd,
        "fd_weights": fd_weights(seed),
        "institutions": list(seed["institutions"]),
        "source": "seed",
        "coverage": 1.0,
    }


def write_io_table(config_dir: str | Path | None = None, dest: str | Path | None = None) -> Path:
    root = _config_dir(config_dir)
    dest_path = Path(dest) if dest is not None else root / "io_table.json"
    payload = table_dict(load_seed(str(root)))
    dest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write config/io_table.json from io_seed.yaml")
    p.add_argument("--config-dir", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    path = write_io_table(args.config_dir, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
