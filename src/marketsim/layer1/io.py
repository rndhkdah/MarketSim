"""Single IO loader (B1). Nothing else may read the table or call `build_A` for validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from marketsim.core.config import Config
from marketsim.core.errors import IOTableError

COL_SUM_TOL = 5e-5


@dataclass
class IOTable:
    codes: tuple[str, ...]
    names: tuple[str, ...]
    tiers: tuple[str, ...]
    A: np.ndarray
    mu: np.ndarray
    final_demand: dict[str, np.ndarray]
    fd_weights: dict[str, float]
    institutions: tuple[str, ...]
    source: str
    coverage: float
    path: Path | None = None
    _L: np.ndarray | None = field(default=None, repr=False, compare=False)
    _G: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.codes = tuple(self.codes)
        self.names = tuple(self.names)
        self.tiers = tuple(self.tiers)
        self.institutions = tuple(self.institutions)
        self.A = np.asarray(self.A, dtype=float)
        self.mu = np.asarray(self.mu, dtype=float)
        self.final_demand = {k: np.asarray(v, dtype=float) for k, v in self.final_demand.items()}

    @property
    def n(self) -> int:
        return len(self.codes)

    @property
    def index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.codes)}

    @cached_property
    def L(self) -> np.ndarray:
        """Demand-pull inverse (I − A)⁻¹."""
        n = self.n
        return np.linalg.inv(np.eye(n) - self.A)

    @cached_property
    def G(self) -> np.ndarray:
        """Cost-push inverse (I − Aᵀ)⁻¹."""
        n = self.n
        return np.linalg.inv(np.eye(n) - self.A.T)

    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def output_multipliers(self) -> np.ndarray:
        return self.L.sum(axis=0)

    def baseline_final_demand(self, scale: float = 100.0) -> np.ndarray:
        """Weighted mix of FD vectors, summing to `scale` cr / month."""
        d = np.zeros(self.n)
        for key, weight in self.fd_weights.items():
            d = d + float(weight) * scale * self.final_demand[key]
        return d

    def to_dict(self) -> dict[str, Any]:
        return {
            "codes": list(self.codes),
            "names": list(self.names),
            "tiers": list(self.tiers),
            "A": self.A.tolist(),
            "mu": self.mu.tolist(),
            "final_demand": {k: v.tolist() for k, v in self.final_demand.items()},
            "fd_weights": dict(self.fd_weights),
            "institutions": list(self.institutions),
            "source": self.source,
            "coverage": self.coverage,
        }


def _check(table: IOTable) -> None:
    n = table.n
    if table.A.shape != (n, n):
        raise IOTableError(f"A must be {n}×{n}, got {table.A.shape}")
    if np.any(table.A < -1e-12):
        raise IOTableError("A has negative entries")
    col = table.A.sum(axis=0)
    if np.max(np.abs(col - table.mu)) > COL_SUM_TOL:
        raise IOTableError("column sums of A do not match mu (±5e-5)")
    rho = table.spectral_radius()
    if rho >= 1.0:
        raise IOTableError(f"ρ(A) = {rho:.6f} >= 1 (Hawkins–Simon failed)")
    for key, vec in table.final_demand.items():
        if abs(float(vec.sum()) - 1.0) > 1e-6:
            raise IOTableError(f"final-demand vector {key} does not sum to 1")


def load_io(path: str | Path) -> IOTable:
    p = Path(path)
    raw = json.loads(p.read_text())
    fd = {k: np.asarray(v, dtype=float) for k, v in raw["final_demand"].items()}
    table = IOTable(
        codes=tuple(raw["codes"]),
        names=tuple(raw["names"]),
        tiers=tuple(raw["tiers"]),
        A=np.asarray(raw["A"], dtype=float),
        mu=np.asarray(raw["mu"], dtype=float),
        final_demand=fd,
        fd_weights={k: float(v) for k, v in raw["fd_weights"].items()},
        institutions=tuple(raw["institutions"]),
        source=str(raw.get("source", "unknown")),
        coverage=float(raw.get("coverage", 1.0)),
        path=p,
    )
    _check(table)
    return table


def save_io(table: IOTable, path: str | Path) -> None:
    Path(path).write_text(json.dumps(table.to_dict(), indent=2) + "\n")


def resolve_io_path(cfg: Config) -> Path:
    """`io_table_bea.json` when `world.io_source == bea` and the file exists, else the seed."""
    root = cfg.config_dir
    bea = root / "io_table_bea.json"
    if cfg.world.io_source == "bea" and bea.exists():
        return bea
    return root / "io_table.json"
