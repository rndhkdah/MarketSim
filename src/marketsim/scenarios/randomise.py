"""Per-episode domain randomisation and the public observe() whitelist (T6.22).

Ranges are calibration-style [lo, hi] around shipped defaults — not new
economics. They live here because ``WorldSettings`` is ``extra='forbid'``
and ``core/config.py`` is outside this card's Files list.

Randomness is only ``core.rng.stream("randomise")``. Episode draws are
logged in the replay header and never published through ``observe()``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marketsim.core.errors import StateError
from marketsim.core.rng import RngHub
from marketsim.pricing.mispricing import CALM_INDEX_VOL_MAX, CALM_INDEX_VOL_MIN

STREAM_NAME = "randomise"
# Dotted paths ``load_config`` already accepts (bundle root).
CONFIG_PREFIXES: tuple[str, ...] = ("markets.",)

# Units documented per key. Bounds hug shipped defaults / §6.3 bands.
DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "markets.turnover": (0.003, 0.005),  # 1/day; shipped 0.004 (§6.2)
    "markets.impact.Y": (0.64, 0.96),  # dimensionless; shipped 0.8 ±20% (§6.4)
    "markets.mm.s0": (0.0004, 0.0006),  # dimensionless half-spread; shipped 0.0005
    "mispricing.target_ann_vol": (CALM_INDEX_VOL_MIN, CALM_INDEX_VOL_MAX),  # annual decimal
    "mispricing.phi_s": (0.96, 0.99),  # dimensionless / day; shipped 0.98 (§6.3)
    "mispricing.theta_0": (0.03, 0.05),  # 1/day; shipped 0.04 (§6.3)
}

OBSERVE_PUBLIC_KEYS: frozenset[str] = frozenset(
    {
        "tick",
        "agent_id",
        "releases",
        "news",
        "fills",
        "orders",
        "portfolio",
        "quotes",
        "markets",
        "prices",
        "reports",
        "firm_decisions",
        "last_decision",
        "goods",
        "labour",
        "regions",
        "bonds",
    }
)

HIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "xi",
        "xi_j",
        "mispricing",
        "sentiment",
        "noise",
        "capital",
        "arb_capital",
        "A",
        "last_impact",
        "_month_truth",
        "month_truth",
        "unpublished",
        "truth",
        "noise_scale",
        "idio_vol",
        "vintages_unpublished",
    }
)


def _as_range(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            raise ValueError(f"{name} range lo>hi: {lo} > {hi}")
        return lo, hi
    raise ValueError(f"{name} must be a [lo, hi] pair")


def resolve_ranges(raw: Mapping[str, Any] | None = None) -> dict[str, tuple[float, float]]:
    """Merge optional caller ranges onto defaults. Each value is ``[lo, hi]``."""
    out = dict(DEFAULT_RANGES)
    if raw:
        for key in raw:
            out[str(key)] = _as_range(raw[key], str(key))
    return out


def sample_overrides(
    rng: RngHub,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Uniform draw per key via stream ``randomise``. Values are in each key's units."""
    table = resolve_ranges(None) if ranges is None else {k: _as_range(v, k) for k, v in ranges.items()}
    stream = rng.stream(STREAM_NAME)
    return {key: lo + (hi - lo) * float(stream.random()) for key, (lo, hi) in sorted(table.items())}


def config_overrides(draws: Mapping[str, float]) -> dict[str, float]:
    """Subset of draws that are valid ``load_config`` / ``World.create`` dotted paths."""
    return {k: float(v) for k, v in sorted(draws.items()) if k.startswith(CONFIG_PREFIXES)}


def replay_header(
    *,
    seed: int,
    overrides: Mapping[str, float],
    config_hash: str | None = None,
) -> dict[str, Any]:
    """Replay-log header: seed, stream name, sampled overrides (sorted)."""
    header: dict[str, Any] = {
        "seed": int(seed),
        "stream": STREAM_NAME,
        "overrides": {k: float(overrides[k]) for k in sorted(overrides)},
    }
    if config_hash is not None:
        header["config_hash"] = str(config_hash)
    return header


def _strip_hidden(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _strip_hidden(v)
            for k, v in value.items()
            if str(k) not in HIDDEN_KEYS and not str(k).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_hidden(v) for v in value]
    return value


def sanitise_observe(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the observation schema whitelist; drop ξ / arb capital / unpublished vintages."""
    return {
        key: _strip_hidden(obs[key])
        for key in obs
        if key in OBSERVE_PUBLIC_KEYS and key not in HIDDEN_KEYS
    }


def observe(world: Any, agent_id: str) -> dict[str, Any]:
    """``World.observe`` filtered to published market/econ fields (no hidden state)."""
    return sanitise_observe(world.observe(agent_id))


class DomainRandomiser:
    """Holds one episode's draws. Serialise with ``to_state`` / ``from_state``."""

    def __init__(self, ranges: Mapping[str, Any] | None = None) -> None:
        self.ranges = resolve_ranges(ranges)
        self.overrides: dict[str, float] = {}

    def draw(self, rng: RngHub) -> dict[str, float]:
        """Sample and store overrides. Units follow ``DEFAULT_RANGES``."""
        self.overrides = sample_overrides(rng, self.ranges)
        return dict(self.overrides)

    def config_overrides(self) -> dict[str, float]:
        return config_overrides(self.overrides)

    def replay_header(self, *, seed: int, config_hash: str | None = None) -> dict[str, Any]:
        return replay_header(seed=seed, overrides=self.overrides, config_hash=config_hash)

    def to_state(self) -> dict[str, Any]:
        return {
            "ranges": {k: [float(self.ranges[k][0]), float(self.ranges[k][1])] for k in sorted(self.ranges)},
            "overrides": {k: float(self.overrides[k]) for k in sorted(self.overrides)},
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> DomainRandomiser:
        if "ranges" not in state or "overrides" not in state:
            raise StateError("randomise state needs ranges and overrides")
        obj = cls(state["ranges"])
        obj.overrides = {str(k): float(v) for k, v in state["overrides"].items()}
        return obj
