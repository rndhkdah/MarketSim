"""Record a stepping ``World`` into golden-comparable monthly series (T10.16).

The recorder is a **caller-side reader**, never a ``World`` module: a module would
enter ``World.to_state()`` and change every ``state_hash``. It samples
``RealEconomy.last_agg`` after each tick and appends one row per closed month, so
recording a run is observationally neutral — the hash of a recorded run equals the
hash of the same run without a recorder.

It is a reporting tool with full read access to the economy, **not** an observation
path for agents: nothing here is published through ``observe()``, so the T6.22
hidden-state whitelist is untouched.

Units follow master plan §6 — ``cr``/month for flows, indices (1.0 at baseline) for
prices, annual decimals for rates, fractions for ratios, ``tick`` in simulated days.
Series names match ``tests/golden/data/aggregate_baseline.npz`` (``gdp``, ``cpi``,
``u``, ``r``, ``w``, ``x``, ``p``) so a recording and a golden compare key by key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "run-v1"

# name -> (source attribute on Aggregates, unit). ``gdp`` keeps the golden's name
# for ``Aggregates.gdp_prod_real`` so a recording overlays a golden directly.
SCALAR_SERIES: tuple[tuple[str, str, str], ...] = (
    ("gdp", "gdp_prod_real", "cr/month at baseline prices"),
    ("gdp_exp_real", "gdp_exp_real", "cr/month at baseline prices"),
    ("gdp_prod_nom", "gdp_prod_nom", "cr/month"),
    ("gdp_exp_nom", "gdp_exp_nom", "cr/month"),
    ("cpi", "cpi", "index (1.0 at baseline)"),
    ("core_cpi", "core_cpi", "index (1.0 at baseline)"),
    ("u", "u", "fraction of the labour force"),
    ("utilisation", "utilisation", "fraction of capacity"),
    ("govt_balance", "govt_balance", "cr/month (+ surplus)"),
    ("debt_gdp", "debt_gdp", "fraction of annual GDP"),
    ("trade_balance", "trade_balance", "cr/month (+ surplus)"),
    ("saving_rate", "saving_rate", "fraction of disposable income"),
    ("leverage", "leverage", "net debt / annual EBITDA"),
    ("pi12", "pi12", "annual decimal"),
)

# Series read off the economy rather than off ``Aggregates``.
ECONOMY_SERIES: tuple[tuple[str, str], ...] = (
    ("r", "annual decimal"),
    ("w", "cr / person / month"),
)

MATRIX_SERIES: tuple[tuple[str, str], ...] = (
    ("x", "cr/month at baseline prices, per sector"),
    ("p", "index (1.0 at baseline), per sector"),
)

SERIES_UNITS: dict[str, str] = {
    **{name: unit for name, _src, unit in SCALAR_SERIES},
    **dict(ECONOMY_SERIES),
    **dict(MATRIX_SERIES),
    "months": "months since the run started (1-based, one row per closed month)",
    "ticks": "simulated days at the month close",
}


@dataclass(frozen=True)
class NewsRow:
    """One published headline. ``tick`` is simulated days; severity is ordinal 0–3."""

    tick: int
    event_id: str
    category: str
    headline: str
    severity_hint: int
    is_rumour: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "event_id": self.event_id,
            "category": self.category,
            "headline": self.headline,
            "severity_hint": self.severity_hint,
            "is_rumour": self.is_rumour,
        }


@dataclass
class Recording:
    """Monthly series from one run. ``scalars`` are ``(n,)``; ``matrices`` are ``(n, …)``."""

    codes: tuple[str, ...] = ()
    months: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    ticks: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    matrices: dict[str, np.ndarray] = field(default_factory=dict)
    news: tuple[NewsRow, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.months.shape[0])

    def series_names(self) -> tuple[str, ...]:
        """Scalar series present, sorted (deterministic iteration)."""
        return tuple(sorted(self.scalars))

    def matrix_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.matrices))

    def shared_series(self, other: Recording) -> tuple[str, ...]:
        """Scalar series both recordings carry, sorted."""
        return tuple(sorted(set(self.scalars) & set(other.scalars)))


def _economy(world: Any) -> Any:
    """The attached ``RealEconomy``, or ``None``.

    Replace with ``marketsim.core.lookup.find_module(world, "real_economy")`` once
    T10.00 lands; kept local so this card adds no file another card owns.
    """
    for mod in getattr(world, "modules", ()):
        if getattr(mod, "name", None) == "real_economy":
            return mod
    return None


class RunRecorder:
    """Sample a ``World`` once per closed month. Reading only — never a ``World`` module."""

    def __init__(self, world: Any) -> None:
        self._world = world
        self._months: list[int] = []
        self._ticks: list[int] = []
        self._scalars: dict[str, list[float]] = {name: [] for name, _s, _u in SCALAR_SERIES}
        self._scalars.update({name: [] for name, _u in ECONOMY_SERIES})
        self._matrices: dict[str, list[np.ndarray]] = {name: [] for name, _u in MATRIX_SERIES}
        self._news: dict[tuple[str, int], NewsRow] = {}
        self._codes: tuple[str, ...] = ()
        eco = _economy(world)
        self._last_month = int(getattr(eco, "month", 0)) if eco is not None else 0

    def sample(self) -> bool:
        """Append a row if a month closed since the last call. Returns whether it did."""
        self._collect_news()
        eco = _economy(self._world)
        if eco is None:
            return False
        month = int(eco.month)
        agg = getattr(eco, "last_agg", None)
        if month <= self._last_month or agg is None:
            return False
        self._last_month = month
        self._codes = tuple(eco.codes)
        self._months.append(month)
        self._ticks.append(int(self._world.clock.tick))
        for name, source, _unit in SCALAR_SERIES:
            self._scalars[name].append(float(getattr(agg, source)))
        self._scalars["r"].append(float(eco.cb.r))
        self._scalars["w"].append(float(eco.w))
        self._matrices["x"].append(np.asarray(agg.x, dtype=float).copy())
        self._matrices["p"].append(np.asarray(eco.p, dtype=float).copy())
        return True

    def _collect_news(self) -> None:
        """Accumulate published headlines; ``_observations`` is cleared every tick."""
        try:
            published = self._world.observe("*")
        except Exception:  # noqa: BLE001 - a world with no publisher is not an error here
            return
        for raw in published.get("news") or ():
            if not isinstance(raw, dict):
                continue
            row = NewsRow(
                tick=int(raw.get("tick", 0)),
                event_id=str(raw.get("id", "")),
                category=str(raw.get("category", "")),
                headline=str(raw.get("headline", "")),
                severity_hint=int(raw.get("severity_hint", 0)),
                is_rumour=bool(raw.get("is_rumour", False)),
            )
            self._news[(row.event_id, row.tick)] = row

    def recording(self) -> Recording:
        """Snapshot what has been sampled so far."""
        n = len(self._months)
        scalars = {k: np.asarray(v, dtype=float) for k, v in sorted(self._scalars.items()) if v}
        matrices = {k: np.stack(v) if v else np.zeros((0, 0)) for k, v in sorted(self._matrices.items())}
        news = tuple(self._news[k] for k in sorted(self._news))
        return Recording(
            codes=self._codes,
            months=np.asarray(self._months, dtype=np.int64),
            ticks=np.asarray(self._ticks, dtype=np.int64),
            scalars=scalars,
            matrices=matrices,
            news=news,
            meta={
                "schema_version": SCHEMA_VERSION,
                "seed": int(getattr(self._world, "seed", 0)),
                "n_months": n,
                "state_hash": self._world.state_hash(),
                "module_names": [str(getattr(m, "name", "")) for m in getattr(self._world, "modules", ())],
                "units": dict(sorted(SERIES_UNITS.items())),
            },
        )


def record_run(world: Any, n_ticks: int) -> Recording:
    """Step ``world`` for ``n_ticks`` days, sampling every tick. Returns the recording."""
    rec = RunRecorder(world)
    for _ in range(int(n_ticks)):
        world.step(1)
        rec.sample()
    return rec.recording()


def _sidecar(path: str | Path) -> tuple[Path, Path]:
    base = Path(path)
    return base.with_suffix(".npz"), base.with_suffix(".json")


def write_run(path: str | Path, rec: Recording) -> tuple[Path, Path]:
    """Write ``<stem>.npz`` (arrays) and ``<stem>.json`` (metadata + news). Returns both."""
    npz_path, json_path = _sidecar(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "months": rec.months,
        "ticks": rec.ticks,
        "codes": np.asarray(rec.codes, dtype="<U16"),
    }
    for name in rec.series_names():
        arrays[name] = rec.scalars[name]
    for name in rec.matrix_names():
        arrays[name] = rec.matrices[name]
    np.savez(npz_path, **arrays)
    payload = {
        **rec.meta,
        "codes": list(rec.codes),
        "news": [row.to_dict() for row in rec.news],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return npz_path, json_path


def read_series(path: str | Path, *, prefix: str = "") -> Recording:
    """Read any run-shaped ``.npz``. ``prefix`` selects one family (``"ns_"`` in the r1 golden).

    1-D float arrays whose length matches the row count become scalar series; arrays
    whose first axis matches become matrices. Anything else is ignored, so committed
    goldens load unchanged.
    """
    npz_path = Path(path).with_suffix(".npz")
    with np.load(npz_path, allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files}
    codes = tuple(str(c) for c in raw.get("codes", ()))
    selected = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)} if prefix else raw
    months = selected.get("months")
    n = int(months.shape[0]) if months is not None else _row_count(selected)
    scalars: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    for name in sorted(selected):
        arr = selected[name]
        if name in ("months", "ticks", "codes") or not np.issubdtype(arr.dtype, np.floating):
            continue
        if arr.ndim == 1 and arr.shape[0] == n:
            scalars[name] = arr.astype(float)
        elif arr.ndim >= 2 and arr.shape[0] == n:
            matrices[name] = arr.astype(float)
    if months is None:
        months = np.arange(1, n + 1, dtype=np.int64)
    ticks = selected.get("ticks")
    return Recording(
        codes=codes,
        months=np.asarray(months, dtype=np.int64),
        ticks=np.asarray(ticks if ticks is not None else months, dtype=np.int64),
        scalars=scalars,
        matrices=matrices,
        meta={"source": str(npz_path), "prefix": prefix},
    )


def _row_count(arrays: dict[str, np.ndarray]) -> int:
    """Modal first-axis length of the float arrays — the run's month count."""
    lengths: dict[int, int] = {}
    for arr in arrays.values():
        if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.floating):
            lengths[int(arr.shape[0])] = lengths.get(int(arr.shape[0]), 0) + 1
    if not lengths:
        return 0
    return max(sorted(lengths), key=lambda k: lengths[k])


def read_run(path: str | Path) -> Recording:
    """Read a recorder artifact written by :func:`write_run` (``.npz`` + ``.json``)."""
    npz_path, json_path = _sidecar(path)
    rec = read_series(npz_path)
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        news = tuple(
            NewsRow(
                tick=int(row.get("tick", 0)),
                event_id=str(row.get("event_id", "")),
                category=str(row.get("category", "")),
                headline=str(row.get("headline", "")),
                severity_hint=int(row.get("severity_hint", 0)),
                is_rumour=bool(row.get("is_rumour", False)),
            )
            for row in payload.get("news", ())
        )
        meta = {k: v for k, v in payload.items() if k != "news"}
        rec = Recording(
            codes=tuple(payload.get("codes", rec.codes)),
            months=rec.months,
            ticks=rec.ticks,
            scalars=rec.scalars,
            matrices=rec.matrices,
            news=news,
            meta=meta,
        )
    return rec
