"""AR(1) primitive shock bus (§2.9). ``ShockBus.inject`` is the only entry point."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from marketsim.core.config import Config
from marketsim.ledger.journal import Entry, Ledger, Tx

KIND_KEY = {
    "demand": "dem",
    "supply": "sup",
    "cost_push": "cost",
    "monetary": "mon",
    "risk_appetite": "risk",
    "fiscal": "fisc",
    "row": "row",
    "imp": "imp",
}
VECTOR_KINDS = frozenset({"supply", "cost_push"})
SCALAR_KINDS = frozenset(k for k in KIND_KEY if k not in VECTOR_KINDS)
DAYS_PER_MONTH = 21


def ar1_rho(persistence_q: float) -> float:
    """Monthly AR(1) coefficient ``ρ = exp(−1/(3·persistence_q))``. ``persistence_q≤0`` → 0."""
    if persistence_q <= 0:
        return 0.0
    return float(np.exp(-1.0 / (3.0 * persistence_q)))


def mid_month_weight(day: int, days_per_month: int = DAYS_PER_MONTH) -> float:
    """Remaining-month fraction ``(21 − d)/21`` with ``d`` the 0-based day of the month."""
    d = int(day)
    if d <= 0:
        return 1.0
    if d >= days_per_month:
        return 0.0
    return (days_per_month - d) / days_per_month


def _spec_dump(cfg: Config, kind: str) -> dict[str, Any]:
    spec = cfg.edges.shocks.get(kind)
    if spec is None:
        return {}
    return spec.model_dump()


class ShockBus:
    """Holds the seven primitives plus ROW/import indices. Writes into ``states`` (``eco.sh``)."""

    def __init__(self, cfg: Config, codes: tuple[str, ...]) -> None:
        assert cfg.dynamics is not None
        self.cfg = cfg
        self.codes = codes
        n = len(codes)
        self.states: dict[str, Any] = {
            "dem": 0.0,
            "cost": np.zeros(n),
            "mon": 0.0,
            "sup": np.zeros(n),
            "fisc": 0.0,
            "row": 0.0,
            "imp": 0.0,
            "risk": 0.0,
            "ds": 0.0,
        }
        self.rho: dict[str, float] = {k: 0.0 for k in self.states}
        self.recon_demand = np.zeros(n)
        self._default_rho: dict[str, float] = {}
        for kind, key in KIND_KEY.items():
            pq = float(_spec_dump(cfg, kind).get("persistence_q", 0.0))
            if kind == "imp":
                pq = float(_spec_dump(cfg, "cost_push").get("persistence_q", pq))
            self._default_rho[key] = ar1_rho(pq)
            self.rho[key] = self._default_rho[key]
        self.rho["ds"] = self._default_rho.get("risk", 0.0)

    @classmethod
    def from_config(cls, cfg: Config, codes: tuple[str, ...] | None = None) -> ShockBus:
        return cls(cfg, tuple(codes or cfg.codes))

    def sync(self) -> None:
        """Derived household-spread term: ``Δspread = 0.5 · z_risk`` (§2.9)."""
        self.states["ds"] = 0.5 * float(self.states["risk"])

    def inject(
        self,
        kind: str,
        magnitude: float,
        persistence_q: float | None = None,
        targets: list[str] | None = None,
        weight: float = 1.0,
        day: int = 0,
        days_per_month: int = DAYS_PER_MONTH,
        economy: Any | None = None,
        tick: int = 0,
        claims_to: list[str] | None = None,
        cat_exposure: float | None = None,
    ) -> float:
        """Add ``magnitude · weight · (21−d)/21`` to the named primitive. Catastrophe is one-off."""
        w = float(magnitude) * float(weight) * mid_month_weight(day, days_per_month)
        if kind == "catastrophe":
            return self._catastrophe(
                kappa=abs(float(magnitude)) * float(weight),
                targets=targets,
                claims_to=claims_to,
                exposure=cat_exposure,
                economy=economy,
                tick=tick,
            )
        if kind not in KIND_KEY:
            raise KeyError(f"unknown shock kind {kind!r}")
        key = KIND_KEY[kind]
        if persistence_q is not None:
            self.rho[key] = ar1_rho(float(persistence_q))
        else:
            self.rho[key] = self._default_rho.get(key, 0.0)
        if kind in VECTOR_KINDS:
            vec = self._target_weights(kind, targets) * w
            self.states[key] = np.asarray(self.states[key], dtype=float) + vec
            if kind == "cost_push":
                # Same goods move the import-price residual (§2.9).
                self.states["imp"] = float(self.states["imp"]) + float(np.max(vec))
                self.rho["imp"] = self.rho[key]
        else:
            self.states[key] = float(self.states[key]) + w
        self.sync()
        return w

    def decay(self) -> None:
        """Monthly AR(1): ``z ← ρ z``. Catastrophe reconstruction demand is consumed by the stepper."""
        for key, rho in self.rho.items():
            val = self.states[key]
            if isinstance(val, np.ndarray):
                self.states[key] = val * rho
            else:
                self.states[key] = float(val) * rho
        self.sync()

    def _target_weights(self, kind: str, targets: list[str] | None) -> np.ndarray:
        n = len(self.codes)
        weights = np.zeros(n)
        if targets is not None:
            for name in targets:
                weights[self.codes.index(name)] = 1.0
            return weights
        if kind == "cost_push":
            dumped = _spec_dump(self.cfg, "cost_push")
            raw = dumped.get("targets") or self.cfg.dynamics.shocks.cost_push_targets  # type: ignore[union-attr]
            for name, wgt in raw.items():
                weights[self.codes.index(name)] = float(wgt)
            return weights
        return np.ones(n)

    def _catastrophe(
        self,
        *,
        kappa: float,
        targets: list[str] | None,
        claims_to: list[str] | None,
        exposure: float | None,
        economy: Any | None,
        tick: int,
    ) -> float:
        if economy is None:
            raise ValueError("catastrophe inject requires economy=")
        dumped = _spec_dump(self.cfg, "catastrophe")
        tgt = tuple(targets or dumped.get("targets") or ("CONSTRUCT", "REALESTATE", "AUTOS"))
        route = tuple(claims_to or dumped.get("claims_to") or ("CONSTRUCT", "AUTOS", "HEALTH"))
        # ``cat_exposure`` is the insured fraction of destroyed replacement cost (spec §2.9).
        frac = 1.0 if exposure is None else float(exposure)
        idx = [economy.codes.index(c) for c in tgt]
        v = economy.real.flat(economy.real.v)
        p = np.asarray(economy.p, dtype=float)
        destroyed_value = 0.0
        for i in idx:
            d_k = kappa * float(economy.k[i])
            d_inv = kappa * float(economy.inv[i])
            destroyed_value += float(v[i] * d_k * p[i] + p[i] * d_inv)
            economy.k[i] *= 1.0 - kappa
            economy.inv[i] *= 1.0 - kappa
        claims = max(0.0, frac * destroyed_value)
        if claims > 0:
            post_catastrophe_claims(economy.ledger, claims, route, tick=tick)
            shares = np.array([1.0 / len(route) for _ in route])
            for c, sh in zip(route, shares, strict=True):
                i = economy.codes.index(c)
                self.recon_demand[i] += sh * claims / max(float(p[i]), 1e-12)
        return claims

    def consume_recon(self) -> np.ndarray:
        out = self.recon_demand.copy()
        self.recon_demand[:] = 0.0
        return out

    def to_state(self) -> dict[str, Any]:
        return {
            "states": {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in self.states.items()},
            "rho": dict(self.rho),
            "recon_demand": self.recon_demand.tolist(),
        }

    def from_state(self, state: dict[str, Any]) -> None:
        for k, v in state["states"].items():
            self.states[k] = np.asarray(v, dtype=float) if isinstance(v, list) else float(v)
        self.rho = {k: float(v) for k, v in state["rho"].items()}
        self.recon_demand = np.asarray(state["recon_demand"], dtype=float)
        self.sync()


def post_catastrophe_claims(
    ledger: Ledger,
    claims: float,
    claims_to: Iterable[str],
    *,
    tick: int,
    region: int = 0,
) -> None:
    """INSURANCE pays ``claims``; claimants receive the same sum. Tag ``insurance_claims``."""
    dest = list(claims_to)
    if not dest or claims == 0.0:
        return
    share = claims / len(dest)
    insurer = f"NPC:{region}:INSURANCE"
    entries = [Entry(insurer, "DEP", -claims)]
    entries.extend(Entry(f"NPC:{region}:{c}", "DEP", share) for c in dest)
    ledger.post(Tx(tick, "insurance_claims", tuple(entries)))
