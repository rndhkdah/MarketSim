"""T4.03 — ``effects_extra`` shapes and setters (§4.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marketsim.events.compose import sample_dist
from marketsim.events.schema import DistSpec, ExtraEffect, parse_extra_variable
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent, net_financial_assets
from marketsim.real.credit import write_off_bank_equity

# Decay time-constant as a fraction of duration (§4.2 shape family).
DECAY_TAU_FRAC = 3.0


def shape_factor(shape: str, age_days: int, duration: int) -> float:
    """Weight in ``[0, 1]``. Zero at ``age >= duration`` so every shape reverts.

    Units: ``age_days`` and ``duration`` are days (ticks).
    ``step`` / ``pulse``: box on ``[0, duration)``.
    ``ramp``: linear ``(age+1)/duration`` on that window.
    ``decay``: ``exp(−age / (duration/3))`` truncated at ``duration``.
    """
    if age_days < 0:
        return 0.0
    if duration <= 0:
        return 1.0 if age_days == 0 else 0.0
    if age_days >= duration:
        return 0.0
    if shape in ("step", "pulse"):
        return 1.0
    if shape == "ramp":
        return float(age_days + 1) / float(duration)
    if shape == "decay":
        tau = float(duration) / DECAY_TAU_FRAC
        return float(math.exp(-float(age_days) / tau))
    raise ValueError(f"unknown extra shape {shape!r}")


@dataclass
class LiveEffect:
    stem: str
    index: str | None
    shape: str
    magnitude: float
    duration: int
    start_tick: int
    posted: bool = False


@dataclass
class ExtraEffects:
    """Active extra-effect overlay. Rest values: multipliers 1, additives 0."""

    live: list[LiveEffect] = field(default_factory=list)
    labour_mult: dict[str, float] = field(default_factory=dict)
    link_capacity_mult: dict[str, float] = field(default_factory=dict)
    link_cost_add: dict[str, float] = field(default_factory=dict)
    world_demand: float = 0.0
    import_price: dict[str, float] = field(default_factory=dict)
    collateral_mult: float = 1.0
    vat: float = 0.0
    sentiment: float = 0.0
    capex_pref: dict[str, float] = field(default_factory=dict)
    want_shift: dict[str, float] = field(default_factory=dict)
    _lf0: np.ndarray | None = None
    _row0: float | None = None
    _imp0: float | None = None

    def start(
        self,
        extras: list[ExtraEffect],
        rng: np.random.Generator,
        tick: int,
        *,
        economy: Any | None = None,
    ) -> list[LiveEffect]:
        """Sample magnitudes and append live effects. Ledger stems post once."""
        started: list[LiveEffect] = []
        for spec in extras:
            mag = (
                sample_dist(spec.magnitude, rng)
                if isinstance(spec.magnitude, DistSpec)
                else float(spec.magnitude)
            )
            stem, idx = parse_extra_variable(spec.variable)
            item = LiveEffect(stem, idx, spec.shape, mag, int(spec.duration), int(tick))
            if stem in ("bank_equity", "vat") and economy is not None:
                _post_ledger(stem, mag, economy, tick)
                item.posted = True
            self.live.append(item)
            started.append(item)
        self.apply(tick, economy)
        return started

    def apply(self, tick: int, economy: Any | None = None) -> None:
        """Recompute overlays from live effects and write setters. Expired effects drop."""
        remaining: list[LiveEffect] = []
        labour: dict[str, float] = {}
        link_c: dict[str, float] = {}
        link_k: dict[str, float] = {}
        world = 0.0
        imp: dict[str, float] = {}
        coll = 1.0
        vat = 0.0
        sent = 0.0
        capex: dict[str, float] = {}
        wants: dict[str, float] = {}
        for item in self.live:
            w = shape_factor(item.shape, int(tick) - item.start_tick, item.duration)
            if w == 0.0 and int(tick) - item.start_tick >= max(item.duration, 1):
                continue
            remaining.append(item)
            add = item.magnitude * w
            if item.stem == "labour_supply":
                key = item.index or "*"
                labour[key] = labour.get(key, 1.0) * (1.0 + add)
            elif item.stem == "link_capacity":
                key = item.index or "*"
                link_c[key] = link_c.get(key, 1.0) * (1.0 + add)
            elif item.stem == "link_cost":
                key = item.index or "*"
                link_k[key] = link_k.get(key, 0.0) + add
            elif item.stem == "world_demand":
                world += add
            elif item.stem == "import_price":
                key = item.index or "*"
                imp[key] = imp.get(key, 0.0) + add
            elif item.stem == "collateral_value":
                coll *= 1.0 + add
            elif item.stem == "vat":
                vat += add
            elif item.stem == "sentiment":
                sent += add
            elif item.stem == "capex_preference":
                key = item.index or "*"
                capex[key] = capex.get(key, 0.0) + add
            elif item.stem == "want_shift":
                key = item.index or "*"
                wants[key] = wants.get(key, 0.0) + add
        self.live = remaining
        self.labour_mult = labour
        self.link_capacity_mult = link_c
        self.link_cost_add = link_k
        self.world_demand = world
        self.import_price = imp
        self.collateral_mult = coll
        self.vat = vat
        self.sentiment = sent
        self.capex_pref = capex
        self.want_shift = wants
        if economy is not None:
            self._write_economy(economy, tick)

    def _write_economy(self, economy: Any, tick: int) -> None:
        if self._lf0 is None:
            self._lf0 = np.asarray(economy.lf, dtype=float).copy()
        lf = np.asarray(self._lf0, dtype=float).copy()
        nat = self.labour_mult.get("*", 1.0)
        lf = lf * nat
        for key, mult in self.labour_mult.items():
            if key == "*":
                continue
            idx = _region_index(economy, key)
            if idx is not None and lf.ndim >= 1 and idx < lf.shape[0]:
                lf[idx] = self._lf0[idx] * mult
        economy.lf = lf
        bus = getattr(economy, "bus", None)
        if bus is not None:
            if self._row0 is None:
                self._row0 = float(bus.states["row"])
            if self._imp0 is None:
                self._imp0 = float(bus.states["imp"])
            bus.states["row"] = float(self._row0) + self.world_demand
            extra_imp = self.import_price.get("*", 0.0)
            if extra_imp == 0.0 and self.import_price:
                extra_imp = max(self.import_price.values())
            bus.states["imp"] = float(self._imp0) + extra_imp
            if bus.want_shifts is not None:
                names = bus.want_shifts.names
                rest = np.ones_like(bus.want_shifts.values)
                for key, add in self.want_shift.items():
                    if key == "*":
                        rest *= 1.0 + add
                    elif key in names:
                        q = names.index(key)
                        rest[:, q] *= 1.0 + add
                bus.want_shifts.values = rest
        govt = getattr(getattr(economy, "policy", None), "govt", None)
        if govt is not None:
            rank = {"autopilot": 0, "event": 1, "scripted": 2, "agent": 3}
            holder = govt.source_of.get("vat", "autopilot")
            if abs(self.vat) > 1e-15 and rank.get(holder, 0) <= 1:
                govt.effective["vat"] = float(self.vat)
                govt.source_of["vat"] = "event"
            elif abs(self.vat) <= 1e-15 and holder == "event":
                govt.effective.pop("vat", None)
                govt.source_of.pop("vat", None)
        del tick

    def to_state(self) -> dict[str, Any]:
        return {
            "live": [item.__dict__ for item in self.live],
            "labour_mult": dict(self.labour_mult),
            "world_demand": self.world_demand,
            "collateral_mult": self.collateral_mult,
            "vat": self.vat,
            "sentiment": self.sentiment,
        }

    def from_state(self, state: dict[str, Any]) -> None:
        self.live = [LiveEffect(**raw) for raw in state.get("live", [])]
        self.labour_mult = dict(state.get("labour_mult", {}))
        self.world_demand = float(state.get("world_demand", 0.0))
        self.collateral_mult = float(state.get("collateral_mult", 1.0))
        self.vat = float(state.get("vat", 0.0))
        self.sentiment = float(state.get("sentiment", 0.0))


def _region_index(economy: Any, key: str) -> int | None:
    if key.isdigit():
        return int(key)
    codes = getattr(getattr(economy, "geom", None), "codes", None)
    if codes and key in codes:
        return list(codes).index(key)
    return None


def _post_ledger(stem: str, magnitude: float, economy: Any, tick: int) -> None:
    ledger: Ledger = economy.ledger
    if stem == "bank_equity":
        frac = abs(float(magnitude))
        write_off_bank_equity(ledger, np.asarray(economy.debt, dtype=float), economy.codes, frac, tick=tick)
        economy.bank_equity = float(net_financial_assets(ledger)[ledger.entities.id("BANKSYS")])
    elif stem == "vat":
        # One-off balanced VAT posting (cr). Rate path is the lever; this keeps SFC in the unit test.
        amt = float(magnitude)
        if abs(amt) > 1e-14:
            hh = "HH:0"
            ledger.post(Tx(tick, "vat", (Entry(hh, "DEP", -amt), Entry("GOVT", "DEP", amt))))
    if getattr(economy, "check_sfc", False):
        assert_consistent(ledger, tick)
