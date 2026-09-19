"""Columnar journal, position matrix, and balanced posting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from marketsim.core.errors import LedgerError
from marketsim.ledger.entities import EntityRegistry
from marketsim.ledger.instruments import (
    DEFAULT_FINANCIAL,
    DEFAULT_REAL,
    EQUITY_PREFIX,
    InstrumentRegistry,
)

FLOW_TAGS: tuple[str, ...] = (
    "consumption",
    "vat",
    "excise",
    "tariff",
    "subsidy",
    "govt_purchases",
    "investment",
    "residential",
    "exports",
    "imports",
    "intermediate",
    "wages",
    "transfers",
    "income_tax",
    "corp_tax",
    "interest_loans",
    "interest_deposits",
    "interest_bonds",
    "interest_reserves",
    "cb_remittance",
    "dividends",
    "loan_new",
    "loan_repay",
    "loan_writeoff",
    "bond_issue",
    "bond_redeem",
    "equity_issue",
    "equity_trade",
    "fees",
    "transaction_tax",
    "insurance_claims",
    "capital_transfer",
    "opening",
)


@dataclass(frozen=True)
class Entry:
    entity: str
    instrument: str
    amount: float


@dataclass
class Tx:
    tick: int
    tag: str
    entries: tuple[Entry, ...]
    memo: str = ""

    def __post_init__(self) -> None:
        self.tick = int(self.tick)
        self.entries = tuple(self.entries)


@dataclass
class Ledger:
    """Signed `pos[entity, instrument]`: + asset, − liability. Financial columns net to 0."""

    entities: EntityRegistry
    instruments: InstrumentRegistry
    pos: np.ndarray
    debug_journal: bool = False
    flow_tags: tuple[str, ...] = FLOW_TAGS
    _ticks: list[int] = field(default_factory=list)
    _tags: list[str] = field(default_factory=list)
    _ent: list[int] = field(default_factory=list)
    _inst: list[int] = field(default_factory=list)
    _amt: list[float] = field(default_factory=list)
    _debug: list[Tx] = field(default_factory=list)
    period_flows: dict[tuple[str, str, str], float] = field(default_factory=dict)
    period: int = 0

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=float)

    @classmethod
    def empty(cls, *, debug_journal: bool = False) -> Ledger:
        ents = EntityRegistry()
        inst = InstrumentRegistry()
        for name in DEFAULT_FINANCIAL:
            inst.register(name, financial=True)
        for name in DEFAULT_REAL:
            inst.register(name, financial=False)
        return cls(
            entities=ents,
            instruments=inst,
            pos=np.zeros((0, len(inst))),
            debug_journal=debug_journal,
        )

    @classmethod
    def from_yaml(cls, path: str | Path, *, codes: tuple[str, ...] | None = None, n_regions: int = 1) -> Ledger:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        led = cls.empty(debug_journal=bool(raw.get("debug_journal", False)))
        for name in raw.get("entities", {}).get("institutions", []):
            led.register_entity(name)
        codes = codes or ()
        for r in range(n_regions):
            led.register_entity(f"HH:{r}")
            for code in codes:
                led.register_entity(f"NPC:{r}:{code}")
                led.register_instrument(f"{EQUITY_PREFIX}{code}", financial=True)
        tags = raw.get("flow_tags")
        if tags:
            led.flow_tags = tuple(tags)
        return led

    def register_entity(self, name: str) -> int:
        if name in self.entities:
            return self.entities.id(name)
        idx = self.entities.register(name)
        if self.pos.shape[0] == idx:
            self.pos = np.vstack([self.pos, np.zeros((1, self.pos.shape[1]))]) if self.pos.size else np.zeros((1, self.pos.shape[1]))
        return idx

    def register_instrument(self, name: str, *, financial: bool | None = None) -> int:
        if name in self.instruments:
            return self.instruments.id(name)
        idx = self.instruments.register(name, financial=financial)
        if self.pos.shape[1] == idx:
            pad = np.zeros((self.pos.shape[0], 1))
            self.pos = np.hstack([self.pos, pad]) if self.pos.size else np.zeros((self.pos.shape[0], 1))
        return idx

    def position(self, entity: str, instrument: str) -> float:
        return float(self.pos[self.entities.id(entity), self.instruments.id(instrument)])

    def post(self, tx: Tx) -> None:
        by_inst: dict[str, float] = defaultdict(float)
        for e in tx.entries:
            if e.instrument not in self.instruments:
                self.register_instrument(e.instrument)
            if self.instruments.is_financial(e.instrument):
                by_inst[e.instrument] += float(e.amount)
        for inst, total in by_inst.items():
            if abs(total) > 1e-9:
                raise LedgerError(
                    f"unbalanced Tx tag={tx.tag!r} instrument={inst}: sum={total}",
                )
        for e in tx.entries:
            if e.entity not in self.entities:
                self.register_entity(e.entity)
            ei = self.entities.id(e.entity)
            ii = self.instruments.id(e.instrument)
            self.pos[ei, ii] += float(e.amount)
            self._ticks.append(tx.tick)
            self._tags.append(tx.tag)
            self._ent.append(ei)
            self._inst.append(ii)
            self._amt.append(float(e.amount))
        self._accumulate_flows(tx)
        if self.debug_journal:
            self._debug.append(tx)

    def _accumulate_flows(self, tx: Tx) -> None:
        by_inst: dict[str, list[Entry]] = defaultdict(list)
        for e in tx.entries:
            if self.instruments.is_financial(e.instrument):
                by_inst[e.instrument].append(e)
        for entries in by_inst.values():
            payers = [e for e in entries if e.amount < 0]
            payees = [e for e in entries if e.amount > 0]
            if len(payers) == 1 and len(payees) == 1:
                key = (payers[0].entity, payees[0].entity, tx.tag)
                self.period_flows[key] = self.period_flows.get(key, 0.0) + float(payees[0].amount)

    def close_period(self) -> dict[tuple[str, str, str], float]:
        flows = dict(self.period_flows)
        self.period_flows.clear()
        self.period += 1
        return flows

    def to_state(self) -> dict[str, Any]:
        return {
            "entities": self.entities.to_state(),
            "instruments": self.instruments.to_state(),
            "pos": self.pos.copy(),
            "debug_journal": self.debug_journal,
            "flow_tags": list(self.flow_tags),
            "ticks": list(self._ticks),
            "tags": list(self._tags),
            "ent": list(self._ent),
            "inst": list(self._inst),
            "amt": list(self._amt),
            "period_flows": {f"{a}|{b}|{t}": v for (a, b, t), v in self.period_flows.items()},
            "period": self.period,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Ledger:
        led = cls(
            entities=EntityRegistry.from_state(state["entities"]),
            instruments=InstrumentRegistry.from_state(state["instruments"]),
            pos=np.asarray(state["pos"], dtype=float),
            debug_journal=bool(state.get("debug_journal", False)),
            flow_tags=tuple(state.get("flow_tags", FLOW_TAGS)),
            _ticks=list(state.get("ticks", [])),
            _tags=list(state.get("tags", [])),
            _ent=list(state.get("ent", [])),
            _inst=list(state.get("inst", [])),
            _amt=list(state.get("amt", [])),
            period_flows={
                tuple(k.split("|")): float(v)  # type: ignore[misc]
                for k, v in state.get("period_flows", {}).items()
            },
            period=int(state.get("period", 0)),
        )
        return led
