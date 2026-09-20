"""Found a firm and acquire capacity (T5.03 / §5.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marketsim.core.errors import ConfigError, LedgerError
from marketsim.firms.accounts import firm_entity, post_founding_capital, register_firm
from marketsim.firms.firm import Firm, FirmRegistry, Plant
from marketsim.ledger.journal import Entry, Ledger, Tx


def npc_entity(region: str, sector: str) -> str:
    """NPC mass in one cell. Unitless id."""
    return f"NPC:{region}:{sector}"


def months_from_build_lag_q(build_lag_q: float) -> int:
    """Quarters → months (3 months / quarter)."""
    return max(1, int(round(3.0 * float(build_lag_q))))


@dataclass
class NpcCell:
    """NPC producer in one (region, sector) cell. ``capacity`` is real units / month."""

    region: str
    sector: str
    capacity: float
    capital_stock: float = 0.0  # cr; replacement cost of K

    @property
    def cell(self) -> tuple[str, str]:
        return (self.region, self.sector)

    @property
    def unit_replacement(self) -> float:
        """cr per unit of capacity."""
        if self.capacity <= 0:
            return 0.0
        return self.capital_stock / self.capacity


@dataclass
class CellBook:
    """NPC capacity by cell. Iteration is sorted (region, sector)."""

    cells: dict[tuple[str, str], NpcCell] = field(default_factory=dict)

    def get(self, region: str, sector: str) -> NpcCell:
        key = (region, sector)
        if key not in self.cells:
            raise ConfigError(f"unknown NPC cell {region}/{sector}")
        return self.cells[key]

    def total_capacity(self, region: str, sector: str, registry: FirmRegistry) -> float:
        npc = self.get(region, sector).capacity
        firm_k = 0.0
        for firm in registry:
            for plant in firm.plants:
                if plant.cell == (region, sector):
                    firm_k += plant.capacity
        return npc + firm_k

    def to_state(self) -> dict[str, Any]:
        rows = []
        for key in sorted(self.cells):
            c = self.cells[key]
            rows.append([c.region, c.sector, c.capacity, c.capital_stock])
        return {"cells": rows}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CellBook:
        book = cls()
        for region, sector, cap, k in state.get("cells", ()):
            book.cells[(str(region), str(sector))] = NpcCell(
                str(region), str(sector), float(cap), float(k)
            )
        return book


def split_npc_share(npc_capacity: float, npc_initial_share: float) -> tuple[float, float]:
    """Return (npc_kept, released_to_firms). ``npc_initial_share`` is dimensionless."""
    if not 0.0 <= npc_initial_share <= 1.0:
        raise ConfigError("npc_initial_share must be in [0, 1]")
    kept = float(npc_capacity) * float(npc_initial_share)
    return kept, float(npc_capacity) - kept


def found_firm(
    registry: FirmRegistry,
    ledger: Ledger,
    *,
    firm_id: str,
    operator: str,
    capital: float,
    source: str,
    tick: int = 0,
    tag: str = "equity_issue",
) -> Firm:
    """Register an empty firm and post founding capital (cr)."""
    register_firm(ledger, firm_id, operator=operator)
    post_founding_capital(ledger, firm_id=firm_id, amount=capital, source=source, tick=tick, tag=tag)
    firm = Firm(id=firm_id, operator=operator, cash=capital)
    registry.add(firm)
    return firm


def _plant_on(firm: Firm, region: str, sector: str) -> Plant:
    for plant in firm.plants:
        if plant.cell == (region, sector):
            return plant
    plant = Plant(region=region, sector=sector, capacity=0.0)
    firm.plants = (*firm.plants, plant)
    return plant


def buy_from_npc(
    book: CellBook,
    registry: FirmRegistry,
    ledger: Ledger,
    *,
    firm_id: str,
    region: str,
    sector: str,
    capacity: float,
    tick: int = 0,
) -> float:
    """Buy existing K from the NPC mass at replacement value (cr). Capacity is conserved."""
    if capacity <= 0:
        raise LedgerError("purchase capacity must be > 0")
    npc = book.get(region, sector)
    if capacity > npc.capacity + 1e-12:
        raise LedgerError(f"NPC {region}/{sector} has only {npc.capacity} to sell")
    unit = npc.unit_replacement
    cash = unit * capacity
    capital = npc.capital_stock * (capacity / npc.capacity) if npc.capacity else 0.0
    npc.capacity -= capacity
    npc.capital_stock -= capital
    firm = registry[firm_id]
    plant = _plant_on(firm, region, sector)
    plant.capacity += capacity
    firm.cash -= cash
    buyer = firm_entity(firm_id)
    seller = npc_entity(region, sector)
    if seller not in ledger.entities:
        ledger.register_entity(seller)
    ledger.post(
        Tx(
            tick,
            "capital_transfer",
            (
                Entry(buyer, "DEP", -cash),
                Entry(seller, "DEP", cash),
                Entry(seller, "CAPITAL", -capital),
                Entry(buyer, "CAPITAL", capital),
            ),
            memo=f"buy {capacity} {region}/{sector}",
        )
    )
    return cash


def start_greenfield(
    registry: FirmRegistry,
    ledger: Ledger,
    *,
    firm_id: str,
    region: str,
    sector: str,
    capacity: float,
    lag_m: int,
    unit_cost: float,
    setup_premium: float = 0.0,
    seller: str,
    tick: int = 0,
    new_cell: bool = False,
) -> float:
    """Pay replacement (cr) now; capacity arrives after ``lag_m`` months."""
    if capacity <= 0 or lag_m < 1 or unit_cost < 0:
        raise LedgerError("greenfield capacity, lag_m and unit_cost must be valid")
    premium = setup_premium if new_cell else 0.0
    cash = unit_cost * capacity * (1.0 + premium)
    firm = registry[firm_id]
    plant = _plant_on(firm, region, sector)
    plant.construction_pipeline = (*plant.construction_pipeline, (int(lag_m), float(capacity)))
    firm.cash -= cash
    buyer = firm_entity(firm_id)
    if seller not in ledger.entities:
        ledger.register_entity(seller)
    ledger.post(
        Tx(
            tick,
            "investment",
            (Entry(buyer, "DEP", -cash), Entry(seller, "DEP", cash)),
            memo=f"greenfield {region}/{sector}",
        )
    )
    return cash


def advance_construction(registry: FirmRegistry, *, months: int = 1) -> None:
    """Age every pipeline by ``months``; completed capacity comes online."""
    for firm in registry:
        updated: list[Plant] = []
        for plant in firm.plants:
            live = plant.capacity
            leftover: list[tuple[int, float]] = []
            for remaining, cap in plant.construction_pipeline:
                left = int(remaining) - int(months)
                if left <= 0:
                    live += cap
                else:
                    leftover.append((left, cap))
            updated.append(
                Plant(
                    region=plant.region,
                    sector=plant.sector,
                    capacity=live,
                    productivity=plant.productivity,
                    vintage=plant.vintage,
                    construction_pipeline=tuple(leftover),
                )
            )
        firm.plants = tuple(updated)
