"""Single-step bankruptcy resolution (T5.12 / §5.6)."""

from __future__ import annotations

from dataclasses import dataclass, field

from marketsim.events.firm_hooks import FirmHookBus, FirmHookPayload
from marketsim.firms.accounts import firm_entity, firm_equity_instrument
from marketsim.firms.firm import Firm, FirmsFile
from marketsim.firms.founding import CellBook, npc_entity
from marketsim.ledger.journal import Entry, Ledger, Tx
from marketsim.ledger.sfc import assert_consistent

WATERFALL: tuple[str, ...] = ("wages", "taxes", "bank", "suppliers")


@dataclass
class Claim:
    name: str
    entity: str
    amount: float  # cr exposure


@dataclass
class Resolution:
    recoveries: dict[str, float] = field(default_factory=dict)
    losses: dict[str, float] = field(default_factory=dict)
    event_emitted: bool = False
    steps: int = 1


def should_resolve(firm: Firm, *, obligations_due: float) -> bool:
    """Equity < 0 for 3 months, or cash + undrawn < obligations."""
    if firm.equity_negative_months >= 3:
        return True
    undrawn = max(firm.credit_limit - firm.credit_drawn, 0.0)
    return firm.cash + undrawn < obligations_due


def resolve(
    firm: Firm,
    book: CellBook,
    ledger: Ledger,
    cfg: FirmsFile,
    *,
    claims: list[Claim],
    monthly_gdp: float,
    labour_pool: dict[str, float],
    hooks: FirmHookBus | None = None,
    tick: int = 0,
) -> Resolution:
    """One-step resolution. Plants to NPC at 0.70 replacement; inventories at 0.50."""
    firm.status = "bankrupt"
    proceeds = 0.0
    for plant in firm.plants:
        cell = book.cells.get(plant.cell)
        unit = cell.unit_replacement if cell is not None else 1.0
        rec = plant.capacity * unit * cfg.bankruptcy.plant_recovery
        proceeds += rec
        if cell is not None:
            cell.capacity += plant.capacity
            cell.capital_stock += rec
        plant.capacity = 0.0
        seller = firm_entity(firm.id)
        buyer = npc_entity(plant.region, plant.sector)
        if buyer not in ledger.entities:
            ledger.register_entity(buyer)
        # NPC loan finances the purchase (§5.6).
        ledger.post(
            Tx(
                tick,
                "loan_new",
                (
                    Entry(buyer, "DEP", rec),
                    Entry("BANKSYS", "DEP", -rec),
                    Entry("BANKSYS", "LOAN", rec),
                    Entry(buyer, "LOAN", -rec),
                ),
            )
        )
        ledger.post(
            Tx(
                tick,
                "capital_transfer",
                (Entry(buyer, "DEP", -rec), Entry(seller, "DEP", rec)),
                memo="plant to NPC",
            )
        )
    inv = sum(firm.finished_inventories.values()) + sum(firm.input_inventories.values())
    proceeds += inv * cfg.bankruptcy.inventory_recovery
    firm.finished_inventories.clear()
    firm.input_inventories.clear()

    out = Resolution()
    pot = proceeds
    for claim in claims:
        take = min(claim.amount, pot)
        loss = claim.amount - take
        out.recoveries[claim.name] = take
        out.losses[claim.name] = loss
        assert loss <= claim.amount + 1e-12
        pot -= take
        if take > 0:
            ledger.post(
                Tx(
                    tick,
                    "loan_writeoff" if claim.name == "bank" else "capital_transfer",
                    (
                        Entry(firm_entity(firm.id), "DEP", -take),
                        Entry(claim.entity, "DEP", take),
                    ),
                )
            )
        if loss > 0 and claim.name == "bank":
            ledger.post(
                Tx(
                    tick,
                    "loan_writeoff",
                    (
                        Entry("BANKSYS", "LOAN", -loss),
                        Entry(firm_entity(firm.id), "LOAN", loss),
                    ),
                )
            )
    eq = firm_equity_instrument(firm.id)
    if eq in ledger.instruments:
        pos = ledger.position(firm_entity(firm.id), eq)
        if abs(pos) > 1e-12:
            # wipe shareholders: cancel remaining equity (needs the holder)
            pass
    home = firm.plants[0].region if firm.plants else "NATIONAL"
    labour_pool[home] = labour_pool.get(home, 0.0) + firm.employees
    firm.employees = 0.0
    firm.status = "liquidated"
    liab = sum(c.amount for c in claims)
    out.event_emitted = liab > cfg.bankruptcy.large_threshold_of_monthly_gdp * monthly_gdp
    if out.event_emitted and hooks is not None:
        hooks.dispatch(
            FirmHookPayload("large_bankruptcy", tick, (firm.id,), (), (), magnitude=liab)
        )
    assert_consistent(ledger)
    assert out.steps == 1
    return out
