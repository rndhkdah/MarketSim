"""T4.12 / T5.15 — firm-level event hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FIRM_EVENT_IDS = (
    "plant_accident",
    "strike",
    "recall",
    "large_bankruptcy",
    "firm_plant_accident",
    "firm_strike",
    "firm_recall",
)


@dataclass
class FirmHookPayload:
    """Payload for a firm-targeted event. ``firms`` is a tuple of firm ids (empty = NPC cell)."""

    event_id: str
    tick: int
    firms: tuple[str, ...]
    sectors: tuple[str, ...]
    regions: tuple[str, ...]
    magnitude: float | None = None


@dataclass
class FirmHookBus:
    """Records dispatches. Phase 5 binds real plant/strike/recall/bankruptcy handlers."""

    calls: list[FirmHookPayload] = field(default_factory=list)

    def dispatch(self, payload: FirmHookPayload) -> None:
        self.calls.append(payload)

    def to_state(self) -> dict[str, Any]:
        return {"calls": [p.__dict__ for p in self.calls]}

    def from_state(self, state: dict[str, Any]) -> None:
        self.calls = [FirmHookPayload(**raw) for raw in state.get("calls", [])]


def payload_from_event(spec: Any, tick: int, *, magnitude: float | None = None) -> FirmHookPayload:
    targets = spec.targets or {}
    firms = targets.get("firms") or []
    sectors = targets.get("sectors") or []
    regions = targets.get("regions") or []
    if sectors == "all":
        sectors = []
    if regions == "all":
        regions = []
    return FirmHookPayload(
        event_id=spec.id,
        tick=int(tick),
        firms=tuple(str(x) for x in firms),
        sectors=tuple(str(x) for x in sectors),
        regions=tuple(str(x) for x in regions),
        magnitude=magnitude,
    )


def apply_firm_hook(
    payload: FirmHookPayload,
    registry: Any,
    ledger: Any,
    *,
    duration_m: int = 1,
) -> None:
    """Apply a firm event through plants / labour / inventory and the ledger."""
    from marketsim.firms.accounts import firm_entity
    from marketsim.ledger.journal import Entry, Tx

    mag = float(payload.magnitude or 0.10)
    for fid in payload.firms:
        firm = registry[fid]
        ent = firm_entity(fid)
        if payload.event_id in {"plant_accident", "firm_plant_accident"}:
            for plant in firm.plants:
                if payload.sectors and plant.sector not in payload.sectors:
                    continue
                if payload.regions and plant.region not in payload.regions:
                    continue
                lost = plant.capacity * mag
                plant.capacity -= lost
                break
        elif payload.event_id in {"strike", "firm_strike"}:
            firm.vacancies = 0.0
            parked = firm.employees
            firm.employees = 0.0
            firm.backlog[("strike", "labour")] = parked
            del duration_m
        elif payload.event_id in {"recall", "firm_recall"}:
            inv = sum(firm.finished_inventories.values())
            firm.finished_inventories.clear()
            hit = inv * mag if inv else mag
            if "DEP" in ledger.instruments and ent in ledger.entities:
                sink = "HH:0" if "HH:0" in ledger.entities else "GOVT"
                if sink not in ledger.entities:
                    ledger.register_entity(sink)
                ledger.post(
                    Tx(payload.tick, "insurance_claims", (Entry(ent, "DEP", -hit), Entry(sink, "DEP", hit)))
                )
        registry._firms[fid] = firm  # keep identity; plants mutated in place

