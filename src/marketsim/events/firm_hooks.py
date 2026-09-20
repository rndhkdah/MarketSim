"""T4.12 — firm-level event hooks (no-op until Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FIRM_EVENT_IDS = ("plant_accident", "strike", "recall", "large_bankruptcy")


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
