"""Module protocol and tick-pipeline phases (master plan §7)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol


class Phase(str, Enum):
    EVENTS = "EVENTS"
    INGEST = "INGEST"
    REAL = "REAL"
    SETTLE = "SETTLE"
    VALUE = "VALUE"
    MARKET = "MARKET"
    PUBLISH = "PUBLISH"


PIPELINE: tuple[Phase, ...] = (
    Phase.EVENTS,
    Phase.INGEST,
    Phase.REAL,
    Phase.SETTLE,
    Phase.VALUE,
    Phase.MARKET,
    Phase.PUBLISH,
)


class Module(Protocol):
    name: str

    def reset(self, ctx: Any) -> None: ...

    def on_phase(self, ctx: Any, phase: Phase) -> None: ...

    def to_state(self) -> dict[str, Any]: ...

    def from_state(self, state: dict[str, Any]) -> None: ...
