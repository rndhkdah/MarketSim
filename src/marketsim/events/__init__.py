"""Event catalog (Phase 4)."""

from marketsim.events.firm_hooks import FIRM_EVENT_IDS, FirmHookBus, FirmHookPayload
from marketsim.events.schema import (
    CATEGORIES,
    PRIMITIVES,
    EventSpec,
    load_catalog,
    load_event,
    parse_extra_variable,
)

__all__ = [
    "CATEGORIES",
    "FIRM_EVENT_IDS",
    "FirmHookBus",
    "FirmHookPayload",
    "PRIMITIVES",
    "EventSpec",
    "load_catalog",
    "load_event",
    "parse_extra_variable",
]
