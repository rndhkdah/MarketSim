"""Tick clock and a deterministic event queue."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from marketsim.core.calendar import Calendar


@dataclass(order=True, slots=True)
class _Item:
    timestamp: int
    priority: int
    sequence: int
    payload: Any = field(compare=False)


class EventQueue:
    """Min-heap keyed (timestamp, priority, sequence). Lower priority value fires first."""

    def __init__(self) -> None:
        self._heap: list[_Item] = []
        self._seq = 0

    def schedule(self, timestamp: int, payload: Any, priority: int = 0) -> int:
        seq = self._seq
        self._seq += 1
        heapq.heappush(self._heap, _Item(timestamp, priority, seq, payload))
        return seq

    def pop_due(self, t: int) -> list[Any]:
        out: list[Any] = []
        while self._heap and self._heap[0].timestamp <= t:
            out.append(heapq.heappop(self._heap).payload)
        return out

    def to_state(self) -> dict[str, Any]:
        items = sorted(self._heap)
        return {
            "seq": self._seq,
            "items": [
                {
                    "timestamp": it.timestamp,
                    "priority": it.priority,
                    "sequence": it.sequence,
                    "payload": it.payload,
                }
                for it in items
            ],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> EventQueue:
        q = cls()
        q._seq = int(state["seq"])
        for raw in state["items"]:
            heapq.heappush(
                q._heap,
                _Item(
                    int(raw["timestamp"]),
                    int(raw["priority"]),
                    int(raw["sequence"]),
                    raw["payload"],
                ),
            )
        return q


@dataclass
class Clock:
    """Advances ticks and owns the due-queue drain."""

    calendar: Calendar = field(default_factory=Calendar)
    queue: EventQueue = field(default_factory=EventQueue)
    tick: int = 0

    def advance(self) -> tuple[int, list[Any]]:
        due = self.queue.pop_due(self.tick)
        t = self.tick
        self.tick += 1
        return t, due

    def to_state(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "days_per_month": self.calendar.days_per_month,
            "queue": self.queue.to_state(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Clock:
        return cls(
            calendar=Calendar(days_per_month=int(state["days_per_month"])),
            queue=EventQueue.from_state(state["queue"]),
            tick=int(state["tick"]),
        )
