"""Stateful protocol — every engine object that owns mutable state implements this."""

from __future__ import annotations

from typing import Any, Protocol


class Stateful(Protocol):
    def to_state(self) -> dict[str, Any]: ...

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Stateful: ...
