"""Dynamic entity registry. Integer ids are stable once assigned."""

from __future__ import annotations

from typing import Any


class EntityRegistry:
    """Name → dense integer id. New registrations append; existing ids never move."""

    def __init__(self) -> None:
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: list[str] = []

    def register(self, name: str) -> int:
        existing = self._name_to_id.get(name)
        if existing is not None:
            return existing
        idx = len(self._id_to_name)
        self._name_to_id[name] = idx
        self._id_to_name.append(name)
        return idx

    def id(self, name: str) -> int:
        try:
            return self._name_to_id[name]
        except KeyError as exc:
            raise KeyError(f"unknown entity {name!r}") from exc

    def name(self, idx: int) -> str:
        return self._id_to_name[idx]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._name_to_id

    def __len__(self) -> int:
        return len(self._id_to_name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._id_to_name)

    def to_state(self) -> dict[str, Any]:
        return {"names": list(self._id_to_name)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> EntityRegistry:
        reg = cls()
        for name in state["names"]:
            reg.register(name)
        return reg
