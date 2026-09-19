"""Dynamic instrument registry. Financial instruments must net to zero across entities."""

from __future__ import annotations

from typing import Any

DEFAULT_FINANCIAL = ("DEP", "LOAN", "GB_BILL", "GB_NOTE", "GB_BOND", "CORP_POOL", "CLOAN", "RES")
DEFAULT_REAL = ("CAPITAL", "INVENTORY", "HOUSING")
EQUITY_PREFIX = "EQ:"


class InstrumentRegistry:
    """Name → dense integer id. Financial vs real is fixed at registration."""

    def __init__(self) -> None:
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: list[str] = []
        self._financial: list[bool] = []

    def register(self, name: str, *, financial: bool | None = None) -> int:
        existing = self._name_to_id.get(name)
        if existing is not None:
            return existing
        if financial is None:
            financial = name.startswith(EQUITY_PREFIX) or name in DEFAULT_FINANCIAL
        idx = len(self._id_to_name)
        self._name_to_id[name] = idx
        self._id_to_name.append(name)
        self._financial.append(bool(financial))
        return idx

    def id(self, name: str) -> int:
        try:
            return self._name_to_id[name]
        except KeyError as exc:
            raise KeyError(f"unknown instrument {name!r}") from exc

    def name(self, idx: int) -> str:
        return self._id_to_name[idx]

    def is_financial(self, name: str) -> bool:
        return self._financial[self.id(name)]

    def is_financial_id(self, idx: int) -> bool:
        return self._financial[idx]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._name_to_id

    def __len__(self) -> int:
        return len(self._id_to_name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._id_to_name)

    @property
    def financial_names(self) -> tuple[str, ...]:
        return tuple(n for n, f in zip(self._id_to_name, self._financial, strict=True) if f)

    def to_state(self) -> dict[str, Any]:
        return {
            "names": list(self._id_to_name),
            "financial": list(self._financial),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> InstrumentRegistry:
        reg = cls()
        for name, fin in zip(state["names"], state["financial"], strict=True):
            reg.register(name, financial=bool(fin))
        return reg
