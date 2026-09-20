"""Live-state helpers for ``(R, …)`` arrays. ``R = 1`` callers see the national slice."""

from __future__ import annotations

from typing import Any

import numpy as np


class RegionalArray:
    """Descriptor: stored as ``_{name}`` with leading region axis; ``R = 1`` returns slice 0."""

    def __set_name__(self, owner: type, name: str) -> None:
        self.public = name
        self.priv = f"_{name}"

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            return self
        arr = getattr(obj, self.priv)
        return arr[0] if obj.R == 1 else arr

    def __set__(self, obj: Any, value: Any) -> None:
        arr = getattr(obj, self.priv)
        value = np.asarray(value, dtype=float)
        if value.shape == arr.shape:
            arr[...] = value
            return
        if obj.R == 1 and value.shape == arr.shape[1:]:
            arr[0] = value
            return
        if obj.R == 1 and arr.shape == (1,) and value.ndim == 0:
            arr[0] = float(value)
            return
        raise ValueError(f"{self.priv} expected {arr.shape} or an R=1 slice, got {value.shape}")


def as_rs(arr: np.ndarray, r: int, *tail: int) -> np.ndarray:
    """View ``arr`` as ``(R, *tail)``. Accepts an already-shaped array or an R=1 slice."""
    a = np.asarray(arr, dtype=float)
    want = (r, *tail)
    if a.shape == want:
        return a
    if r == 1 and a.shape == tail:
        return a.reshape(want)
    raise ValueError(f"cannot view shape {a.shape} as {want}")
