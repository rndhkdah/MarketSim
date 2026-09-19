"""Erlang(k) delay chains — mass-conserving and signal-smoother forms.

Per-stage exit probability ``a = k / (mean_m + k)`` so the mean lag is exactly
``k (1-a)/a = mean_m`` months. ``mean_m = 0`` is a pass-through.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _exit_prob(k: int, mean_m: np.ndarray) -> np.ndarray:
    return k / (mean_m + k)


class ErlangChain:
    """Mass-conserving pipeline of goods / capacity / money."""

    def __init__(self, k: int, mean_m: float | np.ndarray, shape: tuple[int, ...]) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self.shape = tuple(shape)
        mean = np.asarray(mean_m, dtype=float) * np.ones(shape)
        if np.any(mean < 0):
            raise ValueError("mean_m must be >= 0")
        self.mean_m = mean
        self.a = _exit_prob(self.k, mean)
        self.passthrough = np.all(mean == 0)
        self.q = np.zeros((self.k,) + self.shape)

    def push(self, inflow: np.ndarray | float) -> np.ndarray:
        x = np.asarray(inflow, dtype=float) * np.ones(self.shape)
        if self.passthrough:
            return x
        for i in range(self.k):
            self.q[i] += x
            x = self.a * self.q[i]
            self.q[i] -= x
        return x

    def seed(self, flow: np.ndarray | float) -> None:
        """Fill each stage so a constant inflow `flow` is a fixed point."""
        f = np.asarray(flow, dtype=float) * np.ones(self.shape)
        if self.passthrough:
            self.q[:] = 0.0
            return
        # q_i = flow * (1-a)/a  (same fill every stage)
        fill = f * (1.0 - self.a) / self.a
        for i in range(self.k):
            self.q[i] = fill

    def content(self) -> np.ndarray:
        return self.q.sum(axis=0)

    def to_state(self) -> dict[str, Any]:
        return {"k": self.k, "mean_m": self.mean_m, "q": self.q.copy()}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ErlangChain:
        chain = cls(int(state["k"]), state["mean_m"], tuple(np.asarray(state["q"]).shape[1:]))
        chain.q = np.asarray(state["q"], dtype=float)
        return chain


class ErlangSmoother:
    """Signal form: ``s_i += a (x - s_i)``. Does not conserve mass."""

    def __init__(self, k: int, mean_m: float | np.ndarray, init: np.ndarray | float) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        init_arr = np.asarray(init, dtype=float)
        self.k = int(k)
        self.shape = init_arr.shape
        mean = np.asarray(mean_m, dtype=float) * np.ones(self.shape)
        if np.any(mean < 0):
            raise ValueError("mean_m must be >= 0")
        self.mean_m = mean
        self.a = _exit_prob(self.k, mean)
        self.passthrough = np.all(mean == 0)
        self.s = np.array([init_arr.copy() for _ in range(self.k)])

    def push(self, u: np.ndarray | float) -> np.ndarray:
        x = np.asarray(u, dtype=float) * np.ones(self.shape)
        if self.passthrough:
            return x
        for i in range(self.k):
            self.s[i] += self.a * (x - self.s[i])
            x = self.s[i]
        return x.copy()

    def to_state(self) -> dict[str, Any]:
        return {"k": self.k, "mean_m": self.mean_m, "s": self.s.copy()}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> ErlangSmoother:
        s = np.asarray(state["s"], dtype=float)
        sm = cls(int(state["k"]), state["mean_m"], s[0])
        sm.s = s
        return sm
