"""Earnings expectations from public information (T6.02 / §6.3).

``E^e_j`` is the drift-compensated 12-month EMA of *published* sector after-tax
profit. Units: cr/month (same as the published monthly profit flow; not
annualised). ``g_lr,j = 0.1 · published growth`` is an annual decimal.
Only information public at *t* is used — never unpublished / true series.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from marketsim.layer1.build_io import CODES

TAU_EE_M = 12.0  # §6.3: 12-month EMA of published after-tax profit
G_LR_ANCHOR = 0.1  # §6.3: g_lr,j = 0.1 · published growth


def drift_compensated_ema(
    s: np.ndarray,
    x: np.ndarray,
    pi_e: float,
    tau_m: float = TAU_EE_M,
) -> np.ndarray:
    """One step of ``s ← s·g + (x − s·g)/τ`` with ``g = exp(π_e/12)``.

    Parameters
    ----------
    s:
        Previous smoother state (cr/month), shape ``(n,)``.
    x:
        Published monthly after-tax profit (cr/month), shape ``(n,)``.
    pi_e:
        Expected inflation, annual decimal.
    tau_m:
        Smoother time constant in months (default ``TAU_EE_M`` = 12).

    Returns
    -------
    Updated ``E^e`` (cr/month), shape ``(n,)``.
    """
    s_arr = np.asarray(s, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    # annual π_e → monthly drift factor; 12 months/year (master plan §6, §6.3)
    g = float(np.exp(float(pi_e) / 12.0))
    drifted = s_arr * g
    return drifted + (x_arr - drifted) / float(tau_m)


def anchored_g_lr(published_growth: np.ndarray) -> np.ndarray:
    """``g_lr = G_LR_ANCHOR · published growth``. Annual decimal, shape ``(n,)``."""
    return G_LR_ANCHOR * np.asarray(published_growth, dtype=float)


def read_published_profit(source: Any) -> np.ndarray:
    """Published after-tax profit (cr/month). Never reads ``truth_*`` / ``profit``."""
    return np.asarray(source.published_profit, dtype=float)


def read_published_growth(source: Any) -> np.ndarray:
    """Published real growth (annual decimal). Never reads ``truth_*`` / ``growth``."""
    return np.asarray(source.published_growth, dtype=float)


def published_vintage(vintages: Mapping[int, Any], month: int) -> np.ndarray | None:
    """Published vector for ``month``, or ``None`` if that month is unpublished.

    Unpublished months are absent keys and cannot leak.
    """
    if month not in vintages:
        return None
    return np.asarray(vintages[month], dtype=float)


def _published_map(source: Any) -> Mapping[int, Any]:
    """Month→array map of published vintages. Never falls back to ``truth``."""
    if isinstance(source, Mapping):
        return source
    published = source.published
    if not isinstance(published, Mapping):
        raise TypeError("published vintages must be a month→array map")
    return published


def _published_growth_src(source: Any, published_growth: np.ndarray | Mapping[int, Any] | None) -> Any:
    if published_growth is not None:
        return published_growth
    if isinstance(source, Mapping):
        raise TypeError("published_growth is required when vintages is a mapping")
    return source.published_growth


class EarningsExpectations:
    """``E^e`` and ``g_lr`` from information public at *t* (§6.3).

    Vector length is ``len(CODES)`` (18) or the caller-supplied ``n`` / ``ee0``.
    State is only ``ee`` (cr/month) and ``g_lr`` (annual decimal).
    """

    def __init__(self, ee0: np.ndarray | None = None, *, n: int | None = None) -> None:
        if ee0 is not None:
            ee = np.asarray(ee0, dtype=float)
            if ee.ndim != 1:
                raise ValueError("ee0 must be a 1-d sector vector")
            if n is not None and int(n) != ee.size:
                raise ValueError(f"n={n} does not match ee0 length {ee.size}")
            self.ee = ee.copy()
        else:
            self.ee = np.zeros(int(n) if n is not None else len(CODES), dtype=float)
        self.g_lr = np.zeros(self.ee.size, dtype=float)

    def update(
        self,
        published_profit: np.ndarray,
        published_growth: np.ndarray,
        pi_e: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance one month from caller-supplied published arrays only.

        Parameters
        ----------
        published_profit:
            After-tax profit (cr/month), shape ``(n,)``.
        published_growth:
            Real growth, annual decimal, shape ``(n,)``.
        pi_e:
            Expected inflation, annual decimal.

        Returns
        -------
        ``(E^e, g_lr)`` — cr/month and annual decimal, each shape ``(n,)``.
        """
        x = np.asarray(published_profit, dtype=float)
        growth = np.asarray(published_growth, dtype=float)
        if x.shape != self.ee.shape or growth.shape != self.ee.shape:
            raise ValueError(
                f"published vectors must have shape {(self.ee.size,)}, got {x.shape} and {growth.shape}"
            )
        self.ee = drift_compensated_ema(self.ee, x, pi_e, TAU_EE_M)
        self.g_lr = anchored_g_lr(growth)
        return self.ee, self.g_lr

    def update_from_books(self, books: Any, pi_e: float) -> tuple[np.ndarray, np.ndarray]:
        """Update from a books object. Reads only ``published_profit`` / ``published_growth``."""
        return self.update(read_published_profit(books), read_published_growth(books), pi_e)

    def update_from_vintages(
        self,
        vintages: Any,
        month: int,
        pi_e: float,
        published_growth: np.ndarray | Mapping[int, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Advance if ``month`` is in the published map; else ``None`` (no state change).

        ``vintages`` is a month→array mapping or an object with ``.published``
        and ``.published_growth``. A ``.truth`` table is never read.
        """
        profit_map = _published_map(vintages)
        x = published_vintage(profit_map, month)
        if x is None:
            return None
        growth_src = _published_growth_src(vintages, published_growth)
        if isinstance(growth_src, Mapping):
            g = published_vintage(growth_src, month)
            if g is None:
                return None
        else:
            g = np.asarray(growth_src, dtype=float)
        return self.update(x, g, pi_e)

    def to_state(self) -> dict[str, Any]:
        return {"ee": np.asarray(self.ee, dtype=float).tolist(), "g_lr": np.asarray(self.g_lr, dtype=float).tolist()}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> EarningsExpectations:
        obj = cls(np.asarray(state["ee"], dtype=float))
        obj.g_lr = np.asarray(state["g_lr"], dtype=float).copy()
        return obj
